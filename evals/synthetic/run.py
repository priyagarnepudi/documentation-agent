"""
Synthetic eval runner.

Runs the documentation agent on all questions from questions_generated.csv
in parallel (default concurrency=5), and saves the results to a dated JSON
file with a '_synthetic' suffix, e.g.:

    evals/evals_run_2026_03_16_synthetic.json

Usage:
    uv run python -m evals.synthetic.run [options]

Examples:
    uv run python -m evals.synthetic.run --limit 5
    uv run python -m evals.synthetic.run --questions evals/synthetic/data/questions.csv --limit 10
"""

import os
import json
import time
import random
import asyncio
import argparse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from cost_tracker import cost_usd, CostAccumulator
from doc_agent import DocumentationAgentConfig, create_agent, run_agent
from evals.synthetic.judge import (
    create_correctness_judge,
    format_correctness_prompt,
    create_instruction_judge,
    format_instruction_prompt,
    create_trajectory_judge,
    format_trajectory_prompt,
    fix_instruction_user_context,
)
from evals.utils import map_progress, fmt_time, collect_tools
from tools import create_documentation_tools_cached

load_dotenv()


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def make_agent():
    """Create a fresh agent (search tools + config)."""
    search_tools = create_documentation_tools_cached()
    config = DocumentationAgentConfig()
    return create_agent(config, search_tools)


async def run_agent_on_row(agent, row: dict, cost: CostAccumulator) -> dict:
    """Run the agent on a single row from questions_generated.csv."""
    question = row["question"]
    try:
        result = await run_agent(agent, question)
        rag_response = result.output.model_dump()

        tools = collect_tools(result.new_messages())

        usage = result.usage()
        cost.add(usage)
        q_cost = cost_usd(
            cost.model,
            usage.input_tokens or 0,
            usage.output_tokens or 0,
        )

        return {
            "input": row,
            "rag_response": rag_response,
            "tools": tools,
            "cost": q_cost,
        }
    except Exception as exc:
        return {
            "input": row,
            "rag_response": {"answer": f"Agent error: {exc}"},
            "tools": [],
            "cost": 0.0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Judge runner
# ---------------------------------------------------------------------------

async def judge_entry(entry: dict, judges: dict, cost: CostAccumulator) -> dict:
    """Run three judge checks on one eval entry and return enriched entry."""

    # Check 1: Answer correctness
    try:
        correctness = await judges["correctness"].run(
            format_correctness_prompt(entry)
        )
        cost.add(correctness.usage())
        entry["judge_answer_correctness"] = correctness.output.model_dump()
    except Exception as e:
        entry["judge_answer_correctness"] = {
            "reasoning": f"Error: {e}",
            "score": "bad",
        }

    # Check 2: Instruction following
    try:
        instruction = await judges["instruction"].run(
            format_instruction_prompt(entry)
        )
        cost.add(instruction.usage())
        entry["judge_instruction_following"] = instruction.output.model_dump()
    except Exception as e:
        entry["judge_instruction_following"] = {
            "reasoning": f"Error: {e}",
            "score": "bad",
        }
    fix_instruction_user_context(entry)

    # Check 3: Trajectory optimality
    try:
        trajectory = await judges["trajectory"].run(
            format_trajectory_prompt(entry)
        )
        cost.add(trajectory.usage())
        entry["judge_trajectory"] = trajectory.output.model_dump()
    except Exception as e:
        entry["judge_trajectory"] = {
            "reasoning": f"Error: {e}",
            "score": "bad",
            "suggestion": "",
        }

    return entry


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(results: list[dict], agent_cost: CostAccumulator, judge_cost: CostAccumulator, elapsed: float) -> None:
    total = len(results)
    errors = sum(1 for r in results if "error" in r)

    checks = [
        ("Answer Correctness", "judge_answer_correctness"),
        ("Instruction Following", "judge_instruction_following"),
        ("Trajectory Optimality", "judge_trajectory"),
    ]

    print("\n" + "=" * 55)
    print("  SYNTHETIC EVAL RESULTS")
    print("=" * 55)
    print(f"  Total questions : {total}")
    print(f"  Agent errors    : {errors}")
    print("-" * 55)

    for label, key in checks:
        good = sum(1 for r in results if r.get(key, {}).get("score") == "good")
        bad = total - good
        pct = (good / total * 100) if total else 0
        print(f"  {label:<25s}  good: {good}  bad: {bad}  ({pct:.0f}% good)")

    print("-" * 55)
    print("  COST & TIME")
    print("-" * 55)
    print(f"  Agent cost      : ${agent_cost.total_cost:.4f}")
    print(f"  Judge cost      : ${judge_cost.total_cost:.4f}")
    print(f"  Total cost      : ${agent_cost.total_cost + judge_cost.total_cost:.4f}")
    print(f"  Total time      : {fmt_time(elapsed)}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the documentation agent on synthetic questions in parallel."
    )
    parser.add_argument(
        "--questions",
        default="evals/synthetic/data/questions_generated.csv",
        help="Path to questions_generated.csv (default: evals/synthetic/data/questions_generated.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to evals/evals_run_<YYYY_MM_DD>_synthetic.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run on a random subset of N questions instead of the full list.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of parallel agent calls (default: 5).",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    questions_path = args.questions
    if not os.path.isabs(questions_path):
        questions_path = os.path.join(project_root, questions_path)

    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(project_root, output_path)
    else:
        today = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        output_path = os.path.join(
            project_root, "evals", "synthetic", "data", f"evals_run_{today}_synthetic.json"
        )

    # Load questions
    print(f"Loading questions from {questions_path}...")
    df = pd.read_csv(questions_path)
    # Fill NaN so JSON serialisation works cleanly
    df = df.where(pd.notna(df), None)
    rows = df.to_dict(orient="records")
    print(f"  → {len(rows)} questions loaded.")

    if args.limit is not None:
        k = min(args.limit, len(rows))
        rows = random.sample(rows, k)
        print(f"  → Sampling {k} random questions (--limit {args.limit}).")

    # Create shared agent tools and cost accumulators
    search_tools = create_documentation_tools_cached()
    agent_cost = CostAccumulator(model="openai:gpt-4o-mini")
    judge_cost = CostAccumulator(model="openai:gpt-4o-mini")

    t_start = time.perf_counter()

    # --- Phase 1: Agent ---
    print(f"\nPhase 1: Running {len(rows)} questions with concurrency={args.concurrency}...")
    print("━" * 55)

    async def process_agent(row):
        agent = create_agent(DocumentationAgentConfig(), search_tools)
        return await run_agent_on_row(agent, row, agent_cost)

    agent_results = await map_progress(rows, process_agent, max_concurrency=args.concurrency)

    # --- Phase 2: Judge ---
    print(f"\nPhase 2: Running judge checks with concurrency={args.concurrency}...")
    print("━" * 55)

    judges = {
        "correctness": create_correctness_judge(),
        "instruction": create_instruction_judge(),
        "trajectory": create_trajectory_judge(),
    }

    async def process_judge(entry):
        return await judge_entry(entry, judges, judge_cost)

    results = await map_progress(agent_results, process_judge, max_concurrency=args.concurrency)
    elapsed = time.perf_counter() - t_start

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    report(results, agent_cost, judge_cost, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
