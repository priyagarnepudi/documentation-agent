"""
End-to-end evaluation pipeline.

Runs the documentation agent on a set of questions, applies the LLM judge to
each response, and reports how many answers are good/bad in absolute numbers
and as a percentage.

Usage:
    uv run python -m evals.manual.run [options]

Examples:
    uv run python -m evals.manual.run --limit 5
    uv run python -m evals.manual.run --questions evals/manual/data/questions.csv --limit 10
"""

import os
import json
import time
import random
import asyncio
import argparse

import pandas as pd
from dotenv import load_dotenv

from cost_tracker import cost_usd, CostAccumulator
from doc_agent import DocumentationAgentConfig, create_agent, run_agent
from evals.manual.judge import create_log_judge_agent, format_judge_prompt
from evals.utils import collect_tools, fmt_time
from tools import create_documentation_tools_cached

load_dotenv()


# ---------------------------------------------------------------------------
# Step 1 – run the agent on a list of questions
# ---------------------------------------------------------------------------

async def run_agent_on_question(agent, question: str) -> tuple[dict, object]:
    """Run the documentation agent and return (result_dict, usage)."""
    result = await run_agent(agent, question)
    rag_response = result.output.model_dump()

    # Collect tool calls from the message history (skip pydantic-ai's internal
    # 'final_result' call — only keep real search/get_file tool calls)
    tools = collect_tools(result.new_messages())

    entry = {
        "input": {"question": question},
        "rag_response": rag_response,
        "tools": tools,
    }
    return entry, result.usage()


async def run_agent_on_all_questions(
    questions: list[str],
    cost: CostAccumulator,
) -> list[dict]:
    search_tools = create_documentation_tools_cached()
    config = DocumentationAgentConfig()
    agent = create_agent(config, search_tools)

    results = []
    total = len(questions)
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{total}] Question: {question}")
        try:
            t0 = time.perf_counter()
            result, usage = await run_agent_on_question(agent, question)
            elapsed = time.perf_counter() - t0
            cost.add(usage)
            q_cost = cost_usd(
                cost.model,
                usage.input_tokens or 0,
                usage.output_tokens or 0,
            )
            print(f"  → cost: ${q_cost:.4f}  time: {elapsed:.1f}s  "
                  f"tokens: {usage.input_tokens or 0:,} in / {usage.output_tokens or 0:,} out")
        except Exception as exc:
            print(f"  ERROR running agent: {exc}")
            result = {
                "input": {"question": question},
                "rag_response": {"answer": f"Agent error: {exc}"},
                "tools": [],
            }
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Step 2 – apply the LLM judge
# ---------------------------------------------------------------------------

async def judge_result(judge, entry: dict) -> tuple[dict, object]:
    prompt = format_judge_prompt(entry)
    eval_result = await judge.run(prompt)
    verdict = {
        "label": eval_result.output.label,
        "reasoning": eval_result.output.reasoning,
    }
    return verdict, eval_result.usage()


async def judge_all_results(
    agent_results: list[dict],
    cost: CostAccumulator,
) -> list[dict]:
    judge = create_log_judge_agent()
    judged = []
    total = len(agent_results)
    for i, entry in enumerate(agent_results, 1):
        q = entry["input"]["question"]
        print(f"\n[{i}/{total}] Judging: {q}")
        try:
            t0 = time.perf_counter()
            verdict, usage = await judge_result(judge, entry)
            elapsed = time.perf_counter() - t0
            cost.add(usage)
            q_cost = cost_usd(
                cost.model,
                usage.input_tokens or 0,
                usage.output_tokens or 0,
            )
            print(f"  → [{verdict['label'].upper()}]  "
                  f"cost: ${q_cost:.4f}  time: {elapsed:.1f}s  "
                  f"tokens: {usage.input_tokens or 0:,} in / {usage.output_tokens or 0:,} out")
        except Exception as exc:
            print(f"  ERROR judging: {exc}")
            verdict = {"label": "bad", "reasoning": f"Judge error: {exc}"}
        judged.append({**entry, **verdict})

    return judged


# ---------------------------------------------------------------------------
# Step 3 – report
# ---------------------------------------------------------------------------

def report(
    judged_results: list[dict],
    agent_cost: CostAccumulator,
    judge_cost: CostAccumulator,
    agent_elapsed: float,
    judge_elapsed: float,
) -> None:
    total = len(judged_results)
    good = sum(1 for r in judged_results if r["label"] == "good")
    bad = total - good

    pct_good = good / total * 100 if total else 0.0
    pct_bad  = bad  / total * 100 if total else 0.0

    total_cost    = agent_cost.total_cost + judge_cost.total_cost
    total_elapsed = agent_elapsed + judge_elapsed

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Total questions : {total}")
    print(f"  Good            : {good:>4}  ({pct_good:.1f}%)")
    print(f"  Bad             : {bad:>4}  ({pct_bad:.1f}%)")
    print("=" * 55)
    print("  COST & TIME BREAKDOWN")
    print("=" * 55)
    print(f"  Agent ({agent_cost.model}):")
    print(f"    Tokens in/out : {agent_cost.input_tokens:,} / {agent_cost.output_tokens:,}")
    print(f"    Cost          : ${agent_cost.total_cost:.4f}")
    print(f"    Time          : {fmt_time(agent_elapsed)}")
    print(f"  Judge ({judge_cost.model}):")
    print(f"    Tokens in/out : {judge_cost.input_tokens:,} / {judge_cost.output_tokens:,}")
    print(f"    Cost          : ${judge_cost.total_cost:.4f}")
    print(f"    Time          : {fmt_time(judge_elapsed)}")
    print(f"  {'─' * 37}")
    print(f"  Total cost      : ${total_cost:.4f}")
    print(f"  Total time      : {fmt_time(total_elapsed)}")
    print("=" * 55)

    # Per-question detail
    print("\nDetailed breakdown:")
    for r in judged_results:
        q       = r["input"]["question"]
        label   = r["label"].upper()
        reason  = r.get("reasoning", "")[:120]
        marker  = "✓" if r["label"] == "good" else "✗"
        print(f"\n  {marker} [{label}] {q}")
        if reason:
            print(f"    {reason}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the documentation agent on all questions and judge the responses."
    )
    parser.add_argument(
        "--questions",
        default="evals/manual/data/questions.csv",
        help="Path to a CSV file with a 'question' column (default: evals/manual/data/questions.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the raw judged results as JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run on a random subset of N questions instead of the full list.",
    )
    args = parser.parse_args()

    # Resolve path relative to project root when running from anywhere
    questions_path = args.questions
    if not os.path.isabs(questions_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        questions_path = os.path.join(project_root, questions_path)

    print(f"Loading questions from {questions_path}...")
    df_q = pd.read_csv(questions_path)
    questions = df_q["question"].tolist()
    print(f"  → {len(questions)} questions loaded.")

    if args.limit is not None:
        k = min(args.limit, len(questions))
        questions = random.sample(questions, k)
        print(f"  → Sampling {k} random questions (--limit {args.limit}).")

    # --- Agent run ---
    agent_cost = CostAccumulator(model="openai:gpt-4o-mini")
    print("\n" + "━" * 55)
    print("  PHASE 1: Running documentation agent")
    print("━" * 55)
    t0 = time.perf_counter()
    agent_results = await run_agent_on_all_questions(questions, agent_cost)
    agent_elapsed = time.perf_counter() - t0

    # --- Judge ---
    judge_cost = CostAccumulator(model="openai:gpt-4o-mini")
    print("\n" + "━" * 55)
    print("  PHASE 2: Applying LLM judge")
    print("━" * 55)
    t0 = time.perf_counter()
    judged_results = await judge_all_results(agent_results, judge_cost)
    judge_elapsed = time.perf_counter() - t0

    # --- Save raw results if requested ---
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.getcwd(), output_path)
        with open(output_path, "w") as f:
            json.dump(judged_results, f, indent=2)
        print(f"\nRaw results saved to {output_path}")

    # --- Report ---
    report(judged_results, agent_cost, judge_cost, agent_elapsed, judge_elapsed)


if __name__ == "__main__":
    asyncio.run(main())
