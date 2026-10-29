"""
Run three LLM judge checks on synthetic eval data.

For each entry in the eval JSON, runs:
  1. Answer correctness check
  2. Instruction following check
  3. Trajectory optimality check

Saves enriched results and prints a summary.

Usage:
    uv run python -m evals.synthetic.run_judge [options]

Examples:
    uv run python -m evals.synthetic.run_judge --limit 5
    uv run python -m evals.synthetic.run_judge --data evals/synthetic/data/eval_results_20250315_120000.json --limit 10
"""

import os
import json
import time
import asyncio
import argparse

from dotenv import load_dotenv

from cost_tracker import CostAccumulator
from evals.synthetic.judge import (
    create_correctness_judge,
    format_correctness_prompt,
    create_instruction_judge,
    format_instruction_prompt,
    create_trajectory_judge,
    format_trajectory_prompt,
    fix_instruction_user_context,
)
from evals.utils import map_progress, fmt_time

load_dotenv()


# ---------------------------------------------------------------------------
# Run all three checks on a single entry
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

def report(results: list[dict], cost: CostAccumulator, elapsed: float) -> None:
    total = len(results)

    checks = [
        ("Answer Correctness", "judge_answer_correctness"),
        ("Instruction Following", "judge_instruction_following"),
        ("Trajectory Optimality", "judge_trajectory"),
    ]

    print("\n" + "=" * 55)
    print("  JUDGE CHECK RESULTS")
    print("=" * 55)
    print(f"  Total entries evaluated: {total}")
    print("-" * 55)

    for label, key in checks:
        good = sum(1 for r in results if r.get(key, {}).get("score") == "good")
        bad = total - good
        pct = (good / total * 100) if total else 0
        print(f"  {label:<25s}  good: {good}  bad: {bad}  ({pct:.0f}% good)")

    print("-" * 55)
    print("  COST & TIME")
    print("-" * 55)
    print(f"  Model           : {cost.model}")
    print(f"  Tokens in/out   : {cost.input_tokens:,} / {cost.output_tokens:,}")
    print(f"  Total cost      : ${cost.total_cost:.4f}")
    print(f"  Total time      : {fmt_time(elapsed)}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run three LLM judge checks on synthetic eval data."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the synthetic eval JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <input>_judged.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only judge the first N entries.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of parallel judge calls (default: 4).",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)

    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(project_root, output_path)
    else:
        output_path = data_path.replace(".json", "_judged.json")

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        entries = json.load(f)
    print(f"  → {len(entries)} entries loaded.")

    if args.limit is not None:
        entries = entries[: args.limit]
        print(f"  → Limiting to first {len(entries)} entries (--limit {args.limit}).")

    # Create judges
    judges = {
        "correctness": create_correctness_judge(),
        "instruction": create_instruction_judge(),
        "trajectory": create_trajectory_judge(),
    }

    cost = CostAccumulator(model="openai:gpt-4o-mini")

    print(f"\nRunning judge checks with concurrency={args.concurrency}...")
    print("━" * 55)

    t0 = time.perf_counter()

    async def process(entry):
        return await judge_entry(entry, judges, cost)

    results = await map_progress(entries, process, max_concurrency=args.concurrency)
    elapsed = time.perf_counter() - t0

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    report(results, cost, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
