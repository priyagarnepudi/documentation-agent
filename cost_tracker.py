import json
import tempfile
from pathlib import Path
from dataclasses import dataclass


MODEL_PRICES = {
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
    "openai:gpt-5.2": {"input": 1.75, "output": 14.00},
    "anthropic:claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "anthropic:claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

COST_FILE = Path(tempfile.gettempdir()) / "pytest_cost_tracker.jsonl"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES.get(model.lower(), {"input": 0.0, "output": 0.0})
    return (input_tokens / 1_000_000) * prices["input"] + \
           (output_tokens / 1_000_000) * prices["output"]


@dataclass
class CostAccumulator:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage) -> None:
        self.input_tokens  += usage.input_tokens  or 0
        self.output_tokens += usage.output_tokens or 0

    @property
    def total_cost(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


def calculate_cost(model_name, input_tokens, output_tokens):
    return cost_usd(model_name, input_tokens, output_tokens)


def reset_cost_file():
    COST_FILE.unlink(missing_ok=True)


def capture_usage(model, result):
    usage = result.usage()
    entry = {
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    with open(COST_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def display_total_usage():
    print()

    if not COST_FILE.exists():
        print("Total cost: $0.000000")
        return

    totals = {}
    for line in COST_FILE.read_text().splitlines():
        entry = json.loads(line)
        model = entry["model"]
        if model not in totals:
            totals[model] = {"input_tokens": 0, "output_tokens": 0}
        totals[model]["input_tokens"] += entry["input_tokens"]
        totals[model]["output_tokens"] += entry["output_tokens"]

    total_cost = 0
    for model, tokens in totals.items():
        cost = cost_usd(model, tokens["input_tokens"], tokens["output_tokens"])
        print(f"{model}: ${cost:.6f}")
        total_cost += cost

    print(f"Total cost: ${total_cost:.6f}")
