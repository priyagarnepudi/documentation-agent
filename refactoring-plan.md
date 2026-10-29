# Refactoring Plan

## Target Structure

```
documentation-agent/
├── cost_tracker.py                    ← NEW (unified, replaces tests/cost_tracker.py)
├── app.py                            ← EDIT (import GITHUB_BASE from evals.utils)
├── doc_agent.py
├── main.py
├── models.py
├── tools.py
├── otel_test.py
│
├── evals/
│   ├── __init__.py                   ← NEW (empty)
│   ├── utils.py                      ← NEW (shared eval helpers)
│   │
│   ├── manual/
│   │   ├── __init__.py               ← NEW (empty)
│   │   ├── data/                     ← NEW directory
│   │   │   ├── questions.csv         ← MOVE from evals/
│   │   │   ├── evals_run_2026_03_06.json
│   │   │   └── evals_run_2026_03_06_results.json
│   │   ├── create_eval_data.ipynb    ← MOVE from evals/
│   │   ├── run_evals.py              ← MOVE + EDIT
│   │   ├── llm_judge.py              ← MOVE from evals/
│   │   ├── eval_judge.ipynb          ← MOVE from evals/
│   │   ├── evaluate_judge.py         ← MOVE + EDIT
│   │   └── label_evals.py            ← MOVE + EDIT
│   │
│   └── synthetic/
│       ├── __init__.py               ← NEW (empty)
│       ├── data/                     ← NEW directory
│       │   ├── questions_generated.csv
│       │   ├── evals_run_2026_03_16_synthetic.json
│       │   └── evals_run_2026_03_16_synthetic_judged.json
│       ├── synthetic_data_gen.ipynb  ← MOVE from evals/
│       ├── run_evals_synthetic.py    ← MOVE + EDIT
│       ├── run_judge_checks.py       ← MOVE + EDIT
│       ├── llm_judge_checks.py       ← MOVE + EDIT
│       └── explore_judge_results.py  ← MOVE + EDIT
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   ← EDIT (add agent fixture, update import)
│   ├── utils.py                      ← EDIT (import cost_tracker from root)
│   ├── judge.py                      ← EDIT (import cost_tracker from root)
│   ├── test_agent.py                 ← EDIT (remove duplicate agent fixture)
│   ├── test_judge.py                 ← EDIT (remove duplicate agent fixture)
│   └── (DELETE tests/cost_tracker.py)
│
└── trace_replay/
    └── ...
```

---

## Step 1: Create `cost_tracker.py` at project root

**Create** `cost_tracker.py` — merge `tests/cost_tracker.py` with the eval pattern.

Key decisions:
- **Unified `MODEL_PRICES`**: 5 models from `tests/cost_tracker.py` (superset of the 2 models in eval files)
- **Unified function**: `cost_usd()` uses `.get()` with zero-default (safe for unknown models); `calculate_cost()` kept as alias for backward compat
- **`CostAccumulator`**: single copy, used by both tests and evals
- All JSONL-based test tracking functions (`reset_cost_file`, `capture_usage`, `display_total_usage`) preserved

```python
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
```

---

## Step 2: Create `evals/utils.py`

**Create** `evals/utils.py` — shared helpers used across manual and synthetic evals.

Consolidates:
- `fmt_time()` — from `run_evals.py:194`, `run_evals_synthetic.py:155`, `run_judge_checks.py:165`
- `map_progress()` — from `run_evals_synthetic.py:78-96`, `run_judge_checks.py:81-99` (merged `desc` param)
- `collect_tools()` — from `tests/utils.py:22-37` (returns `list[dict]` for eval compat), plus inline loops in `run_evals.py:78-82` and `run_evals_synthetic.py:117-121`
- `GITHUB_BASE` — from `app.py:21`, `label_evals.py:19`, `explore_judge_results.py:29`

```python
import asyncio

from tqdm.auto import tqdm


GITHUB_BASE = "https://github.com/evidentlyai/docs/blob/main/"


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


async def map_progress(seq, func, max_concurrency=5, desc=None):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_semaphore(item):
        async with semaphore:
            return await func(item)

    coros = [run_with_semaphore(el) for el in seq]
    results = []
    tqdm_kwargs = {"total": len(coros)}
    if desc:
        tqdm_kwargs["desc"] = desc
    for coro in tqdm(asyncio.as_completed(coros), **tqdm_kwargs):
        result = await coro
        results.append(result)
    return results


def collect_tools(messages) -> list[dict]:
    tools = []
    for message in messages:
        for part in message.parts:
            if part.part_kind == "tool-call" and part.tool_name != "final_result":
                tools.append({"name": part.tool_name, "args": part.args})
    return tools
```

---

## Step 3: Create directory structure & move files

```bash
mkdir -p evals/manual/data evals/synthetic/data
touch evals/__init__.py evals/manual/__init__.py evals/synthetic/__init__.py
```

### File moves

| From | To |
|------|-----|
| `evals/questions.csv` | `evals/manual/data/questions.csv` |
| `evals/evals_run_2026_03_06.json` | `evals/manual/data/evals_run_2026_03_06.json` |
| `evals/evals_run_2026_03_06_results.json` | `evals/manual/data/evals_run_2026_03_06_results.json` |
| `evals/create_eval_data.ipynb` | `evals/manual/create_eval_data.ipynb` |
| `evals/run_evals.py` | `evals/manual/run_evals.py` |
| `evals/llm_judge.py` | `evals/manual/llm_judge.py` |
| `evals/eval_judge.ipynb` | `evals/manual/eval_judge.ipynb` |
| `evals/evaluate_judge.py` | `evals/manual/evaluate_judge.py` |
| `evals/label_evals.py` | `evals/manual/label_evals.py` |
| `evals/questions_generated.csv` | `evals/synthetic/data/questions_generated.csv` |
| `evals/evals_run_2026_03_16_synthetic.json` | `evals/synthetic/data/evals_run_2026_03_16_synthetic.json` |
| `evals/evals_run_2026_03_16_synthetic_judged.json` | `evals/synthetic/data/evals_run_2026_03_16_synthetic_judged.json` |
| `evals/synthetic_data_gen.ipynb` | `evals/synthetic/synthetic_data_gen.ipynb` |
| `evals/run_evals_synthetic.py` | `evals/synthetic/run_evals_synthetic.py` |
| `evals/run_judge_checks.py` | `evals/synthetic/run_judge_checks.py` |
| `evals/llm_judge_checks.py` | `evals/synthetic/llm_judge_checks.py` |
| `evals/explore_judge_results.py` | `evals/synthetic/explore_judge_results.py` |

---

## Step 4: Edit moved files — remove duplication, update imports

### 4a. `evals/manual/run_evals.py`

**Delete** (lines 21, 26-27): unused `field` import, `sys.path.append`
**Delete** (lines 36-64): entire cost section (`MODEL_PRICES`, `cost_usd`, `CostAccumulator`)
**Delete** (lines 78-82): inline tool collection loop
**Delete** (lines 194-196): inline `fmt_time`

**Add imports:**
```python
from cost_tracker import cost_usd, CostAccumulator
from evals.utils import fmt_time, collect_tools
```

**Replace** inline tool loop with:
```python
tools = collect_tools(result.new_messages())
```

**Update** `llm_judge` import:
```python
from evals.manual.llm_judge import create_log_judge_agent, format_judge_prompt
```

**Update** default `--questions` path:
```python
default="evals/manual/data/questions.csv",
help="Path to a CSV file with a 'question' column (default: evals/manual/data/questions.csv)",
```

**Update** docstring usage paths:
```
python evals/manual/run_evals.py
python evals/manual/run_evals.py --questions my.csv
python evals/manual/run_evals.py --output results.json
```

### 4b. `evals/synthetic/run_evals_synthetic.py`

**Delete** (lines 32-33): `sys.path.append`
**Delete** (lines 40-70): entire cost section (comment + `MODEL_PRICES` + `cost_usd` + `CostAccumulator`)
**Delete** (lines 73-96): entire `map_progress` function + attribution comment
**Delete** (lines 117-121): inline tool collection loop
**Delete** (lines 155-157): inline `fmt_time`

**Add imports:**
```python
from cost_tracker import cost_usd, CostAccumulator
from evals.utils import map_progress, fmt_time, collect_tools
```

**Replace** inline tool loop with:
```python
tools = collect_tools(result.new_messages())
```

**Update** default `--questions` path:
```python
default="evals/synthetic/data/questions_generated.csv",
help="Path to questions_generated.csv (default: evals/synthetic/data/questions_generated.csv)",
```

**Update** default output path construction:
```python
output_path = os.path.join(
    project_root, "evals", "synthetic", "data", f"evals_run_{today}_synthetic.json"
)
```

**Update** docstring usage paths:
```
evals/synthetic/evals_run_2026_03_16_synthetic.json
python evals/synthetic/run_evals_synthetic.py
python evals/synthetic/run_evals_synthetic.py --limit 10
python evals/synthetic/run_evals_synthetic.py --concurrency 3
python evals/synthetic/run_evals_synthetic.py --output my_results.json
```

### 4c. `evals/synthetic/run_judge_checks.py`

**Delete** (line 29): `sys.path.append`
**Delete** (lines 43-73): entire cost section
**Delete** (lines 76-99): entire `map_progress` function + attribution comment
**Delete** (lines 165-167): inline `fmt_time`

**Add imports:**
```python
from cost_tracker import cost_usd, CostAccumulator
from evals.utils import map_progress, fmt_time
```

**Update** `llm_judge_checks` import:
```python
from evals.synthetic.llm_judge_checks import (
    create_correctness_judge, format_correctness_prompt,
    create_instruction_judge, format_instruction_prompt,
    create_trajectory_judge, format_trajectory_prompt,
)
```

**Update** default `--data` path:
```python
default="evals/synthetic/data/evals_run_2026_03_16_synthetic.json",
```

**Update** docstring usage paths:
```
python evals/synthetic/run_judge_checks.py --data evals/synthetic/data/evals_run_2026_03_16_synthetic.json
python evals/synthetic/run_judge_checks.py --data evals/synthetic/data/evals_run_2026_03_16_synthetic.json --limit 5
python evals/synthetic/run_judge_checks.py --data evals/synthetic/data/evals_run_2026_03_16_synthetic.json --concurrency 10
```

### 4d. `evals/manual/llm_judge.py`

No changes needed. Pure definitions, no imports from evals/, no `sys.path`, no cost code.

### 4e. `evals/synthetic/llm_judge_checks.py`

**Delete** (lines 17-18): `sys.path.append` block

No other changes needed. `pythonpath = ["."]` in `pyproject.toml` handles the `from doc_agent import DEFAULT_INSTRUCTIONS` import.

### 4f. `evals/manual/evaluate_judge.py`

**Delete** (lines 1-9): `os`, `sys`, `sys.path.append`, `load_dotenv` scattered imports
**Delete** (line 13): `load_dotenv()` call
**Delete** (line 15): `import re` (unused)
**Delete** (line 23): `import argparse` inside function body
**Delete** (line 119): commented-out `df.to_csv`

**Replace top-level imports with:**
```python
import argparse
import asyncio
import json
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from evals.manual.llm_judge import create_log_judge_agent, format_judge_prompt
```

**Update** default `--data` path:
```python
default="evals/manual/data/evals_run_2026_03_06_results.json",
```

**Fix** `type(iterable) is list` hack (line 50) — use a flag variable:
```python
use_tqdm = False
try:
    from tqdm.asyncio import tqdm
    use_tqdm = True
except ImportError:
    pass

if use_tqdm:
    iterable = tqdm(results, desc="Evaluating")
else:
    iterable = results

for i, row in enumerate(iterable):
    if not use_tqdm:
        print(f"Evaluating {i+1}/{len(results)}...")
```

### 4g. `evals/manual/label_evals.py`

**Delete** (lines 8-17): `sys.path` manipulation + `try/except ImportError` block

**Add imports:**
```python
from doc_agent import DEFAULT_INSTRUCTIONS
from evals.utils import GITHUB_BASE
```

**Add** `data_dir` definition and update `glob.glob` path (line 78):
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
```

### 4h. `evals/synthetic/explore_judge_results.py`

**Delete** (lines 18-27): `sys.path` manipulation + `try/except ImportError` block

**Add imports:**
```python
from doc_agent import DEFAULT_INSTRUCTIONS
from evals.utils import GITHUB_BASE
```

**Add** `data_dir` definition and update `glob.glob` path (line 63):
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
json_files = sorted(glob.glob(os.path.join(data_dir, "*_judged.json")))
```

**Update** docstring usage (line 9):
```
streamlit run evals/synthetic/explore_judge_results.py
```

---

## Step 5: Edit `app.py`

**Delete** line 21:
```python
GITHUB_BASE = "https://github.com/evidentlyai/docs/blob/main/"
```

**Add import:**
```python
from evals.utils import GITHUB_BASE
```

---

## Step 6: Edit `tests/` files

### 6a. Delete `tests/cost_tracker.py`

Entire file replaced by root-level `cost_tracker.py`.

### 6b. Edit `tests/conftest.py`

**Update** import path:
```python
from cost_tracker import display_total_usage, reset_cost_file
```

**Add** shared agent fixture (consolidated from `test_agent.py` and `test_judge.py`):
```python
import pytest
from time import time

from tools import create_documentation_tools_cached
from doc_agent import create_agent, DocumentationAgentConfig


@pytest.fixture(scope="module")
def agent():
    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig()
    return create_agent(agent_config, tools)
```

### 6c. Edit `tests/utils.py`

**Update** import (line 13):
```python
from cost_tracker import capture_usage
```

### 6d. Edit `tests/judge.py`

**Update** import (line 5):
```python
from cost_tracker import capture_usage
```

### 6e. Edit `tests/test_agent.py`

**Delete** imports only used by the fixture:
```python
from time import time
from tools import create_documentation_tools_cached
from doc_agent import create_agent, DocumentationAgentConfig
```

**Delete** the `agent` fixture (lines 11-23) — now in `conftest.py`.

**Final imports:**
```python
import pytest
from tests.utils import collect_tools, run_agent_test
```

### 6f. Edit `tests/test_judge.py`

**Delete** imports only used by the fixture:
```python
from tools import create_documentation_tools_cached
from doc_agent import create_agent, DocumentationAgentConfig
```

**Delete** the `agent` fixture (lines 10-14) — now in `conftest.py`.

**Final imports:**
```python
import pytest
from tests.utils import run_agent_test
from tests.judge import assert_criteria
```

---

## Step 7: Verification

```bash
ruff check .
pytest --co
```

---

## Summary of changes by file

| File | Action | Lines removed |
|------|--------|--------------|
| `cost_tracker.py` (root) | **CREATE** | — |
| `evals/__init__.py` | **CREATE** | — |
| `evals/utils.py` | **CREATE** | — |
| `evals/manual/__init__.py` | **CREATE** | — |
| `evals/manual/data/` | **CREATE** | — |
| `evals/synthetic/__init__.py` | **CREATE** | — |
| `evals/synthetic/data/` | **CREATE** | — |
| `evals/manual/run_evals.py` | **MOVE + EDIT** | ~30 |
| `evals/manual/evaluate_judge.py` | **MOVE + EDIT** | ~10 |
| `evals/manual/label_evals.py` | **MOVE + EDIT** | ~10 |
| `evals/synthetic/run_evals_synthetic.py` | **MOVE + EDIT** | ~60 |
| `evals/synthetic/run_judge_checks.py` | **MOVE + EDIT** | ~55 |
| `evals/synthetic/llm_judge_checks.py` | **MOVE + EDIT** | ~2 |
| `evals/synthetic/explore_judge_results.py` | **MOVE + EDIT** | ~10 |
| `evals/manual/llm_judge.py` | **MOVE** | 0 |
| `evals/manual/create_eval_data.ipynb` | **MOVE** | 0 |
| `evals/manual/eval_judge.ipynb` | **MOVE** | 0 |
| `evals/synthetic/synthetic_data_gen.ipynb` | **MOVE** | 0 |
| `app.py` | **EDIT** | ~1 |
| `tests/conftest.py` | **EDIT** | ~0 (net) |
| `tests/utils.py` | **EDIT** | ~0 |
| `tests/judge.py` | **EDIT** | ~0 |
| `tests/test_agent.py` | **EDIT** | ~14 |
| `tests/test_judge.py` | **EDIT** | ~5 |
| `tests/cost_tracker.py` | **DELETE** | 64 |

**Net result**: ~150 lines of duplication removed, 1 source of truth for costs, clean folder structure.
