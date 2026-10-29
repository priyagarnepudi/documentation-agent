# Monitoring & Evaluation Pipeline

This directory contains the evaluation pipeline for the documentation agent, split into manual (human-in-the-loop) and synthetic (automated) workflows.

## Directory Structure

```text
evals/
├── manual/           # Evaluations using human-labeled ground truth
│   ├── data/         # CSV questions and JSON results
│   ├── run.py        # End-to-end runner (Agent + Judge)
│   ├── run_judge.py  # Judge performance benchmarking
│   ├── judge.py      # LLM judge implementation for manual logs
│   └── app.py        # Streamlit app for labeling and review
├── synthetic/        # Automated evaluations using generated data
│   ├── data/         # Generated questions and agent results
│   ├── run.py        # End-to-end runner (Agent + three Judges)
│   ├── run_judge.py  # Standalone judge runner for existing results
│   ├── judge.py      # Implementation for the three synthetic judges
│   └── app.py        # Streamlit app for exploring results
└── utils.py          # Shared evaluation helpers (cost, formatting, etc.)
```

## Usage

All scripts should be run as Python modules from the project root using `uv run python -m`.

### Manual Evaluations
Used for benchmarking the agent against high-quality, human-curated questions.

```bash
# Run agent and judge on manual questions
uv run python -m evals.manual.run --limit 5

# Start the labeling and review app
uv run streamlit run evals/manual/app.py
```

### Synthetic Evaluations
Used for broad coverage and automated testing using generated questions.

```bash
# Run agent and all three judges on synthetic questions
uv run python -m evals.synthetic.run --limit 10

# Explore judge results and agent trajectories
uv run streamlit run evals.synthetic.app
```

## Shared Logic
- **`utils.py`**: Common functions for progress bars, time formatting, and tool collection.
- **Root `cost_tracker.py`**: Centralized logic for tracking LLM costs across all evaluation phases.
