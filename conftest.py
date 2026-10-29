from pathlib import Path

import pytest

from patch_agent import patch_agent


SQL_TEST_FILE = Path(__file__).with_name("test_agent.py")
SQL_USAGE_TOTALS = {}


@pytest.fixture(autouse=True)
def track_sql_homework_costs(request, monkeypatch):
    # The original documentation tests have their own usage tracking.
    if Path(request.node.path) == SQL_TEST_FILE:
        patch_agent(monkeypatch, SQL_USAGE_TOTALS)


def pytest_sessionstart(session):
    SQL_USAGE_TOTALS.clear()


def pytest_terminal_summary(terminalreporter):
    if not SQL_USAGE_TOTALS:
        return
    terminalreporter.write_sep("=", "SQL homework API usage (estimated)")
    for model, usage in SQL_USAGE_TOTALS.items():
        terminalreporter.write_line(
            f"{model}: {usage.input_tokens} input tokens, "
            f"{usage.output_tokens} output tokens; ${usage.total_cost:.6f}"
        )
    total = sum(usage.total_cost for usage in SQL_USAGE_TOTALS.values())
    terminalreporter.write_line(f"Total estimated cost: ${total:.6f}")
