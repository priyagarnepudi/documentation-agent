from functools import wraps

from pydantic_ai import Agent, RunUsage

from cost_tracker import CostAccumulator


def patch_agent(monkeypatch, totals):
    """Record usage for async Agent.run calls, including judge calls."""
    original_run = Agent.run

    @wraps(original_run)
    async def tracked_run(self, *args, **kwargs):
        usage = kwargs.get("usage")
        if usage is None:
            usage = kwargs["usage"] = RunUsage()
        before_input = usage.input_tokens or 0
        before_output = usage.output_tokens or 0
        try:
            return await original_run(self, *args, **kwargs)
        finally:
            model_name = f"{self.model.system}:{self.model.model_name}"
            total = totals.setdefault(model_name, CostAccumulator(model_name))
            total.input_tokens += (usage.input_tokens or 0) - before_input
            total.output_tokens += (usage.output_tokens or 0) - before_output

    monkeypatch.setattr(Agent, "run", tracked_run)
