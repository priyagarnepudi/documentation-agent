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
