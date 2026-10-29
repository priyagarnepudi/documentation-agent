"""
Three LLM judge checks for evaluating the documentation agent:

1. Answer Correctness  – does the agent's answer contain the reference answer?
2. Instruction Following – does the agent follow its system-prompt instructions?
3. Trajectory Optimality – was the tool-call sequence efficient?
"""

import json as _json
from typing import Literal, Any, Dict

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from doc_agent import DEFAULT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class CorrectnessResult(BaseModel):
    """Result of the answer-correctness check."""

    reasoning: str = Field(
        description="Step-by-step reasoning about whether the agent's answer covers the reference answer."
    )
    score: Literal["good", "bad"] = Field(
        description="'good' if the answer semantically contains the reference answer, 'bad' otherwise."
    )


class InstructionFollowingResult(BaseModel):
    """Result of the instruction-following check."""

    reasoning: str = Field(
        description="Step-by-step reasoning about whether the agent followed its instructions."
    )
    score: Literal["good", "bad"] = Field(
        description="'good' if the agent followed the instructions, 'bad' otherwise."
    )


class TrajectoryResult(BaseModel):
    """Result of the trajectory-optimality check."""

    reasoning: str = Field(
        description="Step-by-step reasoning about the optimality of the tool-call trajectory."
    )
    score: Literal["good", "bad"] = Field(
        description="'good' if the trajectory was reasonably optimal, 'bad' if clearly wasteful."
    )
    suggestion: str = Field(
        description="Concrete suggestion for how the agent could have been more efficient, or 'none' if the trajectory was already optimal."
    )


# ---------------------------------------------------------------------------
# Check 1 – Answer correctness
# ---------------------------------------------------------------------------

CORRECTNESS_INSTRUCTIONS = """
You are an expert evaluator. Your task is to decide whether an AI agent's
answer CORRECTLY ADDRESSES the same topic as the reference answer.

Rules:
- The reference answer may be a SHORT SNIPPET extracted from documentation,
  not a complete answer. Do not expect the agent to reproduce the snippet
  verbatim — instead, check whether the agent's answer covers the same
  CONCEPT or TOPIC that the reference answer addresses.
- The agent's answer does NOT need to be word-for-word identical. It must
  convey the same key information or provide an equivalent correct answer.
- If the reference answer mentions a specific API, class, or method, the
  agent should mention the same or an equivalent one. However, if the agent
  provides a correct alternative approach to solve the same problem, that
  also counts as "good".
- If the agent's answer provides additional correct information beyond the
  reference, that is fine — it should still be marked "good".
- If the agent gives a comprehensive answer that covers the reference
  answer's topic correctly but uses different wording or structure, mark
  it "good".
- Mark "bad" ONLY if the agent's answer is fundamentally wrong, off-topic,
  or completely misses the core concept from the reference answer.

Be fair: focus on whether the agent correctly answered the user's question
on the same topic as the reference answer, not on exact textual overlap.
""".strip()

CORRECTNESS_PROMPT = """
User Question:
{question}

Reference Answer (ground truth):
{reference_answer}

Agent's Answer:
{agent_answer}
""".strip()


def create_correctness_judge() -> Agent:
    return Agent(
        name="correctness_judge",
        model="openai:gpt-4o-mini",
        instructions=CORRECTNESS_INSTRUCTIONS,
        output_type=CorrectnessResult,
    )


def format_correctness_prompt(entry: Dict[str, Any]) -> str:
    return CORRECTNESS_PROMPT.format(
        question=entry["input"]["question"],
        reference_answer=entry["input"]["reference_answer"],
        agent_answer=entry["rag_response"]["answer"],
    )


# ---------------------------------------------------------------------------
# Check 2 – Instruction following
# ---------------------------------------------------------------------------

INSTRUCTION_FOLLOWING_INSTRUCTIONS = """
You are an expert evaluator. You will be given:
1. The system-prompt instructions that were given to a documentation agent.
2. A user question the agent received.
3. The agent's response.
4. The agent's self-check results (if any).

Your task is to verify whether the agent's FINAL ANSWER followed its
instructions. Read the agent's instructions carefully and check whether
the answer violates any of them.

You are judging the output quality, NOT the internal process.

Guidelines for evaluation:
- Read the agent instructions provided below and identify the rules the
  agent was supposed to follow (e.g., formatting rules, package manager
  usage, code style, follow-up question requirements, etc.).
- Check the agent's answer against each applicable rule.
- Conditional rules (like "use the user's variable names when provided")
  only apply when the condition is met. Check the USER CONTEXT PRESENT
  field — if it says NO, do not penalize for using generic variable names.

Do NOT evaluate these:
- Internal tool calls or search process — you only see the final answer.
- Answer completeness or thoroughness — that is not your job.

Mark "good" if the agent followed its instructions well overall.
Mark "bad" only if the agent clearly violated one or more rules from
its instructions.
""".strip()

INSTRUCTION_FOLLOWING_PROMPT = """
=== AGENT INSTRUCTIONS (system prompt) ===
{instructions}

=== USER QUESTION ===
{question}

=== USER CONTEXT PRESENT ===
{has_user_context}

=== AGENT ANSWER ===
{agent_answer}

=== AGENT SELF-CHECKS ===
{self_checks}
""".strip()


def create_instruction_judge() -> Agent:
    return Agent(
        name="instruction_judge",
        model="openai:gpt-4o-mini",
        instructions=INSTRUCTION_FOLLOWING_INSTRUCTIONS,
        output_type=InstructionFollowingResult,
    )


def _has_user_context(question: str) -> bool:
    """Check if the question contains user-specific context like variable names or code."""
    code_indicators = ["```", "my_", "df.", "df[", "dataset.", "import ", ".run("]
    return any(ind in question for ind in code_indicators)


def fix_instruction_user_context(entry: Dict[str, Any]) -> None:
    """Post-process instruction-following results.

    gpt-4o-mini sometimes penalizes for not using 'user context' even when
    the question has none. If the judge marked 'bad' primarily because of
    user context and the question has no user context, override to 'good'.
    """
    result = entry.get("judge_instruction_following", {})
    if result.get("score") != "bad":
        return

    question = entry["input"]["question"]
    if _has_user_context(question):
        return  # question has context — penalty may be valid

    reasoning = result.get("reasoning", "").lower()
    context_phrases = [
        "user context", "variable name", "user-specific",
        "user's specific", "generic variable", "generic example",
    ]
    if any(phrase in reasoning for phrase in context_phrases):
        result["score"] = "good"
        result["reasoning"] += (
            " [AUTO-CORRECTED: Question has no user context, "
            "so user-context penalty does not apply.]"
        )


def format_instruction_prompt(entry: Dict[str, Any]) -> str:
    checks = entry["rag_response"].get("checks", [])
    checks_str = "\n".join(
        f"- [{'+' if c['passed'] else '-'}] {c['rule']}: {c['explanation']}"
        for c in checks
    )
    question = entry["input"]["question"]
    has_context = _has_user_context(question)
    context_str = (
        "YES — the user provided specific variables, code, or setup details. "
        "The agent should use the user's names and context."
        if has_context
        else "NO — the user asked a general question without providing specific "
        "variables, code, or setup. The 'Adapting to user context' rule does NOT apply. "
        "Do NOT penalize for using generic variable names."
    )
    return INSTRUCTION_FOLLOWING_PROMPT.format(
        instructions=DEFAULT_INSTRUCTIONS,
        question=question,
        has_user_context=context_str,
        agent_answer=entry["rag_response"]["answer"],
        self_checks=checks_str or "(none)",
    )


# ---------------------------------------------------------------------------
# Check 3 – Trajectory optimality
# ---------------------------------------------------------------------------

TRAJECTORY_INSTRUCTIONS = """
You are an expert evaluator. You will be given:
1. A user question.
2. The sequence of tool calls (the "trajectory") the agent made.
3. The agent's final answer.

The agent has two tools:
- search(query) — searches the documentation index and returns matching filenames.
- get_file(filename) — retrieves the full content of a specific documentation file.

The agent's instructions prescribe this workflow:
  Iteration 1: one search call.
  Iteration 2: up to 2 MORE searches + one or more get_file calls.
  Iteration 3: synthesize the answer (no tool calls).

This means a NORMAL trajectory is 1-3 search calls + 1-3 get_file calls,
totaling 2-6 tool calls. This is expected and correct behavior.

Evaluate the trajectory for optimality:
- Were the search queries relevant and well-formulated?
- Did the agent retrieve files that were relevant to the question?
- Were there DUPLICATE tool calls? (e.g. calling get_file on the exact same
  file twice is clearly bad)
- Were searches clearly irrelevant to the question?
- More than 4 search calls is excessive. More than 5 get_file calls is excessive.
- Retrieving 2-3 related files to give a comprehensive answer is GOOD, not bad.

IMPORTANT: You are evaluating the TOOL CALL TRAJECTORY only — whether the
sequence of searches and file retrievals was efficient. Do NOT evaluate the
quality or correctness of the final answer. That is a separate check.

Mark "good" if the trajectory was reasonably efficient — the tool calls were
relevant to the question and there were no duplicates or clearly wasted calls.
A trajectory of 2-6 tool calls with no duplicates and relevant queries is
GOOD by default.
Mark "bad" ONLY if there are CLEAR, OBJECTIVE inefficiencies:
  - get_file called on the EXACT SAME filename more than once
  - search queries completely unrelated to the question
  - truly excessive tool use (more than 8 total calls)
Do NOT mark bad just because you think fewer calls could have sufficed.
For example, these are all GOOD trajectories:
  - 1 search + 2 get_file (3 calls, no dupes) = GOOD
  - 2 searches + 3 get_file on different files (5 calls) = GOOD
  - 3 searches + 2 get_file on different files (5 calls) = GOOD
  - 1 search + 1 get_file (2 calls) = GOOD
Retrieving multiple related files to give a comprehensive answer is expected.

These are BAD trajectories:
  - search + get_file("a.mdx") + get_file("b.mdx") + get_file("a.mdx") = BAD (duplicate file)
  - search + get_file("a.mdx") + get_file("b.mdx") + get_file("a.mdx") + get_file("b.mdx") = BAD (loop: fetching same files repeatedly)
  - search + get_file("a.mdx") + search + get_file("a.mdx") + search + get_file("a.mdx") = BAD (loop: repeating the same search-fetch cycle)
  - 4 searches + 5 get_file calls (9 calls total) = BAD (excessive)
  - search("weather forecast") for a question about data drift = BAD (irrelevant)
Provide a concrete suggestion for improvement, or "none" if the trajectory
was reasonable.
""".strip()

TRAJECTORY_PROMPT = """
User Question:
{question}

Tool Call Trajectory:
{tools}

Agent's Final Answer:
{agent_answer}
""".strip()


def create_trajectory_judge() -> Agent:
    return Agent(
        name="trajectory_judge",
        model="openai:gpt-4o-mini",
        instructions=TRAJECTORY_INSTRUCTIONS,
        output_type=TrajectoryResult,
    )


def _normalize_args(raw: str) -> str:
    """Normalize JSON args to a consistent format so duplicates are obvious."""
    try:
        parsed = _json.loads(raw)
        return _json.dumps(parsed, separators=(",", ":"))
    except (ValueError, TypeError):
        return raw


def format_trajectory_prompt(entry: Dict[str, Any]) -> str:
    tools_parts = []
    for i, t in enumerate(entry["tools"], 1):
        tools_parts.append(f"{i}. {t['name']}({_normalize_args(t['args'])})")
    tools_str = "\n".join(tools_parts) or "(no tool calls)"

    return TRAJECTORY_PROMPT.format(
        question=entry["input"]["question"],
        tools=tools_str,
        agent_answer=entry["rag_response"]["answer"],
    )
