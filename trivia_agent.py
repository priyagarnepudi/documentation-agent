
import os
import sys
from pathlib import Path

import logfire
import questionary
from dotenv import load_dotenv
from pydantic_ai import Agent

from trivia_tools import TriviaTools

load_dotenv(Path(__file__).with_name(".env"))
logfire.configure(
    service_name="trivia-agent",
    token=os.getenv("LOGFIRE_TOKEN") or os.getenv("LOGFIRE_WRITE_TOKEN"),
    send_to_logfire="if-token-present",
)
logfire.instrument_pydantic_ai()

instructions = """You are a trivia quizmaster. When asked to play trivia:
1. Use the available tools to fetch trivia questions
2. Ask the player one question at a time with multiple choice options
3. Wait for their answer before moving to the next question
4. When the player answers, explain why the correct answer is correct - add interesting context and facts
5. After all questions, give the final score
Do not reveal correct answers before the player responds.
"""

trivia_tools = TriviaTools()
agent = Agent(
    "openai:gpt-4o-mini",
    name="agent",
    tools=[trivia_tools.get_categories, trivia_tools.get_questions],
    instructions=instructions,
)


def ask_feedback():
    result = questionary.select(
        "How was the trivia session?",
        choices=["👍 Good", "👎 Bad", "Skip"],
    ).ask()
    if result is None or result == "Skip":
        return None
    return 1 if "Good" in result else -1


def run(prompt):
    history = []
    while prompt.strip() and prompt.strip().lower() not in {"quit", "exit", "stop"}:
        result = agent.run_sync(prompt, message_history=history)
        history.extend(result.new_messages())
        print(f"Quizmaster: {result.output}")
        prompt = input("You (or 'quit'): ")


def run_session(prompt):
    with logfire.span("trivia_session"):
        session_context = logfire.get_context()
        run(prompt)
        feedback = ask_feedback()
        if feedback is not None:
            with logfire.attach_context(session_context):
                logfire.info("user_feedback", feedback=feedback)
    logfire.force_flush()


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or input("You (or 'quit'): ")
    if prompt.strip() and prompt.strip().lower() not in {"quit", "exit", "stop"}:
        run_session(prompt)
