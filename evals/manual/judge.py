from typing import Literal, Any, Dict
from pydantic import BaseModel, Field
from pydantic_ai import Agent

judge_instructions = """
You are an expert evaluator assessing the performance of a documentation assistant (RAG agent) for the Evidently AI documentation. 
You will be provided with the user's question, the tools the agent used, and the agent's final response.
Your task is to review the interaction and classify the agent's response as either "good" or "bad".

A response is "good" if:
1. It accurately and completely answers the user's question using ONLY the provided documentation.
2. It correctly identifies if a question is off-topic (e.g., about cooking, sports, or completely unsupported generic tools) and refuses to answer it explicitly.
3. It uses the appropriate tools correctly.
4. The format is easy to read, follows good markdown practices, and contains no broken symbols.

A response is "bad" if ANY of the following apply:
1. It MUST NOT hallucinate integrations, tools, or features that are NOT explicitly detailed in the Evidently documentation context. If it makes up instructions for ANY unprovided tool (e.g., Kubernetes, Datadog, Apache Kafka, MLflow, Grafana) or unsupported data modes (e.g. streaming data), it is "bad".
2. It MUST explicitly refuse to answer general machine learning questions that are not about Evidently itself (e.g., asking how to train a Random Forest model, Support Vector Machines, cross-validation). Providing an answer to these is "bad".
3. When providing UI instructions (like adding panels or alerts), it MUST explicitly mention "Evidently Cloud". If it just says "UI" or "dashboard" without specifying Evidently Cloud, it is "bad".
4. It is factually incorrect, contains broken symbols/formatting (such as the raw word "plus" or random characters), mentions broken images/links, or hallucinates features.

Be extremely strict. The agent should be penalized heavily (labeled "bad") for ANY hallucination, ANY off-topic answer (such as teaching Random Forest or Grafana), or ANY omission of "Evidently Cloud" when discussing the UI. Take a step-by-step approach to reason about the quality, verifying each criteria before providing your final label.
""".strip()

class JudgeEvaluation(BaseModel):
    """
    The output format for the LLM Judge evaluating a RAG log entry.
    """
    reasoning: str = Field(
        description="A step-by-step reasoning evaluating the agent's response against the criteria."
    )
    label: Literal["good", "bad"] = Field(
        description="The final evaluation label for the response: 'good' or 'bad'."
    )

def create_log_judge_agent() -> Agent:
    """
    Creates and returns the judge agent configured to evaluate RAG logs.
    """
    return Agent(
        name="log_judge",
        model="openai:gpt-4o-mini",
        instructions=judge_instructions,
        output_type=JudgeEvaluation
    )

judge_prompt_template = """
Evaluate the agent's performance for the following interaction.

User Question:
{question}

Tools Used:
{tools}

Agent Response:
{answer}
"""

def format_judge_prompt(log_entry: Dict[str, Any]) -> str:
    """
    Formats a single log entry from results.json into a prompt for the judge agent.
    """
    question = log_entry["input"]["question"]
    answer = log_entry["rag_response"]["answer"]
    tools_list = log_entry["tools"]

    tools_str_parts = []
    for t in tools_list:
        name = t['name']
        args = t['args']
        tools_str_parts.append(f"{name}({args})")

    tools = "\n".join(tools_str_parts)
    
    return judge_prompt_template.format(
        question=question,
        tools=tools,
        answer=answer
    )
