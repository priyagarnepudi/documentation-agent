import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from sql_tools import SQLTools, setup_database

load_dotenv()


class SQLResult(BaseModel):
    sql_query: str = Field(description="The SQL query executed to answer the question")
    result_text: str = Field(description="The answer based on the executed query results")
    row_count: int = Field(description="Number of result data rows returned by the SQL tool")


sql_tools = SQLTools()
agent = Agent(
    "openai:gpt-4o-mini",
    output_type=SQLResult,
    tools=[sql_tools.get_schema, sql_tools.run_sql],
    instructions=(
        "You answer questions about NYC taxi trips using DuckDB. "
        "Always start by calling get_schema before running any queries. "
        "Use the actual column names and types from the schema. "
        "Execute a SELECT query using run_sql and answer only from its results. "
        "Do not modify the database. Trip distances are in miles. "
        "The SQL tool returns at most 50 data rows. row_count is the number "
        "of returned data rows, not the number of underlying trips aggregated."
    ),
)


if __name__ == "__main__":
    setup_database()
    question = " ".join(sys.argv[1:]) or input("Question: ")
    result = agent.run_sync(question)
    print(result.output.model_dump_json(indent=2))
