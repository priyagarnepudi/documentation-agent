import math
import re

import pytest

from sql_agent import agent
from sql_tools import setup_database
from utils import collect_tools
from judge import assert_criteria


@pytest.mark.asyncio
async def test_trips_with_more_than_five_passengers():
    setup_database()
    result = await agent.run("How many trips had more than 5 passengers?")
    output = result.output

    assert isinstance(output.sql_query, str)
    assert output.sql_query.strip(), "The agent must return a non-empty SQL query"
    # Verified directly: SELECT COUNT(*) FROM trips WHERE passenger_count > 5.
    # Accept both 22413 and the human-readable formatting 22,413.
    assert "22413" in output.result_text.replace(",", ""), output.result_text


@pytest.mark.asyncio
async def test_schema_before_sql():
    setup_database()
    result = await agent.run("What is the most common payment type?")
    tool_calls = collect_tools(result.new_messages())
    tool_names = [call.name for call in tool_calls]
    print(f"Tool calls in order: {tool_names}")

    assert tool_calls, "The agent must call its SQL tools"
    assert tool_calls[0].name == "get_schema"
    assert "run_sql" in tool_names


@pytest.mark.asyncio
async def test_highest_average_fare_by_hour():
    setup_database()
    result = await agent.run("Which hour of the day has the highest average fare amount?")
    print(result.output.model_dump_json(indent=2))
    await assert_criteria(result, [
        "the SQL query correctly calculates average fare by hour of day",
        "the result identifies a specific hour as having the highest average fare",
        "the result includes the actual average fare amount",
    ])


@pytest.mark.parametrize(
    "question,columns,expected_numbers,expected_text",
    [
        pytest.param(
            "What is the average tip amount for credit card payments?",
            ["tip_amount", "payment_type"],
            [4.169670627490482],
            None,
            id="credit-card-tip",
        ),
        pytest.param(
            "Which pickup location (PULocationID) has the most trips?",
            ["PULocationID"],
            [132],
            None,
            id="most-popular-pickup",
        ),
        pytest.param(
            "What is the average fare for trips longer than 10 miles?",
            ["fare_amount", "trip_distance"],
            [62.88081242825971],
            None,
            id="long-trip-fare",
        ),
        pytest.param(
            "How many trips had zero passengers recorded?",
            ["passenger_count"],
            [31465],
            None,
            id="zero-passengers",
        ),
        pytest.param(
            "What is the busiest day of the week for taxi trips?",
            ["tpep_pickup_datetime"],
            [],
            "Wednesday",
            id="busiest-weekday",
        ),
    ],
)
@pytest.mark.asyncio
async def test_additional_taxi_questions(question, columns, expected_numbers, expected_text):
    setup_database()
    result = await agent.run(question)
    output = result.output
    tool_calls = collect_tools(result.new_messages())
    tool_names = [call.name for call in tool_calls]
    print(f"Question: {question}")
    print(f"Tool calls: {tool_names}")
    print(output.model_dump_json(indent=2))

    assert tool_names, "The agent must use database tools"
    assert tool_names[0] == "get_schema"
    assert "run_sql" in tool_names[1:]
    for column in columns:
        assert re.search(rf"\b{re.escape(column)}\b", output.sql_query, re.IGNORECASE)

    # Compare numeric tokens, accepting thousands separators and cent rounding.
    numbers = [
        float(value)
        for value in re.findall(r"-?\d+(?:\.\d+)?", output.result_text.replace(",", ""))
    ]
    for expected in expected_numbers:
        assert any(
            math.isclose(number, expected, rel_tol=0, abs_tol=0.01)
            for number in numbers
        ), output.result_text
    if expected_text:
        assert expected_text.lower() in output.result_text.lower()
