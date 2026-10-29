from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from doc_agent import DocumentationAgentConfig, create_agent


def test_repeated_searches_finish_after_two_rounds():
    offered_tools = []

    def model(messages, info):
        offered_tools.append([tool.name for tool in info.function_tools])
        if info.function_tools:
            return ModelResponse([ToolCallPart("search", {"query": "drift"})])
        return ModelResponse([TextPart("Answer from retrieved documentation")])

    class SearchTools:
        def search(self, query: str) -> str:
            return "Documentation about drift"

        def get_file(self, filename: str) -> str:
            return "Full documentation"

    agent = create_agent(
        DocumentationAgentConfig(model=FunctionModel(model)), SearchTools()
    )
    for _ in range(2):
        result = agent.run_sync("Explain drift")
        assert result.usage().requests == 3
        assert result.output == "Answer from retrieved documentation"
    assert offered_tools == [["search", "get_file"], ["search", "get_file"], []] * 2
