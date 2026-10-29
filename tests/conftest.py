import pytest
import dotenv
from time import time

from cost_tracker import display_total_usage, reset_cost_file
from doc_agent import create_agent, DocumentationAgentConfig
from tools import create_documentation_tools_cached

dotenv.load_dotenv()

def pytest_sessionstart(session):
    reset_cost_file()

def pytest_sessionfinish(session, exitstatus):
    display_total_usage()


@pytest.fixture(scope="module")
def agent():
    t0 = time()

    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig()

    agent = create_agent(agent_config, tools)

    t1 = time()
    print(f'loading agent took {t1 - t0}')

    return agent
