import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_llm():
    """Mock LLM that returns predictable responses."""
    llm = MagicMock()

    call_count = [0]

    def make_response(*args, **kwargs):
        call_count[0] += 1
        return AIMessage(content=f"Mock response {call_count[0]}")

    async def make_response_async(*args, **kwargs):
        return make_response(*args, **kwargs)

    llm.invoke = MagicMock(side_effect=make_response)
    llm.ainvoke = AsyncMock(side_effect=make_response_async)

    return llm


@pytest.fixture
def mock_llm_dict(mock_llm):
    """LLM dictionary for workflow construction.

    Maps all LLMClass types to the same mock for testing.
    """
    return {
        "react": mock_llm,
        "reasoning": mock_llm,
        "thinking": mock_llm,
        "trainable": mock_llm,
        "fast": mock_llm,
    }


@pytest.fixture
def real_llm():
    """Real LLM for integration tests. Requires .env with API key."""
    from langchain.chat_models import init_chat_model

    try:
        key = open("/science/quine/.env", "r").readline().strip()
        return {"react": init_chat_model("openai:gpt-4o", api_key=key)}
    except FileNotFoundError:
        pytest.skip("No .env file found - skipping integration test")
