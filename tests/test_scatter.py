import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from granofel.workflow import BaseNode, ScatterNode, BaseWorkflow, LLMClass


class ItemProcessor(BaseNode):
	"""Node that processes a single item."""
	class State(BaseNode.State):
		item: str = ""

	async def invoke(self, state, config=None):
		result = await super().invoke(state, config)
		result["processed_item"] = f"processed:{state.item}"
		return result


# Track invocations for isolation tests
invocation_log = []


class LoggingNode(BaseNode):
	"""Node that logs its invocations for testing."""
	class State(BaseNode.State):
		item: str = ""
		branch_id: str = ""

	async def invoke(self, state, config=None):
		invocation_log.append({
			"item": state.item,
			"branch_id": getattr(state, "branch_id", "none"),
			"messages": list(state.messages),
		})
		return await super().invoke(state, config)


@pytest.fixture(autouse=True)
def clear_log():
	invocation_log.clear()


@pytest.mark.asyncio
async def test_scatter_executes_all_branches(mock_llm_dict):
	"""Test that scatter runs workflow for each item."""
	node = ItemProcessor([("human", "Process {state.item}")], name="processor")
	scatter = ScatterNode([node], name="scatter")
	scatter.bind_llm(mock_llm_dict)

	class TestState(ScatterNode.State):
		pass

	state = TestState(items=["a", "b", "c"])
	result = await scatter.invoke(state)

	assert "messages" in result
	# Should have messages from all 3 branches
	assert len(result["messages"]) == 6  # 2 per branch (human + AI)


@pytest.mark.asyncio
async def test_scatter_state_isolation(mock_llm_dict):
	"""Test that branches don't see each other's state changes."""
	node = LoggingNode([("human", "Item: {state.item}")], name="logger")
	scatter = ScatterNode([node], name="scatter")
	scatter.bind_llm(mock_llm_dict)

	state = ScatterNode.State(items=["x", "y", "z"])
	await scatter.invoke(state)

	# Each branch should have seen only its own item
	assert len(invocation_log) == 3
	items_seen = {log["item"] for log in invocation_log}
	assert items_seen == {"x", "y", "z"}

	# Each branch should have started with empty messages (isolated)
	for log in invocation_log:
		assert log["messages"] == []


@pytest.mark.asyncio
async def test_scatter_default_gather_appends_messages(mock_llm_dict):
	"""Test that default gather appends messages from all branches."""
	node = BaseNode([("human", "msg")], name="simple")
	scatter = ScatterNode([node], name="scatter")
	scatter.bind_llm(mock_llm_dict)

	state = ScatterNode.State(items=["1", "2"])
	result = await scatter.invoke(state)

	# messages reducer should append, not replace
	assert len(result["messages"]) == 4  # 2 messages per branch


@pytest.mark.asyncio
async def test_scatter_custom_gather(mock_llm_dict):
	"""Test that custom gather overrides default behavior."""
	node = ItemProcessor([("human", "Process {state.item}")], name="processor")

	class SummaryScatter(ScatterNode):
		def gather(self, results):
			# Custom gather: collect processed items into a list
			items = [r.get("processed_item") for r in results if "processed_item" in r]
			return {"all_processed": items, "messages": []}

	scatter = SummaryScatter([node], name="scatter")
	scatter.bind_llm(mock_llm_dict)

	state = ScatterNode.State(items=["a", "b"])
	result = await scatter.invoke(state)

	assert "all_processed" in result
	assert set(result["all_processed"]) == {"processed:a", "processed:b"}


@pytest.mark.asyncio
async def test_scatter_passes_config_to_branches(mock_llm_dict):
	"""Test that RunnableConfig is passed through to branch execution."""
	configs_received = []

	class ConfigCapture(BaseNode):
		class State(BaseNode.State):
			item: str = ""

		async def invoke(self, state, config=None):
			configs_received.append(config)
			return await super().invoke(state, config)

	node = ConfigCapture([("human", "msg")], name="capture")
	scatter = ScatterNode([node], name="scatter")
	scatter.bind_llm(mock_llm_dict)

	test_config = {"configurable": {"test_key": "test_value"}}
	state = ScatterNode.State(items=["a", "b"])
	await scatter.invoke(state, config=test_config)

	assert len(configs_received) == 2
	for cfg in configs_received:
		assert cfg == test_config


@pytest.mark.asyncio
async def test_scatter_in_workflow(mock_llm_dict):
	"""Test that ScatterNode works within a larger workflow."""
	pre_node = BaseNode([("system", "Starting")], name="pre")

	inner_node = ItemProcessor([("human", "Process {state.item}")], name="inner")
	scatter = ScatterNode([inner_node], name="scatter")

	post_node = BaseNode([lambda x: x, ("human", "Done")], name="post")

	wf = BaseWorkflow("test", mock_llm_dict, node=[pre_node, scatter, post_node])
	runner = wf.compile()

	result = await runner.run({"items": ["a", "b"]})

	assert "messages" in result
	# pre: 2, scatter(2 branches * 2): 4, post: 2 + previous
	assert len(result["messages"]) > 6


@pytest.mark.asyncio
async def test_scatter_empty_items(mock_llm_dict):
	"""Test scatter with empty items list."""
	node = BaseNode([("human", "msg")], name="simple")
	scatter = ScatterNode([node], name="scatter")
	scatter.bind_llm(mock_llm_dict)

	state = ScatterNode.State(items=[])
	result = await scatter.invoke(state)

	assert result == {}


@pytest.mark.asyncio
async def test_scatter_custom_state_fields(mock_llm_dict):
	"""Test scatter with custom state fields in addition to items."""
	class CustomScatter(ScatterNode):
		class State(ScatterNode.State):
			shared_context: str = ""

	class ContextNode(BaseNode):
		class State(BaseNode.State):
			item: str = ""
			shared_context: str = ""

	contexts_seen = []

	class CapturingNode(ContextNode):
		async def invoke(self, state, config=None):
			contexts_seen.append(state.shared_context)
			return await super().invoke(state, config)

	node = CapturingNode([("human", "{state.item}")], name="capture")
	scatter = CustomScatter([node], name="scatter")
	scatter.bind_llm(mock_llm_dict)

	state = CustomScatter.State(items=["a", "b"], shared_context="global_value")
	await scatter.invoke(state)

	# Both branches should see the shared context
	assert len(contexts_seen) == 2
	assert all(ctx == "global_value" for ctx in contexts_seen)


@pytest.mark.asyncio
async def test_scatter_accepts_single_node(mock_llm_dict):
	"""Test that ScatterNode accepts a single BaseNode (not wrapped in list)."""
	node = ItemProcessor([("human", "Process {state.item}")], name="processor")
	# Pass single node instead of list
	scatter = ScatterNode(node, name="scatter")
	scatter.bind_llm(mock_llm_dict)

	state = ScatterNode.State(items=["a", "b"])
	result = await scatter.invoke(state)

	assert "messages" in result
	assert len(result["messages"]) == 4  # 2 per branch


@pytest.mark.asyncio
async def test_scatter_accepts_workflow(mock_llm_dict):
	"""Test that ScatterNode accepts an existing BaseWorkflow."""
	node = ItemProcessor([("human", "Process {state.item}")], name="processor")
	workflow = BaseWorkflow("existing", mock_llm_dict, node=[node])

	# Pass workflow instead of nodes - extracts nodes from workflow
	scatter = ScatterNode(workflow, name="scatter")
	scatter.bind_llm(mock_llm_dict)

	state = ScatterNode.State(items=["x", "y", "z"])
	result = await scatter.invoke(state)

	assert "messages" in result
	assert len(result["messages"]) == 6  # 2 per branch, 3 branches


@pytest.mark.asyncio
async def test_scatter_workflow_input_extracts_nodes(mock_llm_dict):
	"""Test that passing a workflow extracts and uses its nodes."""
	node = ItemProcessor([("human", "Process {state.item}")], name="processor")
	workflow = BaseWorkflow("existing", mock_llm_dict, node=[node])

	scatter = ScatterNode(workflow, name="scatter")
	# Must still bind LLM to create internal workflow
	scatter.bind_llm(mock_llm_dict)

	state = ScatterNode.State(items=["a", "b"])
	result = await scatter.invoke(state)

	assert "messages" in result
	assert len(result["messages"]) == 4  # 2 per branch


def test_scatter_rejects_invalid_input():
	"""Test that ScatterNode raises TypeError for invalid input."""
	with pytest.raises(TypeError, match="nodes must be"):
		ScatterNode(123, name="invalid")

	with pytest.raises(TypeError, match="nodes must be"):
		ScatterNode({"not": "valid"}, name="invalid")


def test_scatter_warns_on_llm_rebinding(mock_llm_dict):
	"""Test that ScatterNode warns when pre-built workflow LLMs are overwritten."""
	node = ItemProcessor([("human", "Process {state.item}")], name="processor")
	workflow = BaseWorkflow("existing", mock_llm_dict, node=[node])

	scatter = ScatterNode(workflow, name="scatter")

	# Create a different LLM dict
	different_llm_dict = {"react": MagicMock()}

	with pytest.warns(UserWarning, match="Original LLM bindings are being overwritten"):
		scatter.bind_llm(different_llm_dict)


def test_scatter_no_warning_when_same_llm_dict(mock_llm_dict):
	"""Test that no warning when bind_llm receives the same dict."""
	node = ItemProcessor([("human", "Process {state.item}")], name="processor")
	workflow = BaseWorkflow("existing", mock_llm_dict, node=[node])

	scatter = ScatterNode(workflow, name="scatter")

	# Same dict reference - no warning expected
	import warnings
	with warnings.catch_warnings():
		warnings.simplefilter("error")
		scatter.bind_llm(mock_llm_dict)
