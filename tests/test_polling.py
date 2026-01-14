import pytest
import asyncio
from granofel.workflow import BaseNode, BaseWorkflow, StateManager, UpdateMessage


class SimpleNode(BaseNode):
	class State(BaseNode.State):
		counter: int = 0


@pytest.mark.asyncio
async def test_state_manager_polling_mode_processes_updates(mock_llm_dict):
	"""Test that polling mode processes queued updates on interval."""
	manager = StateManager(clock_interval=0.01)  # 10ms interval
	manager.init_snapshot({"counter": 0, "messages": []})

	# Submit updates before starting polling
	manager.submit_update_sync(UpdateMessage(source="test", updates={"counter": 1}))
	manager.submit_update_sync(UpdateMessage(source="test", updates={"counter": 2}))

	# Start polling
	manager.start_polling()

	# Wait for at least one polling cycle
	await asyncio.sleep(0.03)

	# Stop polling
	await manager.stop_polling()

	# Should have processed updates
	snapshot = manager.get_snapshot()
	assert snapshot["counter"] == 2  # Last update wins (default reducer)


@pytest.mark.asyncio
async def test_state_manager_polling_increments_version():
	"""Test that polling increments snapshot version."""
	manager = StateManager(clock_interval=0.01)
	manager.init_snapshot({"value": 0, "messages": []})

	initial_version = manager.get_snapshot().version

	manager.submit_update_sync(UpdateMessage(source="test", updates={"value": 1}))

	manager.start_polling()
	await asyncio.sleep(0.03)
	await manager.stop_polling()

	assert manager.get_snapshot().version > initial_version


@pytest.mark.asyncio
async def test_state_manager_stop_polling_processes_remaining():
	"""Test that stop_polling processes any remaining queued updates."""
	manager = StateManager(clock_interval=1.0)  # Long interval
	manager.init_snapshot({"value": 0, "messages": []})

	manager.start_polling()

	# Submit update but don't wait for polling cycle
	manager.submit_update_sync(UpdateMessage(source="test", updates={"value": 42}))

	# Stop immediately - should still process remaining updates
	await manager.stop_polling()

	assert manager.get_snapshot()["value"] == 42


@pytest.mark.asyncio
async def test_state_manager_polling_requires_interval():
	"""Test that start_polling fails without clock_interval."""
	manager = StateManager()  # No clock_interval

	with pytest.raises(ValueError, match="clock_interval must be positive"):
		manager.start_polling()


@pytest.mark.asyncio
async def test_state_manager_double_start_fails():
	"""Test that starting polling twice raises error."""
	manager = StateManager(clock_interval=0.01)
	manager.init_snapshot({})

	manager.start_polling()

	with pytest.raises(RuntimeError, match="Polling already active"):
		manager.start_polling()

	await manager.stop_polling()


@pytest.mark.asyncio
async def test_workflow_with_polling_mode(mock_llm_dict):
	"""Test workflow execution with polling enabled."""
	node1 = SimpleNode([("human", "first")], name="first")
	node2 = SimpleNode([lambda x: x, ("human", "second")], name="second")

	wf = BaseWorkflow("test", mock_llm_dict, node=[node1, node2])
	runner = wf.compile(clock_interval=0.01)

	result = await runner.run({"counter": 0})

	# Should complete successfully with messages
	assert "messages" in result
	assert len(result["messages"]) > 0


@pytest.mark.asyncio
async def test_workflow_polling_stops_on_completion(mock_llm_dict):
	"""Test that polling is stopped when workflow completes."""
	node = SimpleNode([("human", "msg")], name="node")

	wf = BaseWorkflow("test", mock_llm_dict, node=[node])
	runner = wf.compile(clock_interval=0.01)

	await runner.run({"counter": 0})

	# Polling should be stopped
	assert not runner.manager.is_polling


@pytest.mark.asyncio
async def test_workflow_polling_stops_on_error(mock_llm_dict):
	"""Test that polling is stopped even if workflow raises error."""
	class FailingNode(BaseNode):
		async def invoke(self, state, config=None):
			raise RuntimeError("Intentional failure")

	node = FailingNode([("human", "msg")], name="fail")
	node.bind_llm(mock_llm_dict)

	wf = BaseWorkflow("test", mock_llm_dict, node=[node])
	runner = wf.compile(clock_interval=0.01)

	with pytest.raises(RuntimeError, match="Intentional failure"):
		await runner.run({})

	# Polling should still be stopped
	assert not runner.manager.is_polling


@pytest.mark.asyncio
async def test_polling_accumulates_messages(mock_llm_dict):
	"""Test that messages accumulate correctly in polling mode."""
	node1 = SimpleNode([("human", "first")], name="first")
	node2 = SimpleNode([lambda x: x, ("human", "second")], name="second")

	wf = BaseWorkflow("test", mock_llm_dict, node=[node1, node2])
	runner = wf.compile(clock_interval=0.01)

	result = await runner.run({})

	# Messages should accumulate across nodes
	assert "messages" in result
	# At minimum we should have the final node's messages
	assert len(result["messages"]) >= 2
