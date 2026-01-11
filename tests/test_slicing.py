import pytest
from pydantic import BaseModel
from granofel.workflow import BaseNode, BaseWorkflow, Snapshot, StateManager


class NodeA(BaseNode):
    """Node that only sees fields: messages, alpha, shared"""
    class State(BaseNode.State):
        alpha: str
        shared: str


class NodeB(BaseNode):
    """Node that only sees fields: messages, beta, shared"""
    class State(BaseNode.State):
        beta: str
        shared: str


class NodeC(BaseNode):
    """Node that only sees fields: messages, gamma (no overlap except messages)"""
    class State(BaseNode.State):
        gamma: str


# Track what each node actually received
received_states = []


class SpyNodeA(NodeA):
    async def invoke(self, state, config=None):
        received_states.append(("A", state.model_dump()))
        return await super().invoke(state, config)


class SpyNodeB(NodeB):
    async def invoke(self, state, config=None):
        received_states.append(("B", state.model_dump()))
        return await super().invoke(state, config)


class SpyNodeC(NodeC):
    async def invoke(self, state, config=None):
        received_states.append(("C", state.model_dump()))
        return await super().invoke(state, config)


@pytest.fixture(autouse=True)
def clear_received():
    received_states.clear()


class TestSnapshot:
    """Test Snapshot.get_slice behavior directly."""

    def test_slice_extracts_only_declared_fields(self):
        snapshot = Snapshot({
            "alpha": "a_value",
            "beta": "b_value",
            "shared": "shared_value",
            "gamma": "g_value",
            "messages": []
        })

        slice_a = snapshot.get_slice(NodeA.State)
        assert slice_a.alpha == "a_value"
        assert slice_a.shared == "shared_value"
        assert slice_a.messages == []
        assert not hasattr(slice_a, "beta")
        assert not hasattr(slice_a, "gamma")

        slice_b = snapshot.get_slice(NodeB.State)
        assert slice_b.beta == "b_value"
        assert slice_b.shared == "shared_value"
        assert not hasattr(slice_b, "alpha")

        slice_c = snapshot.get_slice(NodeC.State)
        assert slice_c.gamma == "g_value"
        assert not hasattr(slice_c, "alpha")
        assert not hasattr(slice_c, "beta")
        assert not hasattr(slice_c, "shared")

    def test_slice_handles_missing_fields(self):
        """Slice should work even if some fields aren't in snapshot."""
        snapshot = Snapshot({"alpha": "a_value", "messages": []})

        slice_a = snapshot.get_slice(NodeA.State)
        assert slice_a.alpha == "a_value"
        # shared is missing from snapshot, should not be set
        assert not hasattr(slice_a, "shared") or slice_a.shared is None


@pytest.mark.asyncio
async def test_workflow_slices_state_per_node(mock_llm_dict):
    """Test that each node receives only its declared state fields."""
    node_a = SpyNodeA([("human", "msg")], name="node_a")
    node_b = SpyNodeB([("human", "msg")], name="node_b")
    node_c = SpyNodeC([("human", "msg")], name="node_c")

    wf = BaseWorkflow("test", mock_llm_dict, node=[node_a, node_b, node_c])
    runner = wf.compile()

    await runner.run({
        "alpha": "a_value",
        "beta": "b_value",
        "gamma": "g_value",
        "shared": "shared_value",
    })

    # Verify each node got exactly its slice
    assert len(received_states) == 3

    name_a, state_a = received_states[0]
    assert name_a == "A"
    assert "alpha" in state_a
    assert "shared" in state_a
    assert "beta" not in state_a
    assert "gamma" not in state_a

    name_b, state_b = received_states[1]
    assert name_b == "B"
    assert "beta" in state_b
    assert "shared" in state_b
    assert "alpha" not in state_b
    assert "gamma" not in state_b

    name_c, state_c = received_states[2]
    assert name_c == "C"
    assert "gamma" in state_c
    assert "alpha" not in state_c
    assert "beta" not in state_c
    assert "shared" not in state_c


@pytest.mark.asyncio
async def test_nodes_share_overlapping_fields(mock_llm_dict):
    """Test that overlapping fields (like 'shared') are visible to multiple nodes."""
    node_a = SpyNodeA([("human", "msg")], name="node_a")
    node_b = SpyNodeB([("human", "msg")], name="node_b")

    wf = BaseWorkflow("test", mock_llm_dict, node=[node_a, node_b])
    runner = wf.compile()

    await runner.run({
        "alpha": "a_value",
        "beta": "b_value",
        "shared": "both_see_this",
    })

    _, state_a = received_states[0]
    _, state_b = received_states[1]

    # Both nodes should see the shared field
    assert state_a["shared"] == "both_see_this"
    assert state_b["shared"] == "both_see_this"
