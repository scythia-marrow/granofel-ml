import pytest
from granofel.workflow import BaseNode, BaseWorkflow, WorkflowChain, LLMClass


class PlannerNode(BaseNode):
    """Node that takes a problem and outputs a plan."""
    class State(BaseNode.State):
        problem: str = ""
        plan: str = ""


class CompressorNode(BaseNode):
    """Node that takes a plan and outputs compact intuition."""
    class State(BaseNode.State):
        plan: str = ""
        compact: str = ""


class ExecutorNode(BaseNode):
    """Node that takes problem + compact and outputs solution."""
    class State(BaseNode.State):
        problem: str = ""
        compact: str = ""
        solution: str = ""


@pytest.mark.asyncio
async def test_chain_single_workflow(mock_llm_dict):
    """Test chain with single workflow works like normal run."""
    node = PlannerNode([
        ("system", "You are a planner."),
        ("human", "{state.problem}"),
    ], name="planner")

    workflow = BaseWorkflow("plan", mock_llm_dict, node=[node])
    chain = WorkflowChain([workflow])

    result = await chain.run({"problem": "What is 2+2?"})

    assert "messages" in result
    assert len(result["messages"]) == 3  # system, human, AI


@pytest.mark.asyncio
async def test_chain_multiple_workflows(mock_llm_dict):
    """Test chaining multiple workflows in sequence."""
    planner_node = PlannerNode([
        ("system", "Create a plan for: {state.problem}"),
    ], name="planner")

    compressor_node = CompressorNode([
        lambda x: x,
        ("system", "Compress the plan."),
    ], name="compressor")

    planner = BaseWorkflow("planner", mock_llm_dict, node=[planner_node])
    compressor = BaseWorkflow("compressor", mock_llm_dict, node=[compressor_node])

    chain = WorkflowChain([planner, compressor])
    result = await chain.run({"problem": "solve x=5"})

    # Both workflows executed
    assert "messages" in result


@pytest.mark.asyncio
async def test_chain_with_field_mapping(mock_llm_dict):
    """Test field mapping between workflows."""
    # Planner outputs to 'plan' field
    planner_node = PlannerNode([
        ("system", "Plan: {state.problem}"),
    ], name="planner")

    # Compressor expects 'input_plan' field (different name)
    class MappedCompressorNode(BaseNode):
        class State(BaseNode.State):
            input_plan: str = ""

    compressor_node = MappedCompressorNode([
        ("system", "Compress: {state.input_plan}"),
    ], name="compressor")

    planner = BaseWorkflow("planner", mock_llm_dict, node=[planner_node])
    compressor = BaseWorkflow("compressor", mock_llm_dict, node=[compressor_node])

    # Map 'plan' from planner to 'input_plan' for compressor
    chain = WorkflowChain(
        workflows=[planner, compressor],
        mappings=[{"plan": "input_plan"}]
    )

    result = await chain.run({"problem": "test", "plan": "original plan"})
    # After mapping, 'plan' becomes 'input_plan'
    assert "input_plan" in result
    assert "plan" not in result


@pytest.mark.asyncio
async def test_chain_preserves_state_across_workflows(mock_llm_dict):
    """Test that state fields persist across workflow boundaries."""
    node1 = PlannerNode([
        ("human", "{state.problem}"),
    ], name="node1")

    node2 = ExecutorNode([
        lambda x: x,
        ("system", "Problem was: {state.problem}"),
    ], name="node2")

    wf1 = BaseWorkflow("wf1", mock_llm_dict, node=[node1])
    wf2 = BaseWorkflow("wf2", mock_llm_dict, node=[node2])

    chain = WorkflowChain([wf1, wf2])
    result = await chain.run({"problem": "persistent value"})

    # 'problem' should still be accessible
    assert result.get("problem") == "persistent value"


@pytest.mark.asyncio
async def test_chain_stream_yields_per_workflow(mock_llm_dict):
    """Test stream yields state after each workflow."""
    node1 = PlannerNode([("human", "q1")], name="n1")
    node2 = CompressorNode([lambda x: x, ("human", "q2")], name="n2")

    wf1 = BaseWorkflow("first", mock_llm_dict, node=[node1])
    wf2 = BaseWorkflow("second", mock_llm_dict, node=[node2])

    chain = WorkflowChain([wf1, wf2])

    results = []
    async for update in chain.stream({"problem": "test"}):
        results.append(update)

    assert len(results) == 2
    assert "first" in results[0]
    assert "second" in results[1]


def test_chain_requires_workflows():
    """Test that empty workflow list raises error."""
    with pytest.raises(ValueError, match="requires at least one workflow"):
        WorkflowChain([])


def test_chain_validates_mapping_count(mock_llm_dict):
    """Test that mapping count must match transitions."""
    node = PlannerNode([("human", "q")], name="n")
    wf1 = BaseWorkflow("wf1", mock_llm_dict, node=[node])
    wf2 = BaseWorkflow("wf2", mock_llm_dict, node=[node])
    wf3 = BaseWorkflow("wf3", mock_llm_dict, node=[node])

    # 3 workflows = 2 transitions, but providing 1 mapping
    with pytest.raises(ValueError, match="Expected 2 mappings"):
        WorkflowChain([wf1, wf2, wf3], mappings=[{"a": "b"}])


@pytest.mark.asyncio
async def test_chain_distillation_pipeline(mock_llm_dict):
    """Test full distillation pipeline: planner -> compressor -> executor."""
    planner_node = PlannerNode([
        ("system", "Create detailed plan for: {state.problem}"),
    ], llm_type=LLMClass.THINKING, name="planner")

    compressor_node = CompressorNode([
        lambda x: x,
        ("system", "Compress to key insights."),
    ], llm_type=LLMClass.TRAINABLE, name="compressor")

    executor_node = ExecutorNode([
        lambda x: x,
        ("system", "Execute using intuition."),
    ], llm_type=LLMClass.FAST, name="executor")

    planner = BaseWorkflow("planner", mock_llm_dict, node=[planner_node])
    compressor = BaseWorkflow("compressor", mock_llm_dict, node=[compressor_node])
    executor = BaseWorkflow("executor", mock_llm_dict, node=[executor_node])

    chain = WorkflowChain([planner, compressor, executor])

    result = await chain.run({"problem": "What is 2+2?"})

    # All three stages executed
    assert "messages" in result


def test_chain_run_sync(mock_llm_dict):
    """Test synchronous run method."""
    node = PlannerNode([("human", "{state.problem}")], name="n")
    wf = BaseWorkflow("wf", mock_llm_dict, node=[node])

    chain = WorkflowChain([wf])
    result = chain.run_sync({"problem": "test"})

    assert "messages" in result


@pytest.mark.asyncio
async def test_chain_multiple_field_mappings(mock_llm_dict):
    """Test mapping renames multiple fields."""
    node = PlannerNode([("human", "q")], name="n")
    wf1 = BaseWorkflow("wf1", mock_llm_dict, node=[node])
    wf2 = BaseWorkflow("wf2", mock_llm_dict, node=[node])

    chain = WorkflowChain(
        [wf1, wf2],
        mappings=[{"plan": "renamed_plan", "problem": "renamed_problem"}]
    )

    result = await chain.run({"problem": "p", "plan": "pl"})

    assert "renamed_plan" in result
    assert "renamed_problem" in result
    assert "plan" not in result
    assert "problem" not in result
