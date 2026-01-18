import pytest
from granofel.workflow import BaseNode, BaseWorkflow, LLMClass


class ProfessionQuery(BaseNode):
    class State(BaseNode.State):
        profession: str
        user_prompt: str


@pytest.mark.asyncio
async def test_workflow_executes_nodes_in_order(mock_llm_dict):
    """Test that workflow executes nodes sequentially."""
    node1 = ProfessionQuery([
        ("system", "Answer the user's question."),
        ("human", "{state.user_prompt}"),
    ], name="first")

    node2 = ProfessionQuery([
        lambda x: x,
        ("system", "Now respond differently."),
    ], name="second")

    wf = BaseWorkflow("test", mock_llm_dict, node=[node1, node2])
    runner = wf.compile()

    results = []
    async for update in runner.stream({
        "profession": "test",
        "user_prompt": "hello"
    }):
        results.append(update)

    assert len(results) == 2
    assert "first" in results[0]
    assert "second" in results[1]


@pytest.mark.asyncio
async def test_messages_accumulate(mock_llm_dict):
    """Test that messages reducer appends across nodes."""
    node1 = ProfessionQuery([
        ("human", "first message"),
    ], name="first")

    node2 = ProfessionQuery([
        lambda x: x,
        ("human", "second message"),
    ], name="second")

    wf = BaseWorkflow("test", mock_llm_dict, node=[node1, node2])
    runner = wf.compile()

    final = await runner.run({
        "profession": "test",
        "user_prompt": "hello"
    })

    messages = final["messages"]
    # first node: human + AI response
    # second node: previous messages + human + AI response
    assert len(messages) == 6  # 2 from first (human, ai) + 4 from second (prev 2 + human, ai)


@pytest.mark.asyncio
async def test_node_invoke_standalone(mock_llm_dict):
    """Test that nodes can be invoked independently."""
    node = ProfessionQuery([
        ("system", "You are {state.profession}"),
        ("human", "{state.user_prompt}"),
    ], name="standalone")
    node.bind_llm(mock_llm_dict)

    state = ProfessionQuery.State(
        profession="a wizard",
        user_prompt="cast a spell"
    )

    result = await node.invoke(state)

    assert "messages" in result
    assert len(result["messages"]) == 3  # system, human, AI response


class PlannerNode(BaseNode):
    """Node with problem_statement for distillation pipeline test."""
    class State(BaseNode.State):
        problem_statement: str = ""


@pytest.mark.asyncio
async def test_llm_class_types_for_distillation_pipeline(mock_llm_dict):
    """Test that all LLMClass types can be used in a workflow."""
    # Planner node uses THINKING for extended reasoning
    planner = PlannerNode([
        ("system", "Plan the solution step by step."),
        ("human", "{state.problem_statement}"),
    ], llm_type=LLMClass.THINKING, name="planner")

    # Compressor node uses TRAINABLE for fine-tuning
    compressor = BaseNode([
        lambda x: x,
        ("system", "Compress the plan to key insights."),
    ], llm_type=LLMClass.TRAINABLE, name="compressor")

    # Executor node uses FAST for quick execution
    executor = BaseNode([
        lambda x: x,
        ("system", "Execute using the compressed plan."),
    ], llm_type=LLMClass.FAST, name="executor")

    wf = BaseWorkflow("distill", mock_llm_dict, node=[planner, compressor, executor])
    runner = wf.compile()

    results = []
    async for update in runner.stream({"problem_statement": "solve x + 2 = 5"}):
        results.append(update)

    assert len(results) == 3
    assert "planner" in results[0]
    assert "compressor" in results[1]
    assert "executor" in results[2]


def test_llm_class_enum_has_all_types():
    """Verify all LLMClass types are defined."""
    # Original types
    assert LLMClass.REACT.value == "react"
    assert LLMClass.REASONING.value == "reasoning"
    # Distillation pipeline types
    assert LLMClass.THINKING.value == "thinking"
    assert LLMClass.TRAINABLE.value == "trainable"
    assert LLMClass.FAST.value == "fast"
