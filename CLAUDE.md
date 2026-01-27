# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Granofel is a messages-centric framework for building multimodal LLM agents using LangChain. It provides a workflow system with state management, node composition, and parallel execution patterns.

## Best Practices

Use feature branches, pull requests, and the chainlink cli issue tracker.

A general session will proceed as follows:

1) Start a chainlink session, decide which issue to work on
2) Mark the issue as active
3) Create or checkout an associated issue branch named GRAN-<num>\_description
4) Work on the issue until completion
5) Create a PR to main
6) Retrieve merge review comments from the request, implement fixes. Remember to read both general comments from the PR page as well as individual code critiques.
7) Verify that the PR has been merged
8) Close the chainlink issue
9) Add any new issues discovered / out of scope for this session
10) End the session

Be conscientious, NEVER close an issue before the request has been merged.

Separate concerns in both planning and code. ALWAYS have only one concern per PR. NEVER submit PR that depends on another PR.

Prefer functional programming style to OOP style.

Support dependency injection! This means NEVER construct a component class directly, ALWAYS take component classes as arguments. Keep argument lists short. If argument lists get too unweildy that means we need to either use configuration files or restructure the code.

Don't Repeat Yourself (DRY).

## Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_workflow.py

# Run single test by name
pytest tests/test_scatter.py::test_scatter_executes_all_branches -v

# Run with output
pytest -s

# Activate virtualenv (required before running)
source venv/bin/activate
```

## Architecture

### Core Components (`granofel/workflow.py`)

**State Management Pattern:**
- `Snapshot`: Immutable point-in-time state container with version tracking
- `StateManager`: Central coordinator with update queue and reducers. Supports synchronous mode (explicit `process_updates()`) and polling mode (background task on steady clock via `clock_interval`)
- `UpdateMessage`: Delta updates from nodes to StateManager

**Reducers:**
- Default reducer: replace (last write wins)
- Messages reducer: append (message history accumulates)
- Custom reducers can be registered via `register_reducer()`

**Node System:**
- `BaseNode`: Single LLM invocation with messages template. Each node declares a `State` class (Pydantic model) defining the fields it needs
- `ScatterNode`: Parallel execution - runs contained workflow per-item with isolated state, then gathers results. Accepts `BaseWorkflow`, `BaseNode`, or `Sequence[BaseNode]`
- Nodes receive only their declared state slice (via `Snapshot.get_slice()`), not the full workflow state

**Workflow Compilation:**
- `BaseWorkflow`: Container for nodes with LLM binding. Builds composite state class from all node states using MRO-ordered inheritance
- `WorkflowRunner`: Executes compiled workflow, coordinating between nodes and StateManager
- Compile with `wf.compile(clock_interval=0.1)` for polling mode, or `wf.compile()` for synchronous mode

### Message Formatting

Messages support templating with `{state.field}` syntax:
- String: formatted as-is
- 1-tuple `(str,)`: single element formatted
- 2-tuple `(role, content)`: role + content pair
- Callable: receives messages list, returns formatted messages (used for `lambda x: x` to pass through previous messages)

### LLM Classes

Nodes specify which LLM type they need via `LLMClass`, workflow provides LLM dictionary at construction.

Original types:
- `LLMClass.REACT`: Agent-style models with tool use
- `LLMClass.REASONING`: Models with extended reasoning

Distillation pipeline types:
- `LLMClass.THINKING`: Extended reasoning models (o1, Claude thinking) for planning
- `LLMClass.TRAINABLE`: Fine-tunable models for compression learning
- `LLMClass.FAST`: Quick non-thinking models for execution

## Test Fixtures (`tests/conftest.py`)

- `mock_llm`: Returns predictable `AIMessage` responses for unit tests
- `mock_llm_dict`: LLM dictionary with mock under "react" key
- `real_llm`: Integration test fixture requiring API key in `/science/quine/.env`
