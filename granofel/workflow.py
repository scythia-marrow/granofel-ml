import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Sequence, Callable, Any, Type, AsyncIterator, Optional, TypeVar

from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel


def polymorphic(func):
	"""Marks a method as an intentional polymorphic override of a parent class."""
	return func


def format_messages(messages, s: BaseModel):
	fmt = []
	for msg in messages:
		if isinstance(msg, Callable):
			fmt += msg(s.messages)
			continue
		if isinstance(msg, str):
			fmt.append(msg.format(state=s))
			continue
		if isinstance(msg, Sequence) and len(msg) == 1:
			fmt.append((msg[0].format(state=s),))
			continue
		if isinstance(msg, Sequence) and len(msg) == 2:
			fmt.append((msg[0], msg[1].format(state=s),))
			continue
		if isinstance(msg, Sequence) and len(msg) > 2:
			fmt.append((msg[0], msg[1].format(state=s), *msg[2:],))
			continue
		fmt.append(msg)
	return fmt


class LLMClass(str, Enum):
	"""LLM class types for workflow node binding.

	Each type represents a distinct capability profile that nodes can require:

	REACT: Agent-style models with tool use capability for interactive tasks
	REASONING: Models with extended reasoning for complex multi-step problems

	THINKING: Extended reasoning models (o1, Claude thinking) that produce
		detailed chain-of-thought, used for planning phases
	TRAINABLE: Fine-tunable models that can learn compressed reasoning patterns
		from THINKING model outputs
	FAST: Quick inference models for execution, often distilled from TRAINABLE
	"""
	REACT = "react"
	REASONING = "reasoning"
	THINKING = "thinking"
	TRAINABLE = "trainable"
	FAST = "fast"


# --- Global Workspace Components ---

class Snapshot:
	"""Immutable point-in-time state container."""

	def __init__(self, data: dict, version: int = 0):
		self._data = dict(data)  # defensive copy
		self._version = version

	@property
	def version(self) -> int:
		return self._version

	def get_slice(self, state_class: Type[BaseModel]) -> BaseModel:
		"""Extract only fields defined in state_class."""
		fields = state_class.model_fields.keys()
		slice_data = {k: self._data[k] for k in fields if k in self._data}
		return state_class.model_construct(**slice_data)

	def to_dict(self) -> dict:
		return dict(self._data)

	def __getitem__(self, key):
		return self._data[key]

	def __contains__(self, key):
		return key in self._data


class UpdateMessage(BaseModel):
	"""Delta from node to StateManager."""
	source: str
	updates: dict

	class Config:
		arbitrary_types_allowed = True


def _default_reducer(current: Any, delta: Any) -> Any:
	"""Default reducer: replace."""
	return delta


def _messages_reducer(current: list, delta: list) -> list:
	"""Messages reducer: append."""
	if current is None:
		return delta
	return current + delta


class StateManager:
	"""Central state coordinator with update queue and reducers.

	Workflows submit updates to the queue at block boundaries (workflow completion).
	Updates are processed via reducers (messages append, others replace).

	Optional polling mode (clock_interval > 0) enables background processing
	for concurrent workflow scenarios.
	"""

	def __init__(
		self,
		initial_state: dict = None,
		reducers: dict = None,
		clock_interval: Optional[float] = None
	):
		self._snapshot = Snapshot(initial_state or {}, version=0)
		self._queue: asyncio.Queue[UpdateMessage] = asyncio.Queue()
		self._reducers: Dict[str, Callable[[Any, Any], Any]] = {
			"messages": _messages_reducer,
		}
		if reducers:
			self._reducers.update(reducers)

		# Polling mode configuration
		self._clock_interval = clock_interval
		self._polling_task: Optional[asyncio.Task] = None
		self._polling_active = False

	def register_reducer(self, field: str, reducer: Callable[[Any, Any], Any]):
		"""Register a custom reducer for a field."""
		self._reducers[field] = reducer

	def get_snapshot(self) -> Snapshot:
		return self._snapshot

	def init_snapshot(self, data: dict):
		"""Initialize with new state data."""
		self._snapshot = Snapshot(data, version=0)

	async def submit_update(self, msg: UpdateMessage):
		"""Submit an update to the queue."""
		await self._queue.put(msg)

	def submit_update_sync(self, msg: UpdateMessage):
		"""Synchronous submit for non-async contexts."""
		self._queue.put_nowait(msg)

	async def process_updates(self) -> Snapshot:
		"""Process all queued updates and produce new snapshot."""
		if self._queue.empty():
			return self._snapshot

		new_data = dict(self._snapshot._data)

		while not self._queue.empty():
			msg = await self._queue.get()
			for field, delta in msg.updates.items():
				reducer = self._reducers.get(field, _default_reducer)
				current = new_data.get(field)
				new_data[field] = reducer(current, delta)

		self._snapshot = Snapshot(new_data, version=self._snapshot.version + 1)
		return self._snapshot

	@property
	def is_polling(self) -> bool:
		"""Check if polling mode is active."""
		return self._polling_active

	@property
	def clock_interval(self) -> Optional[float]:
		"""Get the configured clock interval."""
		return self._clock_interval

	async def _polling_loop(self):
		"""Background loop that processes updates on steady clock."""
		while self._polling_active:
			await self.process_updates()
			await asyncio.sleep(self._clock_interval)

	def start_polling(self) -> asyncio.Task:
		"""Start background polling task. Returns the task handle."""
		if self._clock_interval is None or self._clock_interval <= 0:
			raise ValueError("clock_interval must be positive to enable polling")
		if self._polling_active:
			raise RuntimeError("Polling already active")

		self._polling_active = True
		self._polling_task = asyncio.create_task(self._polling_loop())
		return self._polling_task

	async def stop_polling(self):
		"""Stop background polling and process any remaining updates."""
		if not self._polling_active:
			return

		self._polling_active = False
		if self._polling_task:
			self._polling_task.cancel()
			try:
				await asyncio.shield(asyncio.sleep(0))  # Let cancellation propagate
			except asyncio.CancelledError:
				pass  # Expected when task is cancelled
			self._polling_task = None

		# Process any remaining updates
		await self.process_updates()


# --- Node and Workflow ---

class JITConfig(BaseModel):
	"""Just-in-time configuration for node compilation."""
	name: str
	PREV: str
	NEXT: str
	llm: Dict[str, Runnable]

	class Config:
		arbitrary_types_allowed = True


class NodeABC(ABC):
	"""Abstract interface for workflow nodes."""

	@abstractmethod
	def bind_llm(self, llm_dict: Dict[str, Runnable]):
		raise NotImplementedError

	@abstractmethod
	async def invoke(self, state, config: RunnableConfig = None) -> dict:
		raise NotImplementedError

	@abstractmethod
	def invoke_sync(self, state, config: RunnableConfig = None) -> dict:
		raise NotImplementedError


class RunnerABC(ABC):
	"""Abstract interface for workflow runners."""

	@abstractmethod
	async def run(self, initial_state: dict, config: RunnableConfig = None) -> dict:
		raise NotImplementedError


class BaseNode(NodeABC):
	class State(BaseModel):
		messages: list = []

	def __init__(
		self,
		messages,
		llm_type: str | LLMClass = LLMClass.REACT,
		name=""
	):
		self.llm_type = llm_type
		self.name = name
		self.messages = messages
		self.llm = None

	def bind_llm(self, llm_dict: Dict[str, Runnable]):
		"""Bind LLM from dictionary."""
		self.llm = llm_dict[self.llm_type]

	async def invoke(self, state: State, config: RunnableConfig = None) -> dict:
		"""Async invoke - returns update dict."""
		config = config or {}
		fmt = format_messages(self.messages, state)
		response = await self.llm.ainvoke(fmt, config)
		return {"messages": fmt + [response]}

	def invoke_sync(self, state: State, config: RunnableConfig = None) -> dict:
		"""Synchronous invoke for testing or non-async contexts."""
		return asyncio.get_event_loop().run_until_complete(
			self.invoke(state, config)
		)


class ScatterNode(BaseNode):
	"""Node that executes a workflow in parallel with per-instance state variations.

	Each branch runs the contained workflow with isolated state.
	Results are combined via gather() using reducers (messages append, others replace).

	Default behavior: scatters over state.items, giving each branch {"item": <value>}.
	Override scatter() for custom splitting logic.
	"""

	class State(BaseNode.State):
		items: List[Any] = []

	def __init__(
		self,
		nodes: "BaseWorkflow | BaseNode | Sequence[BaseNode]",
		name: str = ""
	):
		self.name = name
		self._workflow = None
		self._llm_dict = None
		# BaseNode compatibility - not used by ScatterNode directly
		self.messages = []
		self.llm_type = LLMClass.REACT
		self.llm = None

		# Normalize input to either a workflow or list of nodes
		if isinstance(nodes, BaseWorkflow):
			# Use existing workflow directly
			self._llm_dict = nodes.llm
			self.nodes = nodes.nodes
		elif isinstance(nodes, BaseNode):
			# Single node - wrap in list
			self.nodes = [nodes]
		elif isinstance(nodes, Sequence):
			# Sequence of nodes (original behavior)
			self.nodes = list(nodes)
		else:
			raise TypeError(
				f"nodes must be BaseWorkflow, BaseNode, or Sequence[BaseNode], "
				f"got {type(nodes).__name__}"
			)

	def bind_llm(self, llm_dict: Dict[str, Runnable]):
		"""Bind LLM and create internal workflow."""
		self._llm_dict = llm_dict
		self._workflow = BaseWorkflow(
			name=f"{self.name}_scatter",
			llm=llm_dict,
			node=self.nodes
		)

	def scatter(self, state: State) -> List[dict]:
		"""Split state into per-branch fragments. Default: one branch per item."""
		return [{"item": item} for item in state.items]

	def gather(self, results: List[dict]) -> dict:
		"""Combine results from parallel branches. Default uses reducers."""
		if not results:
			return {}

		combined = {}
		for result in results:
			for field, value in result.items():
				if field == "messages":
					reducer = _messages_reducer
				else:
					reducer = _default_reducer
				current = combined.get(field)
				combined[field] = reducer(current, value)

		return combined

	@polymorphic
	async def invoke(self, state: State, config: RunnableConfig = None) -> dict:
		"""Execute workflow in parallel for each scatter fragment."""
		config = config or {}

		# Get base state dict from the sliced state
		base_state = state.model_dump()

		# Get per-branch fragments
		fragments = self.scatter(state)

		async def run_branch(fragment: dict) -> dict:
			branch_state = {**base_state, **fragment}
			# Fresh runner per branch for state isolation
			runner = self._workflow.compile()
			return await runner.run(branch_state, config)

		# Execute all branches in parallel
		tasks = [run_branch(fragment) for fragment in fragments]
		results = await asyncio.gather(*tasks)

		return self.gather(results)

class WorkflowRunner(RunnerABC):
	"""Executes compiled workflow with local state accumulation.

	Workflows are basic blocks: nodes within a workflow share local state,
	and updates sync to global StateManager only at workflow completion.
	This provides deterministic execution within a workflow while allowing
	async updates between concurrent workflows.
	"""

	def __init__(self,
		name: str,
		nodes: List[BaseNode],
		STATE_MANAGER: TypeVar('STATE_MANAGER',bound=StateManager),
		full_state_class: Type[BaseModel]
	):
		self.name = name
		self.nodes = nodes
		self.manager = STATE_MANAGER()
		self.full_state_class = full_state_class

	def _apply_local_update(self, state: dict, update: dict) -> dict:
		"""Apply node update to local state using replace semantics.

		Within a workflow, we use replace (not reducers) since nodes
		return complete field values. Reducer semantics apply only
		when merging updates at the global level.
		"""
		new_state = dict(state)
		for field, value in update.items():
			new_state[field] = value
		return new_state

	async def stream(self, initial_state: dict, config: RunnableConfig = None) -> AsyncIterator[dict]:
		"""Execute nodes with local accumulation, yield state after each.

		The workflow captures a snapshot at start and accumulates state locally.
		Updates sync to global StateManager only after all nodes complete.
		"""
		config = config or {}

		# Capture initial snapshot from global state
		self.manager.init_snapshot(initial_state)

		# Local state accumulation (isolated from global during execution)
		local_state = self.manager.get_snapshot().to_dict()

		# Track all node updates for global sync at completion
		pending_updates: List[UpdateMessage] = []

		try:
			for node in self.nodes:
				# Create slice from local state for this node
				local_snapshot = Snapshot(local_state)
				state_slice = local_snapshot.get_slice(node.State)

				# Invoke node
				update = await node.invoke(state_slice, config)

				# Queue update for global sync later
				pending_updates.append(UpdateMessage(
					source=node.name,
					updates=update
				))

				# Apply update to local state (replace semantics)
				local_state = self._apply_local_update(local_state, update)

				yield {node.name: dict(local_state)}
		finally:
			# Sync all updates to global at workflow completion
			for update_msg in pending_updates:
				self.manager.submit_update_sync(update_msg)
			await self.manager.process_updates()

	async def run(self, initial_state: dict, config: RunnableConfig = None) -> dict:
		"""Execute workflow and return final state from global snapshot."""
		config = config or {}
		async for _ in self.stream(initial_state, config):
			pass  # Consume all updates
		# Return global state after reducers have been applied
		return self.manager.get_snapshot().to_dict()


class BaseWorkflow:
	class State(BaseModel):
		goal: str = ""

	def __init__(
		self,
		name: str,
		llm: Dict[str, Runnable],
		node: List[BaseNode]
	):
		self.name = name
		self.llm = llm
		self.nodes = node
		self._link(llm, node)
		self.full_state_class = self._build_state_class(node)

	def _link(self, llm: Dict[str, Runnable], nodes: List[BaseNode]):
		"""Bind LLMs and assign names to anonymous nodes."""
		anon = 0
		for n in nodes:
			n.bind_llm(llm)
			if n.name == "":
				n.name = f"anon-{anon}"
				anon += 1

	def _build_state_class(self, nodes: List[BaseNode]) -> Type[BaseModel]:
		"""Build composite state class from all node states.

		Orders by inheritance depth (most specific first) for valid MRO.
		"""
		seen = set()
		candidates = []
		for n in nodes:
			if n.State not in seen:
				seen.add(n.State)
				candidates.append(n.State)

		# Sort by MRO length descending - subclasses have longer MRO
		component = sorted(candidates, key=lambda c: len(c.__mro__), reverse=True)

		return type(
			f"{self.name}_full_state",
			(self.State, *component,),
			{}
		)

	def compile(self,
		STATE_MANAGER : TypeVar('STATE_MANAGER',bound=StateManager) = StateManager,
		WORKFLOW_RUNNER : TypeVar('WORKFLOW_RUNNER',bound=WorkflowRunner) = WorkflowRunner
	) -> WorkflowRunner:
		"""Compile workflow into executable runner.

		Args:
			STATE_MANAGER: The class (possibly dependency-injected) for managing state, must subclass StateManager
			WORKFLOW_RUNNER: The class (possibly dependency-injected) for running a workflow, must subclass WorkflowRunner
		"""
		return WORKFLOW_RUNNER(
			name=self.name,
			nodes=self.nodes,
			STATE_MANAGER=STATE_MANAGER,
			full_state_class=self.full_state_class
		)


class WorkflowChain(RunnerABC):
	"""Sequential execution of workflows with state passing.

	Runs workflows in order, passing state from one to the next.
	Supports field mapping to rename/transform fields between workflows.

	Example for distillation pipeline:
		chain = WorkflowChain(
			workflows=[planner, compressor, executor],
			mappings=[
				{"plan": "input_plan"},  # planner output -> compressor input
				{"compact": "intuition"},  # compressor output -> executor input
			]
		)
		result = await chain.run({"problem": "What is 2+2?"})
	"""

	def __init__(
		self,
		workflows: List[BaseWorkflow],
		mappings: Optional[List[Dict[str, str]]] = None
	):
		"""Initialize workflow chain.

		Args:
			workflows: Sequence of workflows to execute in order
			mappings: Optional list of field mappings, one per transition.
				Each mapping is a dict of {old_name: new_name} for field renaming.
				Length should be len(workflows) - 1.
		"""
		if not workflows:
			raise ValueError("WorkflowChain requires at least one workflow")

		self.workflows = workflows
		self.mappings = mappings or []

		# Validate mappings length
		if self.mappings and len(self.mappings) != len(workflows) - 1:
			raise ValueError(
				f"Expected {len(workflows) - 1} mappings for {len(workflows)} workflows, "
				f"got {len(self.mappings)}"
			)

	def _apply_mapping(self, state: dict, mapping: Dict[str, str]) -> dict:
		"""Apply field renaming from mapping."""
		result = dict(state)
		for old_name, new_name in mapping.items():
			if old_name in result:
				result[new_name] = result.pop(old_name)
		return result

	async def _execute_chain(self, initial_state: dict, config: RunnableConfig = None) -> AsyncIterator[tuple[dict, str]]:
		"""Execute workflows in sequence, yielding (state, workflow_name) after each."""
		config = config or {}
		state = dict(initial_state)

		for i, workflow in enumerate(self.workflows):
			runner = workflow.compile()
			state = await runner.run(state, config)
			yield state, workflow.name

			# Apply field mapping for transition to next workflow
			if i < len(self.mappings):
				state = self._apply_mapping(state, self.mappings[i])

	async def run(self, initial_state: dict, config: RunnableConfig = None) -> dict:
		"""Execute all workflows in sequence.

		Args:
			initial_state: Starting state dict
			config: Optional runnable config passed to each workflow

		Returns:
			Final state after all workflows complete
		"""
		state = initial_state
		async for state, _ in self._execute_chain(initial_state, config):
			continue  # consume all steps, keep final state
		return state

	async def stream(
		self,
		initial_state: dict,
		config: RunnableConfig = None
	) -> AsyncIterator[Dict[str, dict]]:
		"""Execute workflows and yield state after each completes.

		Yields:
			Dict with workflow name as key and state as value
		"""
		async for state, name in self._execute_chain(initial_state, config):
			yield {name: dict(state)}

	def run_sync(self, initial_state: dict, config: RunnableConfig = None) -> dict:
		"""Synchronous version of run for testing."""
		return asyncio.get_event_loop().run_until_complete(
			self.run(initial_state, config)
		)
