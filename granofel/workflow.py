import asyncio
from enum import Enum
from typing import List, Dict, Sequence, Callable, Any, Type, AsyncIterator

from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel


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
	REACT = "react"
	REASONING = "reasoning"


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
	"""Central state coordinator with update queue and reducers."""

	def __init__(self, initial_state: dict = None, reducers: dict = None):
		self._snapshot = Snapshot(initial_state or {}, version=0)
		self._queue: asyncio.Queue[UpdateMessage] = asyncio.Queue()
		self._reducers: Dict[str, Callable[[Any, Any], Any]] = {
			"messages": _messages_reducer,
		}
		if reducers:
			self._reducers.update(reducers)

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


# --- Node and Workflow ---

class JITConfig(BaseModel):
	"""Just-in-time configuration for node compilation."""
	name: str
	PREV: str
	NEXT: str
	llm: Dict[str, Runnable]

	class Config:
		arbitrary_types_allowed = True


class BaseNode:
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
		config = config or {}
		fmt = format_messages(self.messages, state)
		response = self.llm.invoke(fmt, config)
		return {"messages": fmt + [response]}


class WorkflowRunner:
	"""Executes compiled workflow with StateManager coordination."""

	def __init__(self, name: str, nodes: List[BaseNode], manager: StateManager, full_state_class: Type[BaseModel]):
		self.name = name
		self.nodes = nodes
		self.manager = manager
		self.full_state_class = full_state_class

	async def stream(self, initial_state: dict, config: RunnableConfig = None) -> AsyncIterator[dict]:
		"""Execute nodes in order, yield state updates after each."""
		config = config or {}
		self.manager.init_snapshot(initial_state)

		for node in self.nodes:
			snapshot = self.manager.get_snapshot()
			state_slice = snapshot.get_slice(node.State)

			update = await node.invoke(state_slice, config)

			self.manager.submit_update_sync(UpdateMessage(
				source=node.name,
				updates=update
			))
			await self.manager.process_updates()

			yield {node.name: self.manager.get_snapshot().to_dict()}

	async def run(self, initial_state: dict, config: RunnableConfig = None) -> dict:
		"""Execute workflow and return final state."""
		config = config or {}
		final = None
		async for update in self.stream(initial_state, config):
			final = update
		if final:
			return list(final.values())[0]
		return initial_state


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
		"""Build composite state class from all node states."""
		used = set()
		component = []
		for n in nodes:
			if n.State not in used:
				used.add(n.State)
				component.append(n.State)
		return type(
			f"{self.name}_full_state",
			(self.State, *component,),
			{}
		)

	def compile(self) -> WorkflowRunner:
		"""Compile workflow into executable runner."""
		manager = StateManager()
		return WorkflowRunner(
			name=self.name,
			nodes=self.nodes,
			manager=manager,
			full_state_class=self.full_state_class
		)
