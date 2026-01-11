import asyncio
from langchain.chat_models import init_chat_model
from granofel.workflow import BaseNode, BaseWorkflow


class ProfessionQuery(BaseNode):
	class State(BaseNode.State):
		profession: str
		user_prompt: str


async def main():
	start_state = {
		"profession": "A 19th century bonapartist former noble",
		"user_prompt": "So who are you guv?"
	}

	profession = ProfessionQuery([
		("system", "Answer the user's question."),
		("system", "Talk like {state.profession} when doing so"),
		("human", "{state.user_prompt}"),
	])

	response = ProfessionQuery([
		lambda x: x,
		("system", "Now, present the opposing point of view."),
	])

	key = open(".env", "r").readline().strip()

	llm = {
		"react": init_chat_model("openai:gpt-4o", api_key=key)
	}

	wf = BaseWorkflow("test", llm, node=[profession, response])

	runner = wf.compile()
	async for msg in runner.stream(start_state):
		print(msg)


if __name__ == "__main__":
	asyncio.run(main())
