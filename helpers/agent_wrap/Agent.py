from langchain.agents import create_agent
from langgraph.types import Command
import asyncio
import uuid


class Agent:
    def __init__(self, *args, **kwargs):
        thread_id = str(uuid.uuid4())
        self._config = {"configurable": {"thread_id": thread_id}}
        self.interruption = False
        self.agent = create_agent(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return self.agent.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self.agent.ainvoke(*args, **kwargs)

    def run(self, text: str) -> str:
        return self._process_result(self.invoke(self._build_input(text), config=self._config, version="v2"))

    def run_blocks(self, blocks: list) -> str:
        """Like run() but accepts a pre-built content list (text + image blocks)."""
        msg = {"messages": [{"role": "user", "content": blocks}]}
        return self._process_result(self.invoke(msg, config=self._config, version="v2"))

    async def arun(self, text: str) -> str:
        return self._process_result(await self.ainvoke(self._build_input(text), config=self._config, version="v2"))

    def srun(self, text: str) -> str:
        return asyncio.run(self.arun(text))

    def is_interrupted(self):
        return self.interruption

    def _build_input(self, text: str):
        if self.interruption:
            return Command(resume={"decisions": [{"type": "respond", "message": text}]})
        return self._get_message(text)

    def _process_result(self, result) -> str:
        if getattr(result, "interrupts", None):
            self.interruption = True
            return result.interrupts[0].value.get("action_requests")[0].get("args").get("question")
        self.interruption = False
        return result["messages"][-1].content

    def _get_message(self, text: str) -> dict:
        return {
        "messages": [
            {
                "role": "user",
                "content": text,
            }
        ]
    }