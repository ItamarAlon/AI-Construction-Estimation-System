#from langgraph.checkpoint.memory import InMemorySaver
try:
    from .SkillMiddleware import SkillMiddleware
except ImportError:
    # Support running files directly from the skill_loader folder.
    from SkillMiddleware import SkillMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents import create_agent

DEFAULT_SYSTEM_PROMPT = """
"""
PROMPT_ADDITION = """
"""

class SkilledAgent:
    def __init__(self, *args, **kwargs):
        system_prompt = kwargs.pop("system_prompt", "")     
        user_middleware = kwargs.pop("middleware", None)

        if user_middleware is None:
            user_middleware = []
        elif not isinstance(user_middleware, (list, tuple)):
            user_middleware = [user_middleware]

        combined_middleware = [SkillMiddleware(), ToolCallLimitMiddleware(run_limit=12, exit_behavior="end"), *user_middleware]
        kwargs["middleware"] = combined_middleware
        kwargs["system_prompt"] = (system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT) + f"\n\n{PROMPT_ADDITION}"
        self.agent = create_agent(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return self.agent.invoke(*args, **kwargs)

    def run(self, text: str) -> str:
        result = self.invoke(input={
        "messages": [
            {
                "role": "user",
                "content": text,
            }
        ]
    })
        return result["messages"][-1].content


import inspect
from langgraph.pregel.main import Pregel

# Copy signatures from callable definitions.
SkilledAgent.__init__.__signature__ = inspect.signature(create_agent)
SkilledAgent.invoke.__signature__ = inspect.signature(Pregel.invoke)
