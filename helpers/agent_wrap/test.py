from pathlib import Path
import sys
import asyncio

HELPERS_ROOT = Path(__file__).resolve().parent.parent   # helpers/
PROJECT_ROOT = HELPERS_ROOT.parent                      # repo root (get_key lives here)
for _p in [str(PROJECT_ROOT), str(HELPERS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AgentBuilder import AgentBuilder
from Agent import Agent
from langchain_openai import ChatOpenAI
from get_key import get_openrouter_api_key

system_prompt = """
You are a helpful assistant.

You have access to a weather tool. Use it when the user asks you to do something weather related
"""

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.2,
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key(),
)
# agent = AgentBuilder(model=model, system_prompt=system_prompt)\
# .mcp({"spotify": {
#     "args": "C:\\Users\\Alon\\Desktop\\spotify mcp server\\spotify-bulk-actions-mcp\\spotify_bulk_actions_mcp\\server.py", 
#     "command":"C:\\Users\\Alon\\Desktop\\spotify mcp server\\spotify-bulk-actions-mcp\\venv\\Scripts\\python.exe"}})\
# .with_memory()\
# .build()
from tool import get_weather
agent = AgentBuilder(model=model, system_prompt=system_prompt, tools=[get_weather])\
    .ask_user_when_needed()\
    .build()

cont = True

while cont:
    user_input = input("Enter a command: ")
    if user_input == "exit":
        cont = False
    else:
        response = agent.srun(user_input)
        print(response)
