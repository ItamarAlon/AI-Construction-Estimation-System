from langchain.agents import middleware
from SkilledAgent import SkilledAgent
from langchain_openai import ChatOpenAI
from pathlib import Path
import sys

# Add project root (parent of "skill loader") to import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from get_key import get_openrouter_api_key

system_prompt = """
You are a helpful assistant that helps with aritmatic calculations.
"""

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.2,
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key(),
)
agent = SkilledAgent(model=model, system_prompt=system_prompt)
response = agent.run("add 2 and 3")
print(response)

