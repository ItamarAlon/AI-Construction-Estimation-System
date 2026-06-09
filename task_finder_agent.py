from pathlib import Path
import sys

from langchain.tools import tool
from construction_tasks_prices.read_construction_tasks_prices import get_available_tasks_tool
from construction_tasks_prices.read_construction_tasks_prices import get_task_price_tool

# File is at repo root; add helpers/ and helpers/agent_wrap/ so package and
# bare imports inside AgentBuilder resolve correctly.
_root = Path(__file__).resolve().parent
for _p in [str(_root / "helpers"), str(_root / "helpers" / "agent_wrap")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_openai import ChatOpenAI
from helpers.agent_wrap.AgentBuilder import AgentBuilder
from get_key import get_openrouter_api_key

model = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.2,
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key()
)

agent = AgentBuilder(
    model=model,
    tools=[get_available_tasks_tool],
    system_prompt=(
        "You are a helpful assistant that reads and analyses construction plan PDFs. "
        "The user may include PDF page images in their message. Analyse them thoroughly: "
        "identify room names, dimensions, structural elements, annotations, and spatial layout."

        "There are construction task(s) in the pdf. You have an access to a list of available tasks."
        "Read the list, and detect which of the tasks are present in the pdf."
        "The tasks can be explicitly mentioned in the pdf, or can be inferred from its content."
        "Use the tool 'get_available_tasks' to get the list of available construction tasks."
        
        "Also tell the user how can he tell the tasks apart in the map found in the pdf."
        "Which parts of the map are part of each task?"
        "The output will be passed to another agent that will calculate the cost of each task, "
        "so write a through explanation for each task that another agent will understand."
        
        "The output should be in this json format:"
        "{Task 1 Name: explanation on how to find task 1 on the map,"
        "Task 2 Name: explanation on how to find task 2 on the map,...}"
    )
).with_memory().pdf_reader().build()

PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"

if __name__ == "__main__":
    question = PDF_PATH
    while question != "exit":
        answer = agent.run(question)
        print("Agent:", answer)
        question = input("You:")
