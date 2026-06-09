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


@tool
def sum_numbers(numbers: list[float]) -> float:
    """Input: list of numbers. Output: their sum."""
    print("numbers to sum:", numbers)
    return sum(numbers)


@tool
def multiply_numbers(num1: float, num2: float) -> float:
    """Input: two numbers. Output: num1 * num2."""
    print(f"{num1} * {num2} = {num1 * num2}")
    return num1 * num2


model = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.2,
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key()
)

agent = AgentBuilder(
    model=model,
    tools=[get_task_price_tool, sum_numbers, multiply_numbers],
    system_prompt=(
        "You are a helpful assistant that reads and analyses construction plan PDFs. "
        "The user may include PDF page images in their message. Analyse them thoroughly: "
        "identify room names, dimensions, structural elements, annotations, and spatial layout."
        
        # "You are given a list of construction tasks found in the pdf, and explanations for how to tell them apart on the map."
        # "You need to calculate the cost of each construction task."
        # "Use the tool 'get_task_price', passing the exact task name from 'get_available_tasks'. "
        # "For per-meter tasks, sum the lengths of all items in that category using 'sum_numbers', "
        # "then multiply by the unit price using 'multiply_numbers'. "
        # "Report the total cost for each task and an overall grand total."
        # "To find all items in the category on the map, read the explanation given for you on the list."
        # "If the items have lengths (like walls), the length of each item on the map, "
        # "is the number that appears the closest to the item in question (and only that number)."
        # "For example if you are looking for red walls, the number that appears right next to the red wall in the length of the red wall, "
        # "The one that is the closest distance to the red wall (more than any other number)."
        
        "sum all the lengths of yellow walls (only yellow walls, no walls of other color). using the 'sum numbers' tool"
        "To find the length of each yellow wall segment, read the dimension annotation "
        "that is directly attached to that specific segment (the number with an arrow or "
        "tick marks pointing to its two endpoints). Do NOT use room dimensions or any "
        "annotation that spans across multiple elements. If a segment has no annotation, "
        "state that its length is unknown rather than estimating."

    )
).with_memory().pdf_reader().build()

PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"

if __name__ == "__main__":
    question = PDF_PATH
    while question != "exit":
        answer = agent.run(question)
        print("Agent:", answer)
        question = input("You:")
