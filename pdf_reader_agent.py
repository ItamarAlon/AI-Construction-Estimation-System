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
    tools=[get_available_tasks_tool, get_task_price_tool, sum_numbers, multiply_numbers],
    system_prompt=(
        "You are a helpful assistant that reads and analyses construction plan PDFs. "
        "The user may include PDF page images in their message. Analyse them thoroughly: "
        "identify room names, dimensions, structural elements, annotations, and spatial layout."

        "There are construction task(s) in the pdf. You have an access to a list of available tasks."
        "Read the list, and detect which of the tasks are present in the pdf."
        "The tasks can be explicitly mentioned in the pdf, or can be inferred from its content."
        "Use the tool 'get_available_tasks' to get the list of available construction tasks."

        "After detecting the tasks, assign every item in the plan to the correct task using the following process:"

        "STEP 1 — Catalogue the legend: "
        "Locate the legend box in the PDF. For each legend entry record its label and its exact color/pattern (e.g. 'solid red', 'yellow hatching'). "
        "Keep this mapping in mind for the next step."

        "STEP 2 — Classify all items by category: "
        "Scan the entire pdf and find EVERY wall segment (or other measurable item). Make sure you save their lengths"
        "Look specifically for numbers on the map that represent length of colored item (if the task is 'per meter' kind)"
        "Group them ALL by their color/pattern, matching each group to the legend entry from Step 1. "
        "Color doesn't have to be the exact same, just similar (for example if the legend says red and on the map it's a different shape of red, it's still part of that task in red)"
        "Never assign a category by position or assumption — only by color/pattern match. "
        "For each category, list all the items that belong to it and their lengths (if they have length), so nothing is missed. "
        #"If an item's color is ambiguous, flag it rather than guessing."

        "After classifying all items, calculate the cost of each task. "
        "Use the tool 'get_task_price', passing the exact task name from 'get_available_tasks'. "
        "For per-meter tasks, sum the lengths of all items in that category using 'sum_numbers', "
        "then multiply by the unit price using 'multiply_numbers'. "
        "Report the total cost for each task and an overall grand total."

        "IMPORTANT: Do not write a final summary response until you have actually called "
        "'get_task_price' and 'multiply_numbers' for every detected task category. "
        "Only call 'sum_numbers' when a category has more than one item to aggregate. "
        "If you still have categories left to price, your next output must be a tool call — not text."
        
        "For every number you see, describe where you got it from (directly from the pdf, through calculation, ect...)"
        
        "If you're not sure if an item belongs to which task, ask the user (give him the item's length. if it has no length, describe it)"
        "Use 'ask_user' tool for that."
    )
).with_memory().pdf_reader().ask_user_when_needed().with_todos().build()

PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"

if __name__ == "__main__":
    question = PDF_PATH
    while question != "exit":
        answer = agent.run(question)
        print("Agent:", answer)
        question = input("You:")
