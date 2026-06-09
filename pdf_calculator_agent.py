from pathlib import Path
import sys

from langchain.tools import tool
from construction_tasks_prices.read_construction_tasks_prices import get_task_price_tool
from helpers.wall_measurement_tool import get_wall_lengths_by_color

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
    tools=[get_task_price_tool, multiply_numbers, get_wall_lengths_by_color],
    system_prompt=(
        "You are a construction cost estimator. "
        "The user will give you a list of detected construction tasks and a PDF path. "
        "For each task, calculate its cost as follows:\n"
        "1. Call 'get_wall_lengths_by_color' with the PDF path and the wall color that "
        "corresponds to that task (e.g. 'yellow' for demolition, 'red' for new construction). "
        "This returns the exact total wall length in meters.\n"
        "2. Call 'get_task_price' with the exact task name to get the unit price.\n"
        "3. Call 'multiply_numbers' to compute total cost = length × unit price.\n"
        "Report each task's total length, unit price, and cost. "
        "Finish with an overall grand total.\n\n"
        "IMPORTANT: Do not write a final summary until you have called "
        "'get_task_price' and 'multiply_numbers' for every task. "
        "If tasks remain unpriced, your next output must be a tool call."
    )
).with_memory().pdf_reader().with_todos().build()

PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"

if __name__ == "__main__":
    question = PDF_PATH
    while question != "exit":
        answer = agent.run(question)
        print("Agent:", answer)
        question = input("You:")
