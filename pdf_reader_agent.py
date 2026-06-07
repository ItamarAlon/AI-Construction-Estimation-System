from pathlib import Path
import sys
import base64
import re

from pywin.framework.toolmenu import tools

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
import pymupdf as fitz


def _pdf_to_content_blocks(pdf_path: str) -> list:
    """Render every page of a PDF as an image and return content blocks."""
    path = Path(pdf_path.strip("'\""))
    if not path.exists():
        return [{"type": "text", "text": f"File not found: {pdf_path}"}]

    doc = fitz.open(str(path))
    blocks = [{"type": "text", "text": f"PDF: {path.name} ({len(doc)} page(s))"}]
    for i, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            #print("written pdf text: ", text)
            blocks.append({"type": "text", "text": f"Page {i} extracted text:\n{text}"})
        # Render at 4× resolution — needed for small Hebrew text and fine plan details
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        blocks.append({"type": "text", "text": f"Page {i} image:"})
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    doc.close()
    return blocks


def _build_input(user_text: str) -> dict:
    """If the message contains a PDF path, embed the page images in the user message."""
    pdf_match = re.search(r'["\']?[\w\\/: .()-]+\.pdf["\']?', user_text, re.IGNORECASE)

    if pdf_match:
        pdf_blocks = _pdf_to_content_blocks(pdf_match.group())
        content = [{"type": "text", "text": user_text}] + pdf_blocks
    else:
        content = user_text

    return {"messages": [{"role": "user", "content": content}]}

@tool
def sum_numbers(numbers : list[float]):
    """
    Input: List of numbers
    Output: Sum of numbers
    """
    print("numbers to sum: ", numbers)
    return sum(numbers)

@tool
def multiply_numbers(num1, num2):
    """
    Input: 2 numbers
    Output: num1 * num2
    """
    print(f"{num1}*{num2}={num1*num2}")
    return num1 * num2


model = ChatOpenAI(
    model="openai/gpt-4o",   # vision-capable model needed to analyse the page images
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
        "The tasks can be explicitly mentioned in the pdf, or can be inferred from it's content."
        # "Understand which items in the pdf belong to what task. For example what walls are for destruction, and what are for construction."
        # "To do that, look at the colors of the items."
        "Use the tool 'get_available_tasks' to get the list of available constructions tasks"

        "After detecting the tasks, you need to plan how to calculate the cost of executing each task."
        "To do that, look at the entire pdf. Tell me which item belong to which task (The lengths)."
        #"For example, if the tasks are destroying walls and constructing walls, tell me which walls (their lengths) are for construction and which ones are for destruction."
        "Do that by looking at the legend box (if there is one), which tell you which color means what task."
        "After that, you should look for all appearances of the color in the map. All the colored items belong to that specific task, and ONLY those items."

        "After planning, calculate the cost of each task."
        "To do that, use the tool 'get_task_price'."
        "Give the tool the exact name of the task (which you got from the 'get_available_tasks' tool), and it will give you it's price."
        "For some tasks, the price will be given per meter (for example wall demolition price per meter of wall to destroy)."
        "And a task can can be executed multiple times in the pdf (for example destroy multiple kitchens)."
        "Either way, calculate the final price of executing all the tasks, by counting the amount of times a task will be executed (using sum),"
        "and multiplying by the price of the task."
        "For example, count how many meters of wall to destroy in total (by summing the lengths of all walls marked for destruction)"
        ", and multiply by the wall demolition price per meter."

        "Use 'sum_numbers' tool for finding sum of multiple numbers"
        "Use 'multiply_numbers' for multiplying 2 numbers"
    )
).with_memory().build()


if __name__ == "__main__":
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        raw = agent.invoke(_build_input(user_input), config=agent._config, version="v2")
        print("Agent:", agent._process_result(raw))
