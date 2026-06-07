import json
from pathlib import Path
from langchain.tools import tool

_PRICES_FILE = Path(__file__).parent / "construction_tasks_prices.json"

def _get_construction_tasks_prices() -> dict:
    with open(_PRICES_FILE, "r") as f:
        return json.load(f)

def get_task_price(task_name : str):
    tasks_dict = _get_construction_tasks_prices()
    return tasks_dict[task_name]

def get_available_tasks():
    return _get_construction_tasks_prices().keys()

@tool(name_or_callable="get_task_price")
def get_task_price_tool(task_name : str):
    """
    Returns the price of the given task

    Input: Name of the task to find the price of
    Output: Price of the given task
    """
    return get_task_price(task_name)

@tool(name_or_callable="get_available_tasks")
def get_available_tasks_tool():
    """
    Returns a list of the names of all the available construction tasks
    """
    return [task for task in get_available_tasks()]


