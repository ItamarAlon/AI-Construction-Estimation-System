import json
from langchain.tools import tool

TASKS_PRICES_JSON_FILENAME = ""

def _get_construction_tasks_prices() -> dict:
    with open(TASKS_PRICES_JSON_FILENAME + ".json", "r") as f:
        return json.load(f)

def get_task_prices(task_name : str):
    tasks_dict = _get_construction_tasks_prices()
    return tasks_dict[task_name]

def get_available_tasks():
    return _get_construction_tasks_prices().keys()

@tool(name_or_callable="get_task_prices")
def get_task_prices_tool(task_name : str):
    """
    Returns the price of the given task

    Input: Name of the task to find the price of
    Output: Price of the given task
    """
    return get_task_prices(task_name)

@tool(name_or_callable="get_available_tasks")
def get_available_tasks_tool():
    """
    Returns a list of the names of all the available construction tasks
    """
    return [task for task in get_available_tasks()]


