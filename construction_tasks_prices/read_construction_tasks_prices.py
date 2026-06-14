import json
from pathlib import Path
from langchain.tools import tool

_PRICES_FILE = Path(__file__).parent / "construction_tasks_prices.json"

def get_construction_tasks_prices() -> dict:
    with open(_PRICES_FILE, "r") as f:
        return json.load(f)

def get_task_price(task_name : str):
    tasks_dict = get_construction_tasks_prices()
    return tasks_dict[task_name]

def get_available_tasks():
    return get_construction_tasks_prices().keys()

def remove_task(name: str) -> None:
    tasks = get_construction_tasks_prices()
    if name not in tasks:
        raise ValueError(f"Task '{name}' does not exist.")
    del tasks[name]
    with open(_PRICES_FILE, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

def update_task_price(name: str, price: float) -> None:
    tasks = get_construction_tasks_prices()
    if name not in tasks:
        raise ValueError(f"Task '{name}' does not exist.")
    tasks[name] = price
    with open(_PRICES_FILE, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

def add_task(name: str, price: float, per_meter: bool) -> None:
    task_name = f"{name} (per meter)" if per_meter else name
    tasks = get_construction_tasks_prices()
    if task_name in tasks:
        raise ValueError(f"Task '{task_name}' already exists with price {tasks[task_name]}.")
    tasks[task_name] = price
    with open(_PRICES_FILE, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

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


