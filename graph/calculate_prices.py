"""Plain-Python pricing step (no LLM, no PDF).

The estimation agent reports only QUANTITIES (a task-name -> quantity JSON).
This module turns those into costs using the known price list, so the agent
never sees prices or does arithmetic itself.
"""
import json
import re

from construction_tasks_prices.read_construction_tasks_prices import (
    get_construction_tasks_prices,
)


def extract_quantities(agent_output: str) -> dict:
    """Pull the quantities JSON the agent emits as its last ```json block.

    Falls back to the last {...} object in the text if no code block is found.
    """
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", agent_output, re.DOTALL)
    if blocks:
        return json.loads(blocks[-1])
    start, end = agent_output.rfind("{"), agent_output.rfind("}")
    if start != -1 and end > start:
        return json.loads(agent_output[start:end + 1])
    raise ValueError("No quantities JSON found in agent output.")


def price_quantities(quantities: dict) -> dict:
    """Cost each task: quantity * unit_price. Returns a structured breakdown."""
    prices = get_construction_tasks_prices()
    line_items = []
    grand_total = 0.0
    for name, qty in quantities.items():
        unit_price = prices.get(name)
        if unit_price is None:
            line_items.append({"task": name, "quantity": qty, "error": "unknown task name"})
            continue
        cost = qty * unit_price
        grand_total += cost
        line_items.append({
            "task": name,
            "quantity": qty,
            "unit_price": unit_price,
            "cost": cost,
        })
    return {"line_items": line_items, "grand_total": grand_total}


def format_report(breakdown: dict) -> str:
    lines = ["Cost estimate:"]
    for item in breakdown["line_items"]:
        if "error" in item:
            lines.append(f"  - {item['task']}: {item['error']} (qty {item['quantity']})")
        else:
            lines.append(
                f"  - {item['task']}: {item['quantity']} x {item['unit_price']} "
                f"= {item['cost']:.2f}"
            )
    lines.append(f"Grand total: {breakdown['grand_total']:.2f}")
    return "\n".join(lines)


def price_from_agent_output(agent_output: str) -> dict:
    """Convenience: extract quantities from agent text and price them."""
    return price_quantities(extract_quantities(agent_output))
