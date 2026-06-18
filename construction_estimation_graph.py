import re
import pymupdf as fitz
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from pdf_calculator_agent import agent as estimation_agent
from wall_measurement_tool import measure_segments_by_id
from detect_plan_colors import list_present_colors, format_palette
from calculate_prices import extract_quantities, price_quantities, format_report
from logs.write_logs import write_logs


class State(TypedDict):
    pdf_path: str
    palette: str            # detected color palette (hex codes) fed to the agent
    agent_output: str       # raw agent text (detection + classification + segment-assignment JSON)
    classifications: dict   # parsed: task -> {color, page, ids} or {count}
    quantities: dict        # task name -> measured quantity (meters or count)
    breakdown: dict         # priced line items + grand total
    result: str             # human-readable cost report


def _extract_classifications(agent_output: str) -> dict:
    """Pull the classification JSON the agent emits as its last ```json block."""
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", agent_output, re.DOTALL)
    if blocks:
        import json
        return json.loads(blocks[-1])
    import json
    start, end = agent_output.rfind("{"), agent_output.rfind("}")
    if start != -1 and end > start:
        return json.loads(agent_output[start:end + 1])
    raise ValueError("No classification JSON found in agent output.")


def run_detect_colors(state: State) -> dict:
    """Plain Python: list the real colors per page so the agent picks exact hex codes."""
    pdf_path = state["pdf_path"]
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()
    sections = []
    for p in range(1, n_pages + 1):
        sections.append(f"--- Page {p} ---\n{format_palette(list_present_colors(pdf_path, p))}")
    palette = "\n\n".join(sections)
    write_logs("palette: " + palette)
    return {"palette": palette}


def run_estimation(state: State) -> dict:
    # Path on its OWN first line (the image-injection middleware extracts it by
    # regex; text on the same line would corrupt the match). Palette follows.
    agent_input = (
        f"{state['pdf_path']}\n\n"
        f"{state['palette']}\n\n"
        "Use the exact hex codes above when calling list_colored_segments / "
        "measure_segments_by_id."
    )
    agent_output = estimation_agent.run(agent_input)
    classifications = _extract_classifications(agent_output)
    write_logs("agent_output: " + agent_output)
    #write_logs("classifications: " + classifications)
    return {"agent_output": agent_output, "classifications": classifications}

def run_measure(state: State) -> dict:
    """Call measure_segments_by_id for per-meter tasks; pass counts through for per-unit."""
    pdf_path = state["pdf_path"]
    quantities: dict = {}
    for task_name, info in state["classifications"].items():
        if "ids" in info:
            result_text = measure_segments_by_id(pdf_path, info["color"], info["ids"], info.get("page", 1))
            m = re.search(r"Total:\s*([\d.]+)\s*m", result_text)
            quantities[task_name] = float(m.group(1)) if m else 0.0
        elif "count" in info:
            quantities[task_name] = info["count"]
    write_logs("quantities: " + str(quantities))
    return {"quantities": quantities}


def run_pricing(state: State) -> dict:
    breakdown = price_quantities(state["quantities"])
    return {"breakdown": breakdown, "result": format_report(breakdown)}

graph = (
    StateGraph(State)
    .add_node("detect_colors", run_detect_colors)
    .add_node("estimation", run_estimation)
    .add_node("measure", run_measure)
    .add_node("pricing", run_pricing)
    .add_edge(START, "detect_colors")
    .add_edge("detect_colors", "estimation")
    .add_edge("estimation", "measure")
    .add_edge("measure", "pricing")
    .add_edge("pricing", END)
    .compile()
)

#PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"
PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\הריסה (1).pdf"
#PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\בנייה (1).pdf"
#PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\סט תוכניות (1).pdf"

if __name__ == "__main__":
    write_logs("-----------------------------")
    result = graph.invoke({"pdf_path": PDF_PATH})
    print("Result:", result["result"])
    write_logs("Result:" + result["result"])
