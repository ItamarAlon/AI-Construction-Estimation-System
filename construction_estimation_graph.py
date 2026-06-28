import json
import re
import pymupdf as fitz
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from pdf_calculator_agent import agent as estimation_agent
from construction_tasks_prices.read_construction_tasks_prices import get_available_tasks
from wall_measurement_tool import (
    list_colored_segments,
    measure_segments_by_id,
    measure_task_groups,
    count_task_groups,
)
from detect_plan_colors import list_present_colors, format_palette
from calculate_prices import price_quantities, format_report
from render_annotations import render_annotations
from logs.write_logs import write_logs



class State(TypedDict):
    pdf_path: str
    pages: list                       # optional 1-indexed pages to analyze; empty/None = all
    palette: str                      # detected color palette (hex codes)
    segment_blocks: list              # pre-computed listing content blocks
    estimation_agent_output: str      # raw agent text (classification JSON + reasoning)
    agent_classifications: dict       # parsed: task -> {groups:[{color,page,ids}]} or {count:N}
    measured_quantities: dict         # task name -> measured quantity (meters or count)
    annotations: dict                 # PNGs from PDFs + legend marking each task on the plan
    calculated_prices_breakdown: dict # priced line items + grand total
    result: str                       # human-readable cost report


def _extract_classifications(agent_output: str) -> dict:
    """Pull the classification JSON the agent emits as its last ```json block."""
    blocks = re.findall(r"```json\s*(.*?)```", agent_output, re.DOTALL)
    if blocks:
        return json.loads(blocks[-1].strip())
    start, end = agent_output.rfind("{"), agent_output.rfind("}")
    if start != -1 and end > start:
        return json.loads(agent_output[start:end + 1])
    raise ValueError("No classification JSON found in agent output.")


def run_detect_colors(state: State) -> dict:
    """List the real colors per page so the agent picks exact hex codes."""
    pdf_path = state["pdf_path"]
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()
    selected = state.get("pages")
    pages = [p for p in (selected or range(1, n_pages + 1)) if 1 <= p <= n_pages]
    if not pages:
        pages = list(range(1, n_pages + 1))
    write_logs(f"pages: requested={selected or 'all'}, analyzing={pages}")
    sections = []
    for p in pages:
        sections.append(f"--- Page {p} ---\n{format_palette(list_present_colors(pdf_path, p))}")
    palette = "\n\n".join(sections)
    write_logs("palette: " + palette)
    # Persist the normalized page selection so later nodes (and the PDF injection
    # middleware) only render/process those pages.
    return {"palette": palette, "pages": pages if len(pages) < n_pages else []}


def _parse_palette_colors(palette: str) -> list[tuple[str, int]]:
    """Extract (hex_code, page_number) pairs from the palette string."""
    pairs: list[tuple[str, int]] = []
    current_page = 1
    for line in palette.splitlines():
        page_m = re.match(r"---\s*Page\s*(\d+)\s*---", line.strip())
        if page_m:
            current_page = int(page_m.group(1))
        hex_m = re.match(r"\s*(#[0-9a-fA-F]{6})", line)
        if hex_m:
            pairs.append((hex_m.group(1), current_page))
    return pairs


def run_enumerate(state: State) -> dict:
    """Pre-compute segment listings for every palette color — no LLM needed.

    Calls list_colored_segments for each (color, page) pair and collects the
    content blocks (text attribute lines + inline crop images). These are passed
    directly to the agent's initial message so it never needs to call the listing
    tool itself.
    """
    pdf_path = state["pdf_path"]
    pairs = _parse_palette_colors(state["palette"])
    all_blocks: list[dict] = []
    for color, page in pairs:
        result = list_colored_segments.func(pdf_path, color, page)
        all_blocks.append({
            "type": "text",
            "text": f"\n=== Segments for {color}, page {page} ===",
        })
        if isinstance(result, str):
            all_blocks.append({"type": "text", "text": result})
        else:
            all_blocks.extend(result)
    write_logs(f"enumerate: {len(pairs)} color/page pair(s), {len(all_blocks)} total blocks")
    return {"segment_blocks": all_blocks}


def run_estimation(state: State) -> dict:
    """Run the classification agent with pre-enumerated segment data.

    The preamble text block contains the PDF path, which the pdf_injection_middleware
    detects and replaces with rendered page images before the first model call.
    """
    pdf_path = state["pdf_path"]

    pages = state.get("pages") or []
    pages_directive = f"[render_pages: {','.join(str(p) for p in pages)}]\n" if pages else ""

    task_list = "\n".join(f"  - {t}" for t in get_available_tasks())
    preamble = (
        f"{pdf_path}\n{pages_directive}\n"
        f"AVAILABLE TASKS (use exact names in your JSON output):\n{task_list}\n\n"
        f"{state['palette']}\n\n"
        "The segment listings for all palette colors (attributes + zoomed crops) "
        "are provided below. Use the exact hex codes from the palette in your JSON output."
    )

    content = [{"type": "text", "text": preamble}] + state["segment_blocks"]

    agent_output = estimation_agent.run_blocks(content)
    classifications = _extract_classifications(agent_output)
    write_logs("estimation_agent_output: " + agent_output)
    return {"estimation_agent_output": agent_output, "agent_classifications": classifications}


def _measure_group(pdf_path: str, group: dict) -> float:
    """Measure one {color, page, ids} group; return meters (0.0 if unparseable)."""
    result_text = measure_segments_by_id(
        pdf_path, group["color"], group["ids"], group.get("page", 1)
    )
    m = re.search(r"Total:\s*([\d.]+)\s*m", result_text)
    return float(m.group(1)) if m else 0.0


def run_measure(state: State) -> dict:
    """Reduce each task's tagged segments to a quantity.

    Per-meter tasks (name ends '(per meter)') sum segment lengths. Per-unit
    tasks cluster their tagged segments into discrete items by connectivity and
    count them. A task given only {count: N} (no taggable symbol) passes through.
    """
    pdf_path = state["pdf_path"]
    quantities: dict = {}
    for task_name, info in state["agent_classifications"].items():
        groups = info.get("groups") or ([info] if "ids" in info else None)
        if groups is not None:
            if task_name.strip().lower().endswith("(per meter)"):
                quantities[task_name] = measure_task_groups(pdf_path, groups)
            else:
                quantities[task_name] = count_task_groups(pdf_path, groups)
        elif "count" in info:                       # per-unit item with no segments to tag
            quantities[task_name] = info["count"]
    write_logs("measured_quantities: " + str(quantities))
    return {"measured_quantities": quantities}


def run_annotate(state: State) -> dict:
    """Draw the agent's task assignments onto the plan (per-page PNGs for the UI)."""
    annotations = render_annotations(state["pdf_path"], state["agent_classifications"])
    write_logs(f"annotations: {len(annotations['pages'])} page(s) marked; "
               f"legend={annotations['legend']}")
    return {"annotations": annotations}


def run_pricing(state: State) -> dict:
    breakdown = price_quantities(state["measured_quantities"])
    return {"calculated_prices_breakdown": breakdown, "result": format_report(breakdown)}


graph = (
    StateGraph(State)
    .add_node("detect_colors", run_detect_colors)
    .add_node("enumerate", run_enumerate)
    .add_node("estimation", run_estimation)
    .add_node("measure", run_measure)
    .add_node("annotate", run_annotate)
    .add_node("pricing", run_pricing)
    .add_edge(START, "detect_colors")
    .add_edge("detect_colors", "enumerate")
    .add_edge("enumerate", "estimation")
    .add_edge("estimation", "measure")
    .add_edge("measure", "annotate")
    .add_edge("annotate", "pricing")
    .add_edge("pricing", END)
    .compile()
)

#PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\הריסה (1).pdf"
#PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\בנייה (1).pdf"
PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\סט תוכניות (1).pdf"

if __name__ == "__main__":
    write_logs("-----------------------------")
    result = graph.invoke({"pdf_path": PDF_PATH})
    print("Result:", result["result"])
    write_logs("Result:" + result["result"])
