import json
import re
import sys
from pathlib import Path
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
from render_annotations import render_annotations, _is_per_meter
from pdf_tools.scale import compute_scale_factor
from logs.write_logs import write_logs



class State(TypedDict):
    pdf_path: str
    pages: list                       # optional 1-indexed pages to analyze; empty/None = all
    show_measurements: bool           # whether to draw per-segment length labels on annotations
    scale_factor: float               # multiplier applied to all per-meter measurements (default 1.0)
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


def _group_blocks_by_page(segment_blocks: list) -> dict:
    """Split segment_blocks into per-page buckets keyed by page number."""
    pages: dict[int, list] = {}
    current_page = None
    for block in segment_blocks:
        if block.get("type") == "text":
            m = re.search(r"=== Segments for .+, page (\d+) ===", block.get("text", ""))
            if m:
                current_page = int(m.group(1))
        if current_page is not None:
            pages.setdefault(current_page, []).append(block)
    return pages


def _page_fingerprint(blocks: list) -> tuple:
    """Structural fingerprint for a page's segment blocks: (sorted colors, segment text count).

    Used to skip pages that are layout-identical to an already-processed page.
    """
    colors = tuple(sorted(
        m.group(1)
        for b in blocks if b.get("type") == "text"
        for m in [re.search(r"=== Segments for (#[0-9a-fA-F]{6})", b.get("text", ""))]
        if m
    ))
    segment_count = sum(
        1 for b in blocks
        if b.get("type") == "text" and not b.get("text", "").startswith("===")
    )
    return (colors, segment_count)


def _merge_classifications(merged: dict, new: dict) -> None:
    """Merge per-page classification dicts into a single accumulated dict."""
    for task_name, info in new.items():
        if task_name not in merged:
            merged[task_name] = info
        elif "groups" in merged[task_name] and "groups" in info:
            merged[task_name]["groups"].extend(info["groups"])
        elif "count" in merged[task_name] and "count" in info:
            merged[task_name]["count"] += info["count"]


def run_estimation(state: State) -> dict:
    """Run the classification agent once per page with only that page's image and segments."""
    pdf_path = state["pdf_path"]
    task_list = "\n".join(f"  - {t}" for t in get_available_tasks())
    pages_blocks = _group_blocks_by_page(state["segment_blocks"])

    all_outputs: list[str] = []
    merged_classifications: dict = {}
    seen_fingerprints: set = set()

    for page_num, blocks in sorted(pages_blocks.items()):
        fp = _page_fingerprint(blocks)
        if fp in seen_fingerprints:
            write_logs(f"estimation page {page_num}: skipped (duplicate layout of earlier page)")
            continue
        seen_fingerprints.add(fp)
        preamble = (
            f"{pdf_path}\n[render_pages: {page_num}]\n\n"
            f"AVAILABLE TASKS (use exact names in your JSON output):\n{task_list}\n\n"
            f"{state['palette']}\n\n"
            "The segment listings for all palette colors (attributes + zoomed crops) "
            "are provided below. Use the exact hex codes from the palette in your JSON output."
        )
        content = [{"type": "text", "text": preamble}] + blocks
        agent_output = estimation_agent.run_blocks(content)
        all_outputs.append(f"=== Page {page_num} ===\n{agent_output}")
        page_classifications = _extract_classifications(agent_output)
        _merge_classifications(merged_classifications, page_classifications)
        write_logs(f"estimation page {page_num}: {len(page_classifications)} task(s) found")

    combined_output = "\n\n".join(all_outputs)
    write_logs("estimation_agent_output: " + combined_output)
    return {"estimation_agent_output": combined_output, "agent_classifications": merged_classifications}


def run_measure(state: State) -> dict:
    """Reduce each task's tagged segments to a quantity.

    Per-meter tasks (name ends '(per meter)') sum segment lengths. Per-unit
    tasks cluster their tagged segments into discrete items by connectivity and
    count them. A task given only {count: N} (no taggable symbol) passes through.
    """
    pdf_path = state["pdf_path"]
    scale_factor = state.get("scale_factor", 1.0) or 1.0
    quantities: dict = {}
    for task_name, info in state["agent_classifications"].items():
        groups = info.get("groups") or ([info] if "ids" in info else None)
        if groups is not None:
            if task_name.strip().lower().endswith("(per meter)"):
                raw = measure_task_groups(pdf_path, groups)
                quantities[task_name] = round(raw * scale_factor, 2)
            else:
                quantities[task_name] = count_task_groups(pdf_path, groups)
        elif "count" in info:                       # per-unit item with no segments to tag
            quantities[task_name] = info["count"]
    write_logs("measured_quantities: " + str(quantities))
    return {"measured_quantities": quantities}


def run_verify_scale(state: State) -> dict:
    """Auto-detect PDF scale error using geometric calibration (no vision model needed).

    Skipped when the user has already provided a manual scale_factor override.
    """
    if (state.get("scale_factor") or 1.0) != 1.0:
        write_logs("scale_verify: skipped (manual override provided)")
        return {}

    per_meter = {t: i for t, i in state.get("agent_classifications", {}).items()
                 if _is_per_meter(t)}
    if not per_meter:
        return {}

    scale_factor = compute_scale_factor(state["pdf_path"])

    if scale_factor is None:
        write_logs("scale_verify: geometric calibration failed (insufficient annotations)")
        return {}

    if not (0.7 <= scale_factor <= 1.5):
        write_logs(f"scale_verify: scale_factor={scale_factor:.4f} outside plausible range, skipping")
        return {}

    if abs(scale_factor - 1.0) < 0.03:
        write_logs(f"scale_verify: scale_factor={scale_factor:.4f} negligible (<3%), skipping")
        return {}

    new_quantities = {
        task: (round(qty * scale_factor, 2) if _is_per_meter(task) else qty)
        for task, qty in state["measured_quantities"].items()
    }
    write_logs(
        f"scale_verify: scale_factor={scale_factor:.4f}, "
        f"correcting {len(per_meter)} per-meter task(s)"
    )
    return {"measured_quantities": new_quantities, "scale_factor": scale_factor}


def run_annotate(state: State) -> dict:
    """Draw the agent's task assignments onto the plan (per-page PNGs for the UI)."""
    annotations = render_annotations(
        state["pdf_path"],
        state["agent_classifications"],
        show_measurements=state.get("show_measurements", False),
        scale_factor=state.get("scale_factor", 1.0) or 1.0,
    )
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
    .add_node("verify_scale", run_verify_scale)
    .add_node("annotate", run_annotate)
    .add_node("pricing", run_pricing)
    .add_edge(START, "detect_colors")
    .add_edge("detect_colors", "enumerate")
    .add_edge("enumerate", "estimation")
    .add_edge("estimation", "measure")
    .add_edge("measure", "verify_scale")
    .add_edge("verify_scale", "annotate")
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
