"""LangChain @tool wrappers exposed to the estimation agent.

Each tool is a thin, well-documented interface over the underlying geometry
functions. Only these tools are registered with the agent — the internal helpers
they call are not exposed directly.
"""
import math
from pathlib import Path

import pymupdf as fitz
from langchain.tools import tool

from pdf_tools.color_utils import (
    _HUE_TARGETS, _parse_hex, _is_chromatic, _matches_color,
    _is_base_color, _LIST_BASE_COLORS, _namespace,
)
from pdf_tools.calibration import _calibrate, _PLAN_WIDTH_FRACTION, _MIN_UNITS
from pdf_tools.segment_geometry import (
    _collect_colored_segments, _duplicate_canonical, _colocated_counts,
    _collect_black_lines, _describe_segment, _crop_segment_png, _ON_WALL_ENABLED,
)


def _merge_nearby_rects(rects: list, proximity: float) -> list:
    """Iteratively merge rects whose centers are within `proximity` PDF units of each other."""
    clusters = list(rects)
    merged = True
    while merged:
        merged = False
        result = []
        used = [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            current = clusters[i]
            ci = ((current.x0 + current.x1) / 2, (current.y0 + current.y1) / 2)
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                other = clusters[j]
                cj = ((other.x0 + other.x1) / 2, (other.y0 + other.y1) / 2)
                dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
                if dist <= proximity:
                    current = current | other
                    ci = ((current.x0 + current.x1) / 2, (current.y0 + current.y1) / 2)
                    used[j] = True
                    merged = True
            result.append(current)
        clusters = result
    return clusters


@tool
def count_outline_shapes_by_color(pdf_path: str, color: str, page_number: int = 1) -> str:
    """Count distinct unfilled (outline-only) colored shapes in a PDF floor plan.

    Detects items drawn as a colored stroke with no fill — e.g. door arcs,
    window symbols, fixtures — that get_wall_lengths_by_color misses because
    it only reads filled shapes.

    Uses the same auto-calibration and color-matching logic. Nearby path
    segments that belong to the same symbol (door panel + swing arc) are
    grouped so each physical item is counted once. Sizes are reported in cm
    so the agent can sanity-check they look like real doors/windows.

    Args:
        pdf_path: Path to the PDF construction plan.
        color: Stroke color to detect. Supported: red, orange, yellow, green,
               blue, cyan, magenta, purple, black, white, gray.
        page_number: 1-indexed page number to read (default 1 = first page).

    Returns:
        Count of distinct shapes and their estimated sizes in cm.
    """
    supported = list(_HUE_TARGETS) + ["black", "white", "gray"]
    if color.lower() not in supported and _parse_hex(color) is None:
        return (
            f"Unknown color '{color}'. Pass a hex code like '#e6f00a', or one of: "
            f"{', '.join(supported)}"
        )

    path_obj = Path(pdf_path.strip("'\""))
    if not path_obj.exists():
        return f"File not found: {pdf_path}"

    doc = fitz.open(str(path_obj))
    page_idx = page_number - 1
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        return f"Page {page_number} out of range — PDF has {len(doc)} page(s)."
    page = doc[page_idx]

    try:
        cm_per_unit = _calibrate(page)
    except ValueError as e:
        doc.close()
        return f"Calibration failed: {e}"

    max_x = page.rect.width * _PLAN_WIDTH_FRACTION

    rects = []
    for drawing in page.get_drawings():
        stroke = drawing.get("color")
        fill = drawing.get("fill")

        if not stroke or len(stroke) < 3:
            continue
        if not _matches_color(stroke, color):
            continue
        if fill and len(fill) >= 3 and _is_chromatic(fill):
            continue

        rect = drawing["rect"]
        center_x = (rect.x0 + rect.x1) / 2
        if center_x >= max_x:
            continue
        w, h = rect.width, rect.height
        if max(w, h) < _MIN_UNITS:
            continue
        ratio = w / h if h > 0 else float("inf")
        if not (0.25 <= ratio <= 4.0):
            continue

        rects.append(rect)

    doc.close()

    if not rects:
        return f"No unfilled {color} outline shapes found in the floor plan area."

    proximity = 10 / cm_per_unit
    clusters = _merge_nearby_rects(rects, proximity)
    count = len(clusters)
    sizes_cm = sorted(round(max(c.width, c.height) * cm_per_unit) for c in clusters)

    hex_rgb = _parse_hex(color)
    is_black = color.lower() == "black" or (hex_rgb is not None and max(hex_rgb) < 0.2)
    warning = ""
    if is_black:
        warning = (
            "WARNING: black is usually the plan's base drawing color (walls, text, "
            "dimensions, grid) -- a count of black outlines is almost never a real item "
            "count. Only trust this if black is genuinely a dedicated task symbol on THIS "
            "plan; otherwise do NOT count it.\n"
        )
    return (
        f"{warning}"
        f"Found {count} distinct {color} outline shape(s) in the floor plan.\n"
        f"Estimated sizes (largest dimension): {sizes_cm} cm"
    )


@tool
def get_wall_lengths_by_color(
    pdf_path: str, color: str, page_number: int = 1, drawing_type: str = "fill"
) -> str:
    """Measure the total real-world length of all colored segments of a given type in a PDF plan.

    Reads directly from the PDF vector geometry and auto-calibrates the unit conversion
    from the plan's own dimension annotations — works for any scale or paper size.
    Color matching uses HSV hue, so all shades of a color (e.g. dark red, bright red,
    salmon) are treated as the same color.

    Before calling this tool, read the relevant PDF page to determine how the elements
    are drawn: solid filled shapes (drawing_type="fill") or colored outlines/strokes
    (drawing_type="stroke"). Passing the wrong type returns 0 results.

    Args:
        pdf_path: Path to the PDF construction plan.
        color: Color to measure. Supported: red, orange, yellow, green, blue,
               cyan, magenta, purple, black, white, gray.
        page_number: 1-indexed page number to read (default 1 = first page).
        drawing_type: how to match shapes by color —
                      "fill"   — only shapes whose interior is the target color
                                 (solid colored bars, most demolition plans).
                      "stroke" — only shapes whose outline is the target color
                                 and have no solid fill (pure outline elements).
                      "any"    — shapes where either fill or stroke matches
                                 (use when both styles appear, or when unsure).
                      Read the PDF visually to determine which applies. If a
                      call returns 0 results, try a different drawing_type.

    Returns:
        Individual segment lengths in cm and total length in meters.
    """
    if drawing_type not in ("fill", "stroke", "any"):
        return f"Invalid drawing_type '{drawing_type}'. Use 'fill', 'stroke', or 'any'."

    supported = list(_HUE_TARGETS) + ["black", "white", "gray"]
    if color.lower() not in supported and _parse_hex(color) is None:
        return (
            f"Unknown color '{color}'. Pass a hex code like '#e6f00a', or one of: "
            f"{', '.join(supported)}"
        )

    path = Path(pdf_path.strip("'\""))
    if not path.exists():
        return f"File not found: {pdf_path}"

    doc = fitz.open(str(path))
    page_idx = page_number - 1
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        return f"Page {page_number} out of range — PDF has {len(doc)} page(s)."
    page = doc[page_idx]

    try:
        cm_per_unit = _calibrate(page)
    except ValueError as e:
        doc.close()
        return f"Calibration failed: {e}"

    max_x = page.rect.width * _PLAN_WIDTH_FRACTION
    segments: list[float] = []

    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        stroke = drawing.get("color")

        if drawing_type == "fill":
            if not fill or len(fill) < 3 or not _matches_color(fill, color):
                continue
        elif drawing_type == "stroke":
            if not stroke or len(stroke) < 3 or not _matches_color(stroke, color):
                continue
            if fill and len(fill) >= 3 and _is_chromatic(fill):
                continue
        else:  # "any"
            fill_match = fill and len(fill) >= 3 and _matches_color(fill, color)
            stroke_match = stroke and len(stroke) >= 3 and _matches_color(stroke, color)
            if not fill_match and not stroke_match:
                continue

        rect = drawing["rect"]
        if rect.x0 >= max_x:
            continue
        length_units = max(rect.width, rect.height)
        if length_units < _MIN_UNITS:
            continue
        segments.append(round(length_units * cm_per_unit, 1))

    doc.close()

    if not segments:
        return (
            f"No {color} {drawing_type}-colored segments found in the floor plan area. "
            f"If you expected results, try the other drawing_type."
        )

    total_cm = sum(segments)
    total_m = total_cm / 100
    return (
        f"{color} {drawing_type} segments ({len(segments)} found): {sorted(segments)} cm\n"
        f"Total: {total_m:.2f} m ({total_cm:.0f} cm)"
    )


@tool
def measure_total_length_by_coordinates(
    pdf_path: str,
    segments: list[list[float]],
    page_number: int = 1,
) -> str:
    """Measure the total real-world length of line segments defined by their coordinates.

    Each segment is [x1, y1, x2, y2] where all values are fractions (0.0–1.0) of the
    page dimensions (0.0 = left/top edge, 1.0 = right/bottom edge). The scale is read
    automatically from the PDF title block (e.g. "1:100").

    Pass ALL segments for a task in a single call — the tool sums them and returns the
    total in meters plus a per-segment breakdown.

    Use this tool when elements cannot be reliably identified by color alone — e.g.
    dashed lines, composite symbols, partially-measured walls, or any case where you
    need to reason visually about exactly which parts to measure.

    Args:
        pdf_path: path to the PDF file
        segments: list of [x1, y1, x2, y2] coordinate pairs (normalized 0.0–1.0 fractions)
        page_number: 1-indexed page number (default 1)

    Returns:
        Per-segment lengths in cm and grand total in meters.
    """
    path_obj = Path(pdf_path.strip("'\""))
    if not path_obj.exists():
        return f"File not found: {pdf_path}"

    doc = fitz.open(str(path_obj))
    page_idx = page_number - 1
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        return f"Page {page_number} out of range — PDF has {len(doc)} page(s)."
    page = doc[page_idx]

    try:
        cm_per_unit = _calibrate(page)
    except ValueError as e:
        doc.close()
        return f"Calibration failed: {e}"

    w, h = page.rect.width, page.rect.height
    doc.close()

    results: list[float] = []
    for seg in segments:
        if len(seg) != 4:
            return f"Each segment must have exactly 4 values [x1, y1, x2, y2]. Got: {seg}"
        x1, y1, x2, y2 = seg
        dx = (x2 - x1) * w
        dy = (y2 - y1) * h
        length_units = math.sqrt(dx * dx + dy * dy)
        results.append(round(length_units * cm_per_unit, 1))

    total_cm = sum(results)
    lines = [f"  Segment {i + 1}: {r} cm" for i, r in enumerate(results)]
    lines.append(f"Total: {round(total_cm / 100, 2)} m ({round(total_cm, 1)} cm)")
    return "\n".join(lines)


@tool
def list_colored_segments(pdf_path: str, color: str, page_number: int = 1) -> list:
    """List every colored vector segment of a given color in a PDF floor plan, each with an ID.

    Reads the exact geometry from the PDF (no visual guessing). For each segment it
    reports NEUTRAL FACTS — it does not guess what the segment is. Each line gives:
    an ID, real length in cm, orientation (horizontal/vertical/boxy), whether it is
    a solid-fill or a thin-stroke, whether its path is straight or curved, its center
    as a percentage of the page, and 'clusterxN' when N segments pile on one center.
    Each segment's text line is followed by a zoomed image crop of that spot on the
    plan (with surrounding context) so you can see what it is and read nearby labels.

    Use these facts plus the plan image to decide WHICH segments belong to a task,
    then pass the relevant IDs to 'measure_segments_by_id' to get their total length.

    Each ID is namespaced by color and page (e.g. "R3-5" = red, page 3, segment 5),
    so IDs from different colors or pages can never be mixed up. Always pass IDs
    exactly as shown — including the prefix — to 'measure_segments_by_id'.

    Args:
        pdf_path: Path to the PDF construction plan.
        color: Color to list. Either a hex code like '#e6f00a' (recommended —
               use the exact hex from the detected color palette), or a named
               color: red, orange, yellow, green, blue, cyan, magenta, purple,
               black, white, gray.
        page_number: 1-indexed page number to read (default 1 = first page).

    Returns:
        A list of content blocks: a header, then per segment a text line of its
        attributes followed by an image crop, then closing notes. (On an error,
        a single string message is returned instead.)
    """
    supported = list(_HUE_TARGETS) + ["black", "white", "gray"]
    if color.lower() not in supported and _parse_hex(color) is None:
        return (
            f"Unknown color '{color}'. Pass a hex code like '#e6f00a', or one of: "
            f"{', '.join(supported)}"
        )

    if _is_base_color(color) and not _LIST_BASE_COLORS:
        return (
            f"'{color}' is achromatic (black/gray/white) — the plan's base drawing of existing "
            "walls, text, dimensions and grid, not a task layer. Pick a chromatic color from "
            "the detected palette instead."
        )

    path_obj = Path(pdf_path.strip("'\""))
    if not path_obj.exists():
        return f"File not found: {pdf_path}"

    doc = fitz.open(str(path_obj))
    page_idx = page_number - 1
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        return f"Page {page_number} out of range — PDF has {len(doc)} page(s)."
    page = doc[page_idx]

    try:
        cm_per_unit = _calibrate(page)
    except ValueError as e:
        doc.close()
        return f"Calibration failed: {e}"

    segs = _collect_colored_segments(page, color)
    black_lines = _collect_black_lines(page) if _ON_WALL_ENABLED else None
    described = [_describe_segment(s, page, cm_per_unit, black_lines) for s in segs]
    canonical = _duplicate_canonical(segs, page)
    colocated = _colocated_counts(segs, page)
    crops = [_crop_segment_png(page, s["rect"]) for s in segs]
    doc.close()

    if not described:
        return f"No {color} segments found in the floor plan area."

    ns = _namespace(color, page_number)
    blocks: list[dict] = [{
        "type": "text",
        "text": (
            f"Found {len(described)} {color} segment(s) on page {page_number} (IDs prefixed '{ns}-').\n"
            "For each segment below you get neutral geometry AND a zoomed image crop of that "
            "spot on the plan (with surrounding context). Decide for yourself what each one is "
            "-- read any text label visible in the crop; it is the strongest signal.\n"
            "Attributes: length | orientation | solid-fill or thin-stroke | straight or curved "
            "| center (x%,y%) | clusterxN (segments sharing this center)."
        ),
    }]
    has_dups = False
    has_cluster = False
    for i, d in enumerate(described):
        on_wall = f" | on-wall {d['on_wall']}%" if "on_wall" in d else ""
        if canonical[i] != i:
            dup = f" | dup-of {ns}-{canonical[i]}"
            has_dups = True
        else:
            dup = ""
        if colocated[i] > 3:
            cluster = f" | clusterx{colocated[i]}"
            has_cluster = True
        else:
            cluster = ""
        blocks.append({
            "type": "text",
            "text": (
                f"ID {ns}-{i} | {d['length_cm']} cm | {d['orient']} | {d['style']} | {d['shape']}"
                f" | center ({d['cx']}%,{d['cy']}%){cluster}{on_wall}{dup}"
            ),
        })
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{crops[i]}", "detail": "low"},
        })

    notes = []
    if has_cluster:
        notes.append(
            "Note: 'clusterxN' means N segments pile on nearly the same center. A structural "
            "line stands roughly alone; a large cluster of short/curved strokes is typically a "
            "filled symbol or hatch (e.g. a fixture or arc), not building length -- confirm by "
            "reading the label in its crop."
        )
    if has_dups:
        notes.append(
            "Note: segments marked 'dup-of <id>' are a second trace of the same physical element "
            "(e.g. the other edge of a double-line wall). Assign only the referenced <id> to a "
            "task and ignore the duplicate, or you will double-count its length."
        )
    notes.append(
        f"Select the IDs that belong to your task and pass them (with the '{ns}-' prefix) "
        "to measure_segments_by_id."
    )
    blocks.append({"type": "text", "text": "\n".join(notes)})
    return blocks


@tool(name_or_callable="measure_segments_by_id")
def measure_segments_by_id_tool(
    pdf_path: str, color: str, ids: list[str], page_number: int = 1
) -> str:
    """Sum the real-world length of the colored segments with the given IDs.

    Pass the namespaced IDs exactly as reported by 'list_colored_segments'
    (e.g. "R3-5"). The color and page_number must match the prefix of those IDs —
    any ID belonging to a different color/page is rejected, so segments from
    separate listings can never be summed together by mistake.

    Args:
        pdf_path: Path to the PDF construction plan.
        color: Same color string passed to list_colored_segments.
        ids: List of namespaced segment IDs (e.g. ["R3-5", "R3-6"]).
        page_number: Same page_number passed to list_colored_segments (default 1).

    Returns:
        Per-segment lengths in cm and the grand total in meters.
    """
    return measure_segments_by_id(pdf_path, color, ids, page_number)


def measure_segments_by_id(
    pdf_path: str, color: str, ids: list[str], page_number: int = 1
) -> str:
    path_obj = Path(pdf_path.strip("'\""))
    if not path_obj.exists():
        return f"File not found: {pdf_path}"

    if not ids:
        return "No IDs provided. Call list_colored_segments first, then pass the relevant IDs."

    expected_ns = _namespace(color, page_number)

    indices: list[int] = []
    for token in ids:
        prefix, sep, idx_str = str(token).rpartition("-")
        if not sep or not idx_str.isdigit():
            return (
                f"Malformed ID '{token}'. Use the exact IDs from list_colored_segments, "
                f"e.g. '{expected_ns}-0'."
            )
        if prefix != expected_ns:
            return (
                f"ID '{token}' belongs to a different listing (namespace '{prefix}'), "
                f"but this call is for {color} page {page_number} (namespace '{expected_ns}'). "
                f"Only pass IDs prefixed '{expected_ns}-'. Re-run list_colored_segments "
                f"for the segments you actually want."
            )
        indices.append(int(idx_str))

    doc = fitz.open(str(path_obj))
    page_idx = page_number - 1
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        return f"Page {page_number} out of range — PDF has {len(doc)} page(s)."
    page = doc[page_idx]

    try:
        cm_per_unit = _calibrate(page)
    except ValueError as e:
        doc.close()
        return f"Calibration failed: {e}"

    segs = _collect_colored_segments(page, color)
    canonical = _duplicate_canonical(segs, page)
    doc.close()

    for token, i in zip(ids, indices):
        if i < 0 or i >= len(segs):
            return (
                f"Invalid ID '{token}': only {len(segs)} {color} segment(s) exist on "
                f"page {page_number} (valid IDs {expected_ns}-0 to {expected_ns}-{len(segs) - 1}). "
                f"Re-run list_colored_segments."
            )

    lines = []
    total_cm = 0.0
    seen_reps: dict[int, str] = {}
    for token, i in zip(ids, indices):
        length_cm = round(segs[i]["length_units"] * cm_per_unit, 1)
        rep = canonical[i]
        if rep in seen_reps:
            lines.append(
                f"  ID {token}: {length_cm} cm  -- SKIPPED (duplicate trace of "
                f"{seen_reps[rep]}; counted once to avoid double-counting)"
            )
            continue
        seen_reps[rep] = token
        total_cm += length_cm
        lines.append(f"  ID {token}: {length_cm} cm")

    lines.append(f"Total: {round(total_cm / 100, 2)} m ({round(total_cm, 1)} cm)")
    if len(seen_reps) < len(indices):
        lines.append(
            "Note: some selected IDs were duplicate traces of the same physical "
            "element and were counted only once."
        )
    return "\n".join(lines)
