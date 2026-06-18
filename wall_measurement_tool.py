from pathlib import Path
import colorsys
import math
import re
import statistics
from langchain.tools import tool
import pymupdf as fitz

# Hue targets in degrees [0, 360). All shades sharing a hue match the same name.
_HUE_TARGETS: dict[str, float] = {
    "red":     0,
    "orange":  30,
    "yellow":  60,
    "green":   120,
    "cyan":    180,
    "blue":    240,
    "magenta": 300,
    "purple":  300,
}
_HUE_TOLERANCE = 30    # degrees — covers dark/light/desaturated shades of the same hue
_MIN_SATURATION = 0.25  # below this, no dominant hue → treat as achromatic

_MIN_UNITS = 10
_PLAN_WIDTH_FRACTION = 0.70  # title blocks occupy the rightmost ~25-30% of the page

# ---------------------------------------------------------------------------
# "on-wall" attribute: fraction of a segment's centerline that runs along the
# black architectural linework. High = sits on a building wall (likely a real
# wall element); low = floats in open space (likely a dimension/leader line).
# It is a noisy hint, not a classifier — flip _ON_WALL_ENABLED to False to
# remove it entirely (no extra cost, no extra column) if it causes problems.
# ---------------------------------------------------------------------------
_ON_WALL_ENABLED = False
_ON_WALL_THRESHOLD = 8.0   # PDF units; widened to tolerate offset wall-traces
_ON_WALL_SAMPLES = 20      # points sampled along the segment centerline

# ---------------------------------------------------------------------------
# Duplicate-trace detection: walls are often drawn as two parallel strokes (a
# double line) or a closed rectangle, so one physical wall can surface as two
# near-identical segments. Counting both doubles its length. We group such
# near-duplicates deterministically: list_colored_segments flags the extras as
# "dup-of <id>", and measure_segments_by_id counts each group only once as a
# safety net. Flip _DUP_ENABLED to False to disable entirely.
# ---------------------------------------------------------------------------
_DUP_ENABLED = True
_DUP_LENGTH_TOL = 0.08   # max relative length difference within a group
_DUP_CENTER_TOL = 0.04   # max center offset as a fraction of page width/height

# Short, collision-free color codes used to namespace segment IDs (e.g. "R3-5"
# = red, page 3, index 5). "blue"/"black" and "gray"/"grey" must not collide.
_COLOR_CODES: dict[str, str] = {
    "red": "R", "orange": "O", "yellow": "Y", "green": "G",
    "cyan": "C", "blue": "B", "magenta": "M", "purple": "P",
    "black": "K", "white": "W", "gray": "GR", "grey": "GR",
}


def _namespace(color: str, page_number: int) -> str:
    """Build the ID namespace for a (color, page) listing, e.g. ('red', 3) -> 'R3'."""
    code = _COLOR_CODES.get(color.lower(), color.lower())
    return f"{code}{page_number}"


def _matches_color(fill: tuple, color_name: str) -> bool:
    r, g, b = fill[0], fill[1], fill[2]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = h * 360
    color_name = color_name.lower()

    if color_name in _HUE_TARGETS:
        if s < _MIN_SATURATION:
            return False
        target = _HUE_TARGETS[color_name]
        diff = min(abs(h_deg - target), 360 - abs(h_deg - target))
        return diff <= _HUE_TOLERANCE

    if color_name == "black":
        return v < 0.2
    if color_name == "white":
        return v > 0.8 and s < 0.1
    if color_name in ("gray", "grey"):
        return 0.2 <= v <= 0.8 and s < _MIN_SATURATION

    return False


def _is_chromatic(fill: tuple) -> bool:
    _, s, v = colorsys.rgb_to_hsv(fill[0], fill[1], fill[2])
    return s >= _MIN_SATURATION and v >= 0.1


def _calibrate(page) -> float:
    """Derive cm_per_pdf_unit for this page.

    Strategy 1 — scale ratio in title block text (e.g. "1:50").
    Many CAD exports convert dimension numbers to paths, making them
    invisible to text extraction, but the title block scale tag remains
    as real text.  1 PDF point = 1/72 inch = 2.54/72 cm; at scale 1:N
    that point represents N × 2.54/72 real-world cm.

    Strategy 2 — match colored paths to nearby numeric dimension
    annotations (works for PDFs that keep dimensions as text).
    """
    # Strategy 1: parse "1:N" from anywhere on the page
    page_text = page.get_text("text")
    m = re.search(r"\b1:(\d+)\b", page_text)
    if m:
        return int(m.group(1)) * (2.54 / 72)

    # Strategy 2: annotation-matching fallback
    max_x = page.rect.width * _PLAN_WIDTH_FRACTION
    max_dist = page.rect.width * 0.05

    annotations: list[tuple[float, float, float]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if re.fullmatch(r"\d+(\.\d+)?", text):
                    val = float(text)
                    if 5 <= val <= 500:
                        bbox = span["bbox"]
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        if cx < max_x:
                            annotations.append((val, cx, cy))

    if not annotations:
        raise ValueError(
            "Calibration failed: no '1:N' scale tag found in the title block and "
            "no numeric dimension annotations found in the plan area."
        )

    colored_paths: list[tuple[float, float, float]] = []
    for d in page.get_drawings():
        fill = d.get("fill")
        if not fill or len(fill) < 3:
            continue
        if not _is_chromatic(fill):
            continue
        r = d["rect"]
        if r.x0 >= max_x:
            continue
        long_side = max(r.width, r.height)
        if long_side < _MIN_UNITS:
            continue
        colored_paths.append((long_side, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))

    if not colored_paths:
        raise ValueError("No colored wall segments found in the floor plan area.")

    ratios: list[float] = []
    for path_len, px, py in colored_paths:
        nearest = min(annotations, key=lambda a: (a[1] - px) ** 2 + (a[2] - py) ** 2)
        dist = ((nearest[1] - px) ** 2 + (nearest[2] - py) ** 2) ** 0.5
        if dist > max_dist:
            continue
        ratios.append(nearest[0] / path_len)

    if not ratios:
        raise ValueError("Could not match any colored segment to a nearby dimension annotation.")

    return statistics.median(ratios)


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
    if color.lower() not in supported:
        return f"Unknown color '{color}'. Supported: {', '.join(supported)}"

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
        # Skip shapes that are clearly filled with a chromatic color —
        # those are walls already handled by get_wall_lengths_by_color.
        if fill and len(fill) >= 3 and _is_chromatic(fill):
            continue

        rect = drawing["rect"]
        # Use the shape's center to decide if it's in the plan area, not just
        # its left edge — annotation boxes that straddle the boundary would
        # pass an x0-only check but their center is in the title block.
        center_x = (rect.x0 + rect.x1) / 2
        if center_x >= max_x:
            continue
        w, h = rect.width, rect.height
        if max(w, h) < _MIN_UNITS:
            continue
        # Lenient aspect-ratio guard: keeps swing doors (~1:1), double/sliding
        # doors (~2–3:1), and rectangular windows (~1:3), while reliably
        # excluding door panel lines (~13:1) and jamb edge lines (~1:15).
        # Wide annotation-box outlines (~6:1) are also excluded.
        ratio = w / h if h > 0 else float("inf")
        if not (0.25 <= ratio <= 4.0):
            continue

        rects.append(rect)

    doc.close()

    if not rects:
        return f"No unfilled {color} outline shapes found in the floor plan area."

    # Cluster path segments within 10 cm of each other into one symbol.
    proximity = 10 / cm_per_unit
    clusters = _merge_nearby_rects(rects, proximity)
    count = len(clusters)
    sizes_cm = sorted(round(max(c.width, c.height) * cm_per_unit) for c in clusters)

    return (
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
    if color.lower() not in supported:
        return f"Unknown color '{color}'. Supported: {', '.join(supported)}"

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
                continue  # has a solid chromatic fill — use drawing_type="fill" for these
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


def _collect_colored_segments(page, color: str) -> list[dict]:
    """Deterministic, ordered list of colored vector segments in the plan area.

    Both list_colored_segments and measure_segments_by_id call this with the
    same arguments, so a segment's position in the returned list (its ID) is
    stable across the two calls. Identical/duplicate paths are merged, and a
    segment is flagged 'filled' if any path at that location has a chromatic
    fill of the target color (otherwise it is an outline/stroke).
    """
    max_x = page.rect.width * _PLAN_WIDTH_FRACTION
    by_key: dict[tuple, dict] = {}
    order: list[tuple] = []

    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        stroke = drawing.get("color")
        fill_match = fill and len(fill) >= 3 and _matches_color(fill, color)
        stroke_match = stroke and len(stroke) >= 3 and _matches_color(stroke, color)
        if not (fill_match or stroke_match):
            continue

        rect = drawing["rect"]
        if (rect.x0 + rect.x1) / 2 >= max_x:
            continue
        length_units = max(rect.width, rect.height)
        if length_units < _MIN_UNITS:
            continue

        key = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
        if key not in by_key:
            by_key[key] = {"rect": rect, "length_units": length_units, "filled": bool(fill_match)}
            order.append(key)
        else:
            by_key[key]["filled"] = by_key[key]["filled"] or bool(fill_match)

    return [by_key[k] for k in order]


def _duplicate_canonical(segs: list[dict], page) -> list[int]:
    """Map each segment index to its group's representative (lowest) index.

    Two segments are treated as the same physical element when they share
    orientation, have lengths within _DUP_LENGTH_TOL, and centers within
    _DUP_CENTER_TOL of the page size — i.e. the two parallel strokes of a
    double-line wall. The representative of a group is its lowest index;
    every other member points back to it. With detection off (or fewer than
    two segments) each index maps to itself, so callers behave as before.
    """
    n = len(segs)
    canonical = list(range(n))
    if not _DUP_ENABLED or n < 2:
        return canonical

    w, h = page.rect.width, page.rect.height
    centers = []
    for s in segs:
        r = s["rect"]
        long_side, short_side = max(r.width, r.height), min(r.width, r.height)
        # Elongated = much longer than thick (a wall edge). Multiplicative form
        # avoids dividing by a zero short side: a pure axis-aligned line has a
        # zero-width bbox, which must still count as elongated. Square arcs /
        # symbols (long ≈ short) fail this and are excluded from dedup.
        elongated = long_side >= 3 * short_side
        centers.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r.width >= r.height, elongated))

    for i in range(n):
        if canonical[i] != i:
            continue
        cxi, cyi, horiz_i, elong_i = centers[i]
        if not elong_i:
            continue  # only elongated runs (wall edges) can be double-line duplicates;
            #            square arcs / symbols are never a duplicate trace of a wall
        li = segs[i]["length_units"]
        for j in range(i + 1, n):
            if canonical[j] != j:
                continue
            cxj, cyj, horiz_j, elong_j = centers[j]
            if not elong_j or horiz_j != horiz_i:
                continue
            lj = segs[j]["length_units"]
            if abs(li - lj) / max(li, lj) > _DUP_LENGTH_TOL:
                continue
            if abs(cxi - cxj) > _DUP_CENTER_TOL * w:
                continue
            if abs(cyi - cyj) > _DUP_CENTER_TOL * h:
                continue
            canonical[j] = i
    return canonical


def _is_black(color: tuple) -> bool:
    return bool(color) and len(color) >= 3 and colorsys.rgb_to_hsv(*color[:3])[2] < 0.3


def _collect_black_lines(page) -> list[tuple]:
    """Extract the black architectural linework as (point, point) segments."""
    lines: list[tuple] = []
    for drawing in page.get_drawings():
        if not _is_black(drawing.get("color")):
            continue
        for item in drawing["items"]:
            if item[0] == "l":
                lines.append((item[1], item[2]))
            elif item[0] == "re":
                r = item[1]
                corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                for i in range(4):
                    a, b = corners[i], corners[(i + 1) % 4]
                    lines.append((fitz.Point(a), fitz.Point(b)))
    return lines


def _point_segment_distance(px: float, py: float, a, b) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(px - a.x, py - a.y)
    t = max(0.0, min(1.0, ((px - a.x) * dx + (py - a.y) * dy) / len_sq))
    return math.hypot(px - (a.x + t * dx), py - (a.y + t * dy))


def _on_wall_fraction(rect, black_lines: list[tuple]) -> float:
    """Fraction of the segment centerline running within threshold of black linework."""
    if not black_lines:
        return 0.0
    n = _ON_WALL_SAMPLES
    if rect.width >= rect.height:
        cy = (rect.y0 + rect.y1) / 2
        points = [(rect.x0 + (rect.x1 - rect.x0) * k / (n - 1), cy) for k in range(n)]
    else:
        cx = (rect.x0 + rect.x1) / 2
        points = [(cx, rect.y0 + (rect.y1 - rect.y0) * k / (n - 1)) for k in range(n)]
    hits = sum(
        1
        for px, py in points
        if any(_point_segment_distance(px, py, a, b) <= _ON_WALL_THRESHOLD for a, b in black_lines)
    )
    return hits / len(points)


def _describe_segment(seg: dict, page, cm_per_unit: float, black_lines=None) -> dict:
    rect = seg["rect"]
    w, h = rect.width, rect.height
    length_cm = round(seg["length_units"] * cm_per_unit, 1)
    ratio = (w / h) if h > 0 else float("inf")
    if ratio >= 3:
        orient = "horizontal"
    elif ratio <= 1 / 3:
        orient = "vertical"
    elif 0.5 <= ratio <= 2:
        orient = "square (arc/symbol)"
    else:
        orient = "rectangular"
    cx = round((rect.x0 + rect.x1) / 2 / page.rect.width * 100)
    cy = round((rect.y0 + rect.y1) / 2 / page.rect.height * 100)
    described = {
        "length_cm": length_cm,
        "orient": orient,
        "style": "FILLED" if seg["filled"] else "OUTLINE",
        "cx": cx,
        "cy": cy,
    }
    if _ON_WALL_ENABLED and black_lines is not None:
        described["on_wall"] = round(_on_wall_fraction(rect, black_lines) * 100)
    return described


@tool
def list_colored_segments(pdf_path: str, color: str, page_number: int = 1) -> str:
    """List every colored vector segment of a given color in a PDF floor plan, each with an ID.

    Reads the exact geometry from the PDF (no visual guessing). For each segment it
    reports: an ID, its real length in cm, orientation, whether it is FILLED (solid)
    or OUTLINE (stroke only), and its center position as a percentage of the page.

    Use this to decide WHICH segments belong to a measurement task, then pass the
    relevant IDs to 'measure_segments_by_id' to get their total length.

    Each ID is namespaced by color and page (e.g. "R3-5" = red, page 3, segment 5),
    so IDs from different colors or pages can never be mixed up. Always pass IDs
    exactly as shown — including the prefix — to 'measure_segments_by_id'.

    Guidance: thin FILLED elongated segments are usually walls; square/arc OUTLINE
    shapes are usually doors or window symbols; long thin OUTLINE segments far from
    walls are often dimension/leader lines (not real building elements).

    When available, each segment also reports 'on-wall N%' — how much of it runs
    along the building's architectural linework. High (~80-100%) means it sits on a
    real wall; low (~0-20%) on a long segment usually means a dimension/leader line.
    Treat it as a hint, not a rule — colored wall-traces can be drawn slightly
    offset and score in the middle.

    Args:
        pdf_path: Path to the PDF construction plan.
        color: Color to list. Supported: red, orange, yellow, green, blue, cyan,
               magenta, purple, black, white, gray.
        page_number: 1-indexed page number to read (default 1 = first page).

    Returns:
        A numbered list of segments with length, orientation, style, and position.
    """
    supported = list(_HUE_TARGETS) + ["black", "white", "gray"]
    if color.lower() not in supported:
        return f"Unknown color '{color}'. Supported: {', '.join(supported)}"

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
    doc.close()

    if not described:
        return f"No {color} segments found in the floor plan area."

    ns = _namespace(color, page_number)
    lines = [f"Found {len(described)} {color} segment(s) on page {page_number} (IDs prefixed '{ns}-'):"]
    has_dups = False
    for i, d in enumerate(described):
        on_wall = f" | on-wall {d['on_wall']}%" if "on_wall" in d else ""
        if canonical[i] != i:
            dup = f" | dup-of {ns}-{canonical[i]}"
            has_dups = True
        else:
            dup = ""
        lines.append(
            f"ID {ns}-{i} | {d['length_cm']} cm | {d['orient']} | {d['style']} "
            f"| center ({d['cx']}%,{d['cy']}%){on_wall}{dup}"
        )
    if has_dups:
        lines.append(
            "\nNote: segments marked 'dup-of <id>' are a second trace of the same "
            "physical element (e.g. the other edge of a double-line wall). Assign only "
            "the referenced <id> to a task and ignore the duplicate, or you will "
            "double-count its length."
        )
    lines.append(
        f"\nSelect the IDs that belong to your task and pass them (with the '{ns}-' "
        "prefix) to measure_segments_by_id."
    )
    return "\n".join(lines)


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

    # Parse and validate every token before opening the PDF, so a mismatch is
    # reported clearly rather than silently measuring the wrong segments.
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

    # Range-check every requested index before summing, so a bad ID is reported
    # clearly instead of silently skewing the total.
    for token, i in zip(ids, indices):
        if i < 0 or i >= len(segs):
            return (
                f"Invalid ID '{token}': only {len(segs)} {color} segment(s) exist on "
                f"page {page_number} (valid IDs {expected_ns}-0 to {expected_ns}-{len(segs) - 1}). "
                f"Re-run list_colored_segments."
            )

    lines = []
    total_cm = 0.0
    seen_reps: dict[int, str] = {}   # group representative -> first ID that counted it
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
