"""Per-segment geometry: collection, deduplication, on-wall detection, and image crops.

This module owns the PDF-level logic for extracting colored vector segments,
flagging duplicate traces of the same physical element, checking whether a
segment runs along black architectural linework, and rendering zoomed crops.
"""
import base64
import math

import pymupdf as fitz

from pdf_tools.color_utils import _matches_color, _is_chromatic
from pdf_tools.calibration import _PLAN_WIDTH_FRACTION, _MIN_UNITS

# ---------------------------------------------------------------------------
# "on-wall" attribute: fraction of a segment's centerline that runs along the
# black architectural linework. Flip _ON_WALL_ENABLED to False to remove it.
# ---------------------------------------------------------------------------
_ON_WALL_ENABLED = False
_ON_WALL_THRESHOLD = 8.0   # PDF units; widened to tolerate offset wall-traces
_ON_WALL_SAMPLES = 20      # points sampled along the segment centerline

# ---------------------------------------------------------------------------
# Duplicate-trace detection: walls are often drawn as two parallel strokes (a
# double line), so one physical wall can surface as two near-identical segments.
# Flip _DUP_ENABLED to False to disable entirely.
# ---------------------------------------------------------------------------
_DUP_ENABLED = True
_DUP_LENGTH_TOL = 0.08   # max relative length difference within a group
_DUP_CENTER_TOL = 0.04   # max center offset as a fraction of page width/height

_COLOCATED_TOL = 0.03   # center radius (fraction of page) for the co-location count

# ---------------------------------------------------------------------------
# Per-segment image crops
# ---------------------------------------------------------------------------
_CROP_PAD = 55         # PDF units of context to include around the segment bbox
_CROP_MIN_SIDE = 130   # minimum crop box side (units) so a thin sliver still shows context
_CROP_TARGET_PX = 384  # approximate output image size in pixels
_CROP_MAX_ZOOM = 12.0  # cap zoom so tiny segments don't render huge images


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

        has_curve = any(it[0] == "c" for it in drawing["items"])
        key = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
        if key not in by_key:
            by_key[key] = {
                "rect": rect,
                "length_units": length_units,
                "filled": bool(fill_match),
                "curved": has_curve,
            }
            order.append(key)
        else:
            by_key[key]["filled"] = by_key[key]["filled"] or bool(fill_match)
            by_key[key]["curved"] = by_key[key]["curved"] or has_curve

    return [by_key[k] for k in order]


def _duplicate_canonical(segs: list[dict], page) -> list[int]:
    """Map each segment index to its group's representative (lowest) index.

    Two segments are treated as the same physical element when they share
    orientation, have lengths within _DUP_LENGTH_TOL, and centers within
    _DUP_CENTER_TOL of the page size — i.e. the two parallel strokes of a
    double-line wall. The representative of a group is its lowest index;
    every other member points back to it.
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
        elongated = long_side >= 3 * short_side
        centers.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r.width >= r.height, elongated))

    for i in range(n):
        if canonical[i] != i:
            continue
        cxi, cyi, horiz_i, elong_i = centers[i]
        if not elong_i:
            continue
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


def _colocated_counts(segs: list[dict], page) -> list[int]:
    """For each segment, how many segments (incl. itself) cluster at its center.

    A neutral fact, not a judgment: a structural line sits roughly alone,
    whereas a hatched fill or multi-stroke symbol piles many segments on one
    spot. The agent uses this to reason about the segment's nature.
    """
    n = len(segs)
    counts = [1] * n
    if n < 2:
        return counts
    w, h = page.rect.width, page.rect.height
    cx = [(s["rect"].x0 + s["rect"].x1) / 2 for s in segs]
    cy = [(s["rect"].y0 + s["rect"].y1) / 2 for s in segs]
    for i in range(n):
        c = 0
        for j in range(n):
            if abs(cx[i] - cx[j]) <= _COLOCATED_TOL * w and abs(cy[i] - cy[j]) <= _COLOCATED_TOL * h:
                c += 1
        counts[i] = c
    return counts


def _is_black(color: tuple) -> bool:
    import colorsys
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


def _crop_segment_png(page, rect) -> str:
    """Render a small zoomed PNG centered on one segment; return base64 (no prefix)."""
    cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
    side = max(max(rect.width, rect.height) + 2 * _CROP_PAD, _CROP_MIN_SIDE)
    half = side / 2
    clip = fitz.Rect(
        max(page.rect.x0, cx - half),
        max(page.rect.y0, cy - half),
        min(page.rect.x1, cx + half),
        min(page.rect.y1, cy + half),
    )
    zoom = min(_CROP_MAX_ZOOM, _CROP_TARGET_PX / max(side, 1.0))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
    return base64.b64encode(pix.tobytes("png")).decode()


def _describe_segment(seg: dict, page, cm_per_unit: float, black_lines=None) -> dict:
    rect = seg["rect"]
    w, h = rect.width, rect.height
    length_cm = round(seg["length_units"] * cm_per_unit, 1)
    ratio = (w / h) if h > 0 else float("inf")
    if ratio >= 3:
        orient = "horizontal"
    elif ratio <= 1 / 3:
        orient = "vertical"
    else:
        orient = "boxy"
    cx = round((rect.x0 + rect.x1) / 2 / page.rect.width * 100)
    cy = round((rect.y0 + rect.y1) / 2 / page.rect.height * 100)
    described = {
        "length_cm": length_cm,
        "orient": orient,
        "shape": "curved" if seg.get("curved") else "straight",
        "style": "solid-fill" if seg["filled"] else "thin-stroke",
        "cx": cx,
        "cy": cy,
    }
    if _ON_WALL_ENABLED and black_lines is not None:
        described["on_wall"] = round(_on_wall_fraction(rect, black_lines) * 100)
    return described
