"""Render the agent's classification visually on the plan (no new detection).

Takes the per-task segment assignments the agent produced (task -> groups of
{color, page, ids}) and draws a highlighted box around every classified segment,
one overlay color per task, with a legend. Returns per-page PNGs (base64) for the
UI. Pure rendering of data we already have — the IDs resolve to the exact same
rectangles the measurement step uses.
"""
import base64

import pymupdf as fitz

from wall_measurement_tool import (
    _collect_colored_segments,
    _duplicate_canonical,
    _namespace,
    _calibrate,
    cluster_group_item_rects,
)

_PATH_WIDTH = 3.0   # highlight width for per-meter path traces (thicker than the original line)


def _is_per_meter(task_name: str) -> bool:
    """Per-meter tasks are drawn per-segment; per-unit tasks per merged item."""
    return task_name.strip().lower().endswith("(per meter)")

# Distinct, high-contrast overlay colors assigned to tasks in order (RGB 0-1).
_OVERLAY_COLORS = [
    (0.90, 0.10, 0.10),  # red
    (0.10, 0.35, 0.95),  # blue
    (0.05, 0.65, 0.20),  # green
    (0.95, 0.55, 0.00),  # orange
    (0.60, 0.15, 0.75),  # purple
    (0.00, 0.65, 0.65),  # teal
    (0.85, 0.15, 0.55),  # magenta
    (0.55, 0.35, 0.10),  # brown
]
_BOX_PAD = 6        # PDF units of padding around each segment's bbox
_BOX_WIDTH = 1.6    # outline width
_RENDER_ZOOM = 2.0  # output resolution


def _groups_of(info: dict) -> list[dict]:
    """Normalize a task's classification value to a list of {color, page, ids}."""
    if "groups" in info:
        return info["groups"]
    if "ids" in info:
        return [info]
    return []  # per-unit (count) tasks have no segments to draw


def _paths_for_group(page, group: dict, scale_factor: float = 1.0) -> list[tuple]:
    """Resolve a group's IDs to (points, length_m, cx, cy) tuples (dups collapsed).

    points: ordered (x, y) vertices tracing the segment's drawn path.
    length_m: real-world length in meters (None if calibration unavailable).
    cx, cy: center of the segment's bounding box (label anchor point).
    """
    color = group["color"]
    ns = _namespace(color, group.get("page", 1))
    segs = _collect_colored_segments(page, color)
    canonical = _duplicate_canonical(segs, page)
    try:
        cm_per_unit = _calibrate(page)
    except ValueError:
        cm_per_unit = None
    result, seen = [], set()
    for token in group.get("ids", []):
        prefix, sep, idx = str(token).rpartition("-")
        if not sep or not idx.isdigit() or prefix != ns:
            continue
        i = int(idx)
        if 0 <= i < len(segs) and canonical[i] not in seen:
            seen.add(canonical[i])
            r = segs[i]["rect"]
            length_m = (round(segs[i]["length_units"] * cm_per_unit / 100 * scale_factor, 2)
                        if cm_per_unit is not None else None)
            result.append((segs[i].get("points") or [],
                           length_m,
                           (r.x0 + r.x1) / 2,
                           (r.y0 + r.y1) / 2))
    return result


_LABEL_FONTSIZE = 7
_LABEL_PAD = 2

def _draw_label(page, cx: float, cy: float, text: str) -> None:
    """Draw a measurement label with a white background so it's visible over any segment color."""
    tw = fitz.get_text_length(text, fontname="helv", fontsize=_LABEL_FONTSIZE)
    # insert_text point.y is the text baseline; offset so the label is visually centered at cy
    baseline_y = cy + _LABEL_FONTSIZE * 0.25
    page.draw_rect(
        fitz.Rect(
            cx - tw / 2 - _LABEL_PAD,
            baseline_y - _LABEL_FONTSIZE * 0.8 - _LABEL_PAD,
            cx + tw / 2 + _LABEL_PAD,
            baseline_y + _LABEL_FONTSIZE * 0.2 + _LABEL_PAD,
        ),
        color=(0.5, 0.5, 0.5),
        fill=(1.0, 1.0, 1.0),
        width=0.5,
    )
    page.insert_text(
        fitz.Point(cx - tw / 2, baseline_y),
        text,
        fontname="helv",
        fontsize=_LABEL_FONTSIZE,
        color=(0.1, 0.1, 0.1),
    )


def _draw_path(page, points: list, rgb: tuple) -> None:
    """Trace a segment's path as a thick colored highlight over the original line."""
    if len(points) < 2:
        if points:                       # degenerate: a single vertex -> small box
            x, y = points[0]
            page.draw_rect(fitz.Rect(x - _BOX_PAD, y - _BOX_PAD, x + _BOX_PAD, y + _BOX_PAD),
                           color=rgb, width=_BOX_WIDTH)
        return
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        page.draw_line(fitz.Point(x0, y0), fitz.Point(x1, y1),
                       color=rgb, width=_PATH_WIDTH)


def render_annotations(pdf_path: str, classifications: dict,
                       show_measurements: bool = False,
                       scale_factor: float = 1.0) -> dict:
    """Draw per-task highlight boxes on each page. Returns {pages, legend}.

    pages:  [{"page": n, "image_b64": <png>}] for pages that have annotations.
    legend: [{"task": name, "color": "#rrggbb"}] mapping each task to its color.
    """
    task_color = {}
    for idx, task in enumerate(t for t in classifications if _groups_of(classifications[t])):
        task_color[task] = _OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)]

    doc = fitz.open(pdf_path)
    pages_out = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_no = page_idx + 1
        drew = False
        for task, info in classifications.items():
            rgb = task_color.get(task)
            if rgb is None:
                continue
            for group in _groups_of(info):
                if group.get("page", 1) != page_no:
                    continue
                # Per-meter tasks: trace each segment along its actual path (so an
                # L-shaped wall is highlighted along the line, not boxed by its whole
                # bounding rectangle). Per-unit tasks: one box per clustered physical
                # item (a door's arc+header merged) so the overlay matches the count.
                if _is_per_meter(task):
                    for pts, length_m, cx, cy in _paths_for_group(page, group, scale_factor):
                        _draw_path(page, pts, rgb)
                        if show_measurements and length_m is not None:
                            _draw_label(page, cx, cy, f"{length_m}m")
                        drew = True
                else:
                    for r in cluster_group_item_rects(pdf_path, group):
                        box = fitz.Rect(r.x0 - _BOX_PAD, r.y0 - _BOX_PAD,
                                        r.x1 + _BOX_PAD, r.y1 + _BOX_PAD)
                        page.draw_rect(box, color=rgb, width=_BOX_WIDTH)
                        drew = True
        if not drew:
            continue
        _draw_legend(page, classifications, task_color)
        pix = page.get_pixmap(matrix=fitz.Matrix(_RENDER_ZOOM, _RENDER_ZOOM))
        pages_out.append({
            "page": page_no,
            "image_b64": base64.b64encode(pix.tobytes("png")).decode(),
        })
    doc.close()

    legend = [{"task": t, "color": "#%02x%02x%02x" % tuple(round(c * 255) for c in rgb)}
              for t, rgb in task_color.items()]
    return {"pages": pages_out, "legend": legend}


def _draw_legend(page, classifications, task_color):
    """Draw a small color/task key in the top-left corner of the page."""
    x, y = 12, 16
    for task, info in classifications.items():
        rgb = task_color.get(task)
        if rgb is None:
            continue
        page.draw_rect(fitz.Rect(x, y - 7, x + 10, y + 3), color=rgb, fill=rgb)
        label = task.removesuffix(" (per meter)") if task.endswith(" (per meter)") else task
        page.insert_text((x + 16, y + 2), label, fontsize=8, color=rgb)
        y += 14
