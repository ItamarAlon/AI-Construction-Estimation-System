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
    cluster_group_item_rects,
)


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


def _rects_for_group(page, group: dict) -> list:
    """Resolve a group's IDs to fitz.Rect boxes on this page (dups collapsed)."""
    color = group["color"]
    ns = _namespace(color, group.get("page", 1))
    segs = _collect_colored_segments(page, color)
    canonical = _duplicate_canonical(segs, page)
    rects, seen = [], set()
    for token in group.get("ids", []):
        prefix, sep, idx = str(token).rpartition("-")
        if not sep or not idx.isdigit() or prefix != ns:
            continue
        i = int(idx)
        if 0 <= i < len(segs) and canonical[i] not in seen:
            seen.add(canonical[i])
            rects.append(segs[i]["rect"])
    return rects


def render_annotations(pdf_path: str, classifications: dict) -> dict:
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
                # Per-meter tasks: one box per segment. Per-unit tasks: one box
                # per clustered physical item (a door's arc+header merged) so the
                # overlay matches the counted quantity instead of showing N boxes.
                if _is_per_meter(task):
                    rects = _rects_for_group(page, group)
                else:
                    rects = cluster_group_item_rects(pdf_path, group)
                for r in rects:
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
        page.insert_text((x + 16, y + 2), task, fontsize=8, color=rgb)
        y += 14
