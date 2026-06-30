"""Per-meter task measurement: resolve tagged segment IDs to a total length in meters.

measure_task_groups is the main entry point for the graph's measure node.
It handles cross-page deduplication so the same layout on multiple sheets
is counted only once.
"""
from pathlib import Path

import pymupdf as fitz

from pdf_tools.color_utils import _namespace
from pdf_tools.calibration import _calibrate
from pdf_tools.segment_geometry import (
    _collect_colored_segments, _duplicate_canonical,
    _DUP_LENGTH_TOL, _DUP_CENTER_TOL,
)


def _collect_group_segments(pdf_path: str, group: dict, skip_curved: bool = False) -> list[dict]:
    """Resolve one {color, page, ids} group to a list of kept segments.

    Each kept segment is {length_cm, cx, cy, page} with center as a fraction
    (0-1) of the page, so segments from different pages of the same sheet are
    comparable. Within-page double-line duplicates are already collapsed here.

    skip_curved: when True, arc/curve segments are excluded. Used for door
    tasks where only the straight jamb line (the real door width) should be
    measured, not the swing arc.
    """
    color = group["color"]
    page_number = group.get("page", 1)
    ids = group.get("ids", [])
    path_obj = Path(pdf_path.strip("'\""))
    if not ids or not path_obj.exists():
        return []

    expected_ns = _namespace(color, page_number)
    indices = []
    for token in ids:
        prefix, sep, idx_str = str(token).rpartition("-")
        if sep and idx_str.isdigit() and prefix == expected_ns:
            indices.append(int(idx_str))

    doc = fitz.open(str(path_obj))
    page_idx = page_number - 1
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        return []
    page = doc[page_idx]
    try:
        cm_per_unit = _calibrate(page)
    except ValueError:
        doc.close()
        return []
    segs = _collect_colored_segments(page, color)
    canonical = _duplicate_canonical(segs, page)
    w, h = page.rect.width, page.rect.height
    doc.close()

    kept = []
    seen_reps = set()
    for i in indices:
        if i < 0 or i >= len(segs):
            continue
        if skip_curved and segs[i].get("curved"):
            continue
        rep = canonical[i]
        if rep in seen_reps:
            continue
        seen_reps.add(rep)
        r = segs[i]["rect"]
        kept.append({
            "length_cm": segs[i]["length_units"] * cm_per_unit,
            "cx": (r.x0 + r.x1) / 2 / w,
            "cy": (r.y0 + r.y1) / 2 / h,
            "page": page_number,
        })
    return kept


def measure_task_groups(pdf_path: str, groups: list[dict], skip_curved: bool = False) -> float:
    """Total meters for a task spanning one or more {color, page, ids} groups.

    Segments that are geometrically identical across pages (same length and
    center) are counted once — so the same layout drawn on several sheets (e.g.
    page 5 = page 4 + additions) yields the union, not the sum.
    """
    pooled: list[dict] = []
    for g in groups:
        pooled.extend(_collect_group_segments(pdf_path, g, skip_curved=skip_curved))

    kept: list[dict] = []
    for s in pooled:
        dup = False
        for k in kept:
            if k["page"] == s["page"]:
                continue
            longer = max(s["length_cm"], k["length_cm"], 1e-9)
            if (
                abs(s["length_cm"] - k["length_cm"]) / longer <= _DUP_LENGTH_TOL
                and abs(s["cx"] - k["cx"]) <= _DUP_CENTER_TOL
                and abs(s["cy"] - k["cy"]) <= _DUP_CENTER_TOL
            ):
                dup = True
                break
        if not dup:
            kept.append(s)

    return round(sum(s["length_cm"] for s in kept) / 100, 2)
