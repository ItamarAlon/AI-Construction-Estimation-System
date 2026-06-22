"""Per-unit item counting: cluster tagged segments into discrete physical items.

A per-unit item (e.g. a door = arc + jamb line) is made up of several segments
that are spatially connected. This module groups them by bounding-box proximity
and counts distinct items, with cross-page deduplication so the same item on
multiple sheets of the same layout is counted once.
"""
import math
from pathlib import Path

import pymupdf as fitz

from pdf_tools.color_utils import _namespace
from pdf_tools.segment_geometry import (
    _collect_colored_segments, _duplicate_canonical, _DUP_CENTER_TOL,
)

# Two boxes merge when the gap between them is at most this fraction of the
# smaller box's size. 0 would require actual overlap; a small slack tolerates
# the hairline gap between an arc and its header. Stable for 0.0–0.35 on
# הריסה (8 doors), over-merges adjacent doors at >=0.5.
_SYMBOL_GAP_FACTOR = 0.25


def _rect_gap(a, b) -> float:
    """Shortest distance between two axis-aligned rects (0 if they overlap/touch)."""
    dx = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return math.hypot(dx, dy)


def _collect_group_rects(pdf_path: str, group: dict) -> list[dict]:
    """Resolve one {color, page, ids} group to kept segment rects (page coords).

    Each entry is {rect, page, w, h}; within-page double-line duplicates are
    collapsed via _duplicate_canonical so a doubled outline isn't two items.
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
    segs = _collect_colored_segments(page, color)
    canonical = _duplicate_canonical(segs, page)
    w, h = page.rect.width, page.rect.height
    doc.close()

    kept, seen_reps = [], set()
    for i in indices:
        if i < 0 or i >= len(segs):
            continue
        rep = canonical[i]
        if rep in seen_reps:
            continue
        seen_reps.add(rep)
        kept.append({"rect": segs[i]["rect"], "page": page_number, "w": w, "h": h})
    return kept


def _cluster_connected(rects: list) -> list[list[int]]:
    """Union-find clustering of rects by bounding-box connectivity.

    Returns clusters as lists of member indices into `rects`. Two rects join
    when the gap between their boxes is within _SYMBOL_GAP_FACTOR of the smaller
    box's size (an item's parts touch; separate items don't).
    """
    n = len(rects)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        ri = rects[i]["rect"]
        size_i = max(ri.width, ri.height)
        for j in range(i + 1, n):
            rj = rects[j]["rect"]
            size_j = max(rj.width, rj.height)
            tol = _SYMBOL_GAP_FACTOR * max(min(size_i, size_j), 1e-9)
            if _rect_gap(ri, rj) <= tol:
                parent[find(i)] = find(j)

    clusters: dict[int, list] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _cluster_centers(rects: list) -> list[dict]:
    """Cluster rects and return one {cx, cy, page} center (page fraction) each."""
    reps = []
    for members in _cluster_connected(rects):
        xs = [(rects[i]["rect"].x0 + rects[i]["rect"].x1) / 2 for i in members]
        ys = [(rects[i]["rect"].y0 + rects[i]["rect"].y1) / 2 for i in members]
        sample = rects[members[0]]
        reps.append({
            "cx": (sum(xs) / len(xs)) / sample["w"],
            "cy": (sum(ys) / len(ys)) / sample["h"],
            "page": sample["page"],
        })
    return reps


def cluster_group_item_rects(pdf_path: str, group: dict) -> list:
    """Merged bounding box per clustered item in a {color, page, ids} group.

    Each per-unit item (e.g. a door = arc + header) becomes ONE fitz.Rect (the
    union of its segments' boxes), in page coordinates — so the annotation can
    draw one box per physical item instead of one per raw segment.
    """
    rects = _collect_group_rects(pdf_path, group)
    out = []
    for members in _cluster_connected(rects):
        boxes = [rects[i]["rect"] for i in members]
        merged = fitz.Rect(
            min(b.x0 for b in boxes), min(b.y0 for b in boxes),
            max(b.x1 for b in boxes), max(b.y1 for b in boxes),
        )
        out.append(merged)
    return out


def count_task_groups(pdf_path: str, groups: list[dict]) -> int:
    """Count discrete physical items for a per-unit task across its groups.

    Each group's tagged segments are clustered into items by spatial
    connectivity (an item's parts touch; separate items don't). Items that
    repeat across pages of the same layout (same center) are counted once.
    """
    pooled_items: list[dict] = []
    for g in groups:
        rects = _collect_group_rects(pdf_path, g)
        pooled_items.extend(_cluster_centers(rects))

    kept: list[dict] = []
    for item in pooled_items:
        dup = False
        for k in kept:
            if k["page"] == item["page"]:
                continue
            if (
                abs(item["cx"] - k["cx"]) <= _DUP_CENTER_TOL
                and abs(item["cy"] - k["cy"]) <= _DUP_CENTER_TOL
            ):
                dup = True
                break
        if not dup:
            kept.append(item)

    return len(kept)
