"""Plain-Python color discovery (no LLM).

Runs BEFORE the estimation agent and lists the distinct colors actually used by
the plan's vector linework, each with its exact hex code and a segment count.
The agent then picks hex codes from this real palette instead of guessing a
color name from a blurry rendered image — fixing both the closed-color-set
limit and the perception gap.
"""
import colorsys

import pymupdf as fitz

from wall_measurement_tool import (
    _PLAN_WIDTH_FRACTION,
    _MIN_UNITS,
    _MIN_SATURATION,
    _HUE_TARGETS,
    _LIST_BLACK_SEGMENTS,
)

# Chromatic colors within this many hue degrees are treated as one cluster.
_CLUSTER_HUE_TOL = 20


def _approx_name(r: float, g: float, b: float) -> str:
    """Best-effort human name for a color (a hint only, not used for matching)."""
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < _MIN_SATURATION:
        if v < 0.2:
            return "black"
        if v > 0.8:
            return "white"
        return "gray"
    h_deg = h * 360
    best, best_diff = "?", 999.0
    for name, target in _HUE_TARGETS.items():
        diff = min(abs(h_deg - target), 360 - abs(h_deg - target))
        if diff < best_diff:
            best, best_diff = name, diff
    return best


def _to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        round(r * 255), round(g * 255), round(b * 255)
    )


def list_present_colors(pdf_path: str, page_number: int = 1) -> list[dict]:
    """Return the distinct colors used by plan linework on a page.

    Each entry: {"hex", "approx_name", "count", "rgb"}. Colors are clustered by
    hue (chromatic) or by lightness (achromatic); the representative is the
    average of the cluster. Sorted by segment count, most common first.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    max_x = page.rect.width * _PLAN_WIDTH_FRACTION

    # Collect one color per qualifying drawing (prefer fill, else stroke).
    samples: list[tuple] = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if (rect.x0 + rect.x1) / 2 >= max_x:
            continue
        if max(rect.width, rect.height) < _MIN_UNITS:
            continue
        col = drawing.get("fill") or drawing.get("color")
        if not col or len(col) < 3:
            continue
        samples.append((col[0], col[1], col[2]))
    doc.close()

    # Cluster: chromatic by rounded hue, achromatic by rounded value.
    clusters: dict[str, dict] = {}
    for r, g, b in samples:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s < _MIN_SATURATION:
            key = f"achr-{round(v * 4)}"   # ~4 lightness buckets
        else:
            key = f"hue-{round(h * 360 / _CLUSTER_HUE_TOL)}"
        c = clusters.setdefault(key, {"r": 0.0, "g": 0.0, "b": 0.0, "count": 0})
        c["r"] += r
        c["g"] += g
        c["b"] += b
        c["count"] += 1

    palette = []
    for c in clusters.values():
        n = c["count"]
        r, g, b = c["r"] / n, c["g"] / n, c["b"] / n
        # Black is the base drawing color, not a task layer — omit it so the agent
        # never picks it (mirrors _LIST_BLACK_SEGMENTS in wall_measurement_tool).
        if not _LIST_BLACK_SEGMENTS and max(r, g, b) < 0.2:
            continue
        palette.append({
            "hex": _to_hex(r, g, b),
            "approx_name": _approx_name(r, g, b),
            "count": n,
            "rgb": (round(r, 3), round(g, 3), round(b, 3)),
        })
    palette.sort(key=lambda e: e["count"], reverse=True)
    return palette


def format_palette(palette: list[dict]) -> str:
    """Human/agent-readable palette block."""
    if not palette:
        return "No colored linework detected."
    lines = ["Detected color palette (use the exact hex with the measurement tools):"]
    for e in palette:
        lines.append(
            f"  {e['hex']}  (~{e['approx_name']}, {e['count']} segments)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(format_palette(list_present_colors(path, page)))
