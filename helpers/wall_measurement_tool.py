from pathlib import Path
import colorsys
import re
import statistics
from langchain.tools import tool
import pymupdf as fitz

# Hue targets in degrees [0, 360). All shades sharing a hue match the same name.
_HUE_TARGETS: dict[str, float] = {
    "red":     0,
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
    """Derive cm_per_pdf_unit from the plan's own dimension annotations.

    For each colored path in the floor plan area, find the nearest numeric
    annotation and compute annotation_value / path_length. Returns the median,
    making it robust to outliers.
    """
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
        raise ValueError("No numeric dimension annotations found in the floor plan area.")

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


@tool
def get_wall_lengths_by_color(pdf_path: str, color: str) -> str:
    """Measure the total real-world length of all wall segments of a given color in a PDF plan.

    Reads directly from the PDF vector geometry and auto-calibrates the unit conversion
    from the plan's own dimension annotations — works for any scale or paper size.
    Color matching uses HSV hue, so all shades of a color (e.g. dark red, bright red,
    salmon) are treated as the same color.

    Args:
        pdf_path: Path to the PDF construction plan.
        color: Wall color to measure. Supported: red, yellow, green, blue, cyan,
               magenta, purple, black, white, gray.

    Returns:
        Individual segment lengths in cm and total length in meters.
    """
    supported = list(_HUE_TARGETS) + ["black", "white", "gray"]
    if color.lower() not in supported:
        return f"Unknown color '{color}'. Supported: {', '.join(supported)}"

    path = Path(pdf_path.strip("'\""))
    if not path.exists():
        return f"File not found: {pdf_path}"

    doc = fitz.open(str(path))
    page = doc[0]

    try:
        cm_per_unit = _calibrate(page)
    except ValueError as e:
        doc.close()
        return f"Calibration failed: {e}"

    max_x = page.rect.width * _PLAN_WIDTH_FRACTION
    segments: list[float] = []

    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if not fill or len(fill) < 3:
            continue
        if not _matches_color(fill, color):
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
        return f"No {color} wall segments found in the floor plan area."

    total_cm = sum(segments)
    total_m = total_cm / 100
    return (
        f"{color} segments ({len(segments)} found): {sorted(segments)} cm\n"
        f"Total: {total_m:.2f} m ({total_cm:.0f} cm)"
    )
