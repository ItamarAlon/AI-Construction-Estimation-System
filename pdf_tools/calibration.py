"""Scale calibration: derive real-world cm per PDF unit from the plan's title block."""
import re
import statistics

from pdf_tools.color_utils import _is_chromatic

_MIN_UNITS = 10
_PLAN_WIDTH_FRACTION = 0.70  # title blocks occupy the rightmost ~25-30% of the page


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
