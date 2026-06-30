"""Scale calibration: derive real-world cm per PDF unit from the plan's title block."""
import re
import statistics

from pdf_tools.color_utils import _is_chromatic

_MIN_UNITS = 10
# Fraction of the page width that is floor plan; the rest (rightmost strip) is the
# title block, excluded from color/segment detection. 0.70 was too aggressive: on
# plans where the layout extends to the right it sliced through the floor plan and
# dropped real segments (e.g. the vertical piece joining the two halves of a kitchen
# outline, and window-demolition marks), which then broke connectivity counting.
# 0.76 keeps that content while still excluding the title block (which starts ~0.78).
_PLAN_WIDTH_FRACTION = 0.76


def _find_scale_n(text: str) -> int | None:
    """Pick the best '1:N' scale ratio from raw text.

    If multiple different N values appear, use the most frequent one — the
    primary plan scale is mentioned more often (scale bar, title block,
    legend) than incidental detail-view scales.  Ties broken by largest N
    so that an overall 1:500 plan wins over a 1:100 detail when both appear
    equally often.
    """
    all_ns = [int(x) for x in re.findall(r"\b1:(\d+)\b", text)]
    if not all_ns:
        return None
    if len(set(all_ns)) == 1:
        return all_ns[0]
    counts: dict[int, int] = {}
    for n in all_ns:
        counts[n] = counts.get(n, 0) + 1
    return max(counts, key=lambda n: (counts[n], n))


def _calibrate(page) -> float:
    """Derive cm_per_pdf_unit for this page.

    Strategy 1 — scale ratio in title block text (e.g. "1:50").
    Many CAD exports convert dimension numbers to paths, making them
    invisible to text extraction, but the title block scale tag remains
    as real text.  1 PDF point = 1/72 inch = 2.54/72 cm; at scale 1:N
    that point represents N × 2.54/72 real-world cm.

    We search the title block area first (rightmost strip, where _PLAN_WIDTH_FRACTION
    excludes it from plan analysis) to avoid picking up detail-view annotations
    (e.g. "1:100") that appear before the overall plan scale ("1:500") in the
    full-page text stream.

    Strategy 2 — match colored paths to nearby numeric dimension
    annotations (works for PDFs that keep dimensions as text).
    """
    # Strategy 1: search title block area first, then full page
    title_clip = (page.rect.width * _PLAN_WIDTH_FRACTION, 0,
                  page.rect.width, page.rect.height)
    n = _find_scale_n(page.get_text("text", clip=title_clip))
    if n is None:
        n = _find_scale_n(page.get_text("text"))
    if n is not None:
        return n * (2.54 / 72)

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
