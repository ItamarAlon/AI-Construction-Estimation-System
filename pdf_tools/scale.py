"""Auto-detect PDF scale errors by comparing dimension annotation text to leader lines.

The title block declares a nominal scale (e.g. 1:50), but CAD exports sometimes
embed geometry at a different effective scale.  This module measures the real
cm-per-unit ratio directly from the annotation leader lines in the PDF and computes
a correction factor to apply to all per-meter measurements.
"""
import re
import statistics

import pymupdf as fitz

from pdf_tools.calibration import _calibrate
from logs.write_logs import write_logs


def _compute_scale_geometric(page) -> float | None:
    """Return a scale correction factor for this page, or None if undeterminable.

    Matches every numeric text annotation to its nearest horizontal/vertical
    dimension leader line, computes the median cm-per-unit ratio, then divides
    by the title-block cpu to get the correction factor:
      - 0.85–1.10 → annotations match the title block; no detectable scale error
      - outside    → return the ratio directly as the scale correction factor
    """
    title_block_cpu = _calibrate(page)

    annots = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
        if not re.fullmatch(r"\d{2,4}", word):
            continue
        val = int(word)
        if not (10 <= val <= 3000):
            continue
        annots.append((val, (x0 + x1) / 2, (y0 + y1) / 2))

    if not annots:
        return None

    lines = []
    for d in page.get_drawings():
        fill = d.get("fill")
        if fill and len(fill) >= 3:
            r, g, b = fill[0], fill[1], fill[2]
            if max(r, g, b) - min(r, g, b) > 0.1:  # skip chromatic fills (wall paths)
                continue
        for item in d.get("items", []):
            if item[0] != "l":
                continue
            x1_, y1_, x2_, y2_ = item[1].x, item[1].y, item[2].x, item[2].y
            length = ((x2_ - x1_) ** 2 + (y2_ - y1_) ** 2) ** 0.5
            if length < 10:
                continue
            if abs(y2_ - y1_) < 3 or abs(x2_ - x1_) < 3:  # horizontal or vertical only
                lines.append(((x1_ + x2_) / 2, (y1_ + y2_) / 2, length))

    if not lines:
        return None

    ratios = []
    for val, cx, cy in annots:
        nearest = min(lines, key=lambda l: (cx - l[0]) ** 2 + (cy - l[1]) ** 2)
        dist = ((cx - nearest[0]) ** 2 + (cy - nearest[1]) ** 2) ** 0.5
        if dist > 80:
            continue
        ratios.append(val / nearest[2])

    if len(ratios) < 3:
        return None

    med = statistics.median(ratios)
    filtered = [r for r in ratios if 0.8 * med <= r <= 1.2 * med]
    if len(filtered) < 3:
        return None

    annotation_cpu = statistics.median(filtered)
    raw_ratio = annotation_cpu / title_block_cpu

    if 0.85 <= raw_ratio <= 1.10:
        write_logs(f"scale_geometric: ratio={raw_ratio:.4f} near 1.0 — no scale error detected")
        return None

    scale_factor = raw_ratio
    write_logs(
        f"scale_geometric: title_block_cpu={title_block_cpu:.4f}, "
        f"annotation_cpu={annotation_cpu:.4f} (from {len(filtered)}/{len(ratios)} matches), "
        f"scale_factor={scale_factor:.4f}"
    )
    return scale_factor


def compute_scale_factor(pdf_path: str) -> float | None:
    """Open pdf_path and return a scale correction factor for page 1, or None."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    result = _compute_scale_geometric(page)
    doc.close()
    return result
