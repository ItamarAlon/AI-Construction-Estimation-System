"""One-off: compare OLD (max-bbox) vs NEW (gated true-path) measurement per task,
on real tagged classifications, to verify walls stay flat and L-shapes rise.
Not committed."""
import math
import pymupdf as fitz

from pdf_tools.calibration import _calibrate
from pdf_tools.color_utils import _namespace
from pdf_tools.segment_geometry import (
    _collect_colored_segments, _duplicate_canonical, _path_points,
)

_RATIO_CAP = 3.0   # open paths longer than this * max(bbox) fall back to max(bbox)


def _piece_len(points):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def new_length_units(seg):
    """Gated true-path length: closed/filled keep max(bbox); open paths use true
    path length capped at _RATIO_CAP * max(bbox)."""
    r = seg["rect"]
    bbox = max(r.width, r.height)
    if seg.get("filled"):
        return bbox, "filled"
    pts = seg.get("points") or []
    if len(pts) < 3:               # single straight piece (or degenerate)
        poly = _piece_len(pts)
        return (poly if 0 < poly <= bbox * _RATIO_CAP else bbox), "line"
    poly = _piece_len(pts)
    if poly <= bbox * _RATIO_CAP:
        return poly, "open-bend"
    return bbox, "capped"


def compare(pdf_path, classification):
    doc = fitz.open(pdf_path)
    for task, info in classification.items():
        groups = info.get("groups") or ([info] if "ids" in info else [])
        old_tot = new_tot = 0.0
        kinds = {}
        for g in groups:
            color = g["color"]; page_no = g.get("page", 1)
            page = doc[page_no - 1]
            try:
                cmpu = _calibrate(page)
            except ValueError:
                continue
            segs = _collect_colored_segments(page, color)
            canon = _duplicate_canonical(segs, page)
            ns = _namespace(color, page_no)
            seen = set()
            for token in g.get("ids", []):
                pre, sep, idx = str(token).rpartition("-")
                if not sep or not idx.isdigit() or pre != ns:
                    continue
                i = int(idx)
                if not (0 <= i < len(segs)) or canon[i] in seen:
                    continue
                seen.add(canon[i])
                r = segs[i]["rect"]
                old = max(r.width, r.height)
                new, kind = new_length_units(segs[i])
                old_tot += old * cmpu
                new_tot += new * cmpu
                kinds[kind] = kinds.get(kind, 0) + 1
        print(f"  {task[:45]:45s} OLD={old_tot/100:7.2f}m  NEW={new_tot/100:7.2f}m  "
              f"d={(new_tot-old_tot)/100:+6.2f}m  {kinds}")
    doc.close()


if __name__ == "__main__":
    import json, sys
    pdf = sys.argv[1]
    cls = json.load(open(sys.argv[2]))
    print(pdf.split("/")[-1].split("\\")[-1])
    compare(pdf, cls)
