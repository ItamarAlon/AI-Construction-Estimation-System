"""Render the agent's classification visually on the plan (no new detection).

Takes the per-task segment assignments the agent produced (task -> groups of
{color, page, ids}) and draws a highlighted box around every classified segment,
one overlay color per task, with a legend. Returns per-page PNGs (base64) for the
UI. Pure rendering of data we already have — the IDs resolve to the exact same
rectangles the measurement step uses.

Each page is returned as:
  - base_image_b64: the plain PDF page with no highlights
  - task_layers: {task_name: png_b64} — each task's highlights on a white background

The UI stacks layers with mix-blend-mode: multiply, making white invisible so only
the colored highlights bleed through. This lets users toggle tasks on/off client-side.
"""
import base64
import struct
import zlib

import numpy as np
import pymupdf as fitz

from wall_measurement_tool import (
    _collect_colored_segments,
    _duplicate_canonical,
    _namespace,
    _calibrate,
    cluster_group_item_rects,
)

_PATH_WIDTH = 3.0   # highlight width for per-meter path traces
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
_BOX_PAD = 6
_BOX_WIDTH = 1.6
_RENDER_ZOOM = 2.0


def _is_per_meter(task_name: str) -> bool:
    return task_name.strip().lower().endswith("(per meter)")


def _groups_of(info: dict) -> list[dict]:
    if "groups" in info:
        return info["groups"]
    if "ids" in info:
        return [info]
    return []


def _paths_for_group(page, group: dict, scale_factor: float = 1.0, skip_curved: bool = False) -> list[tuple]:
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
            if skip_curved and segs[i].get("curved"):
                continue
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
    tw = fitz.get_text_length(text, fontname="helv", fontsize=_LABEL_FONTSIZE)
    baseline_y = cy + _LABEL_FONTSIZE * 0.25
    page.draw_rect(
        fitz.Rect(
            cx - tw / 2 - _LABEL_PAD,
            baseline_y - _LABEL_FONTSIZE * 0.8 - _LABEL_PAD,
            cx + tw / 2 + _LABEL_PAD,
            baseline_y + _LABEL_FONTSIZE * 0.2 + _LABEL_PAD,
        ),
        color=(0.6, 0.5, 0.0),
        fill=(1.0, 1.0, 0.75),   # light yellow — not nuked by the near-white transparency pass
        width=0.8,
    )
    page.insert_text(
        fitz.Point(cx - tw / 2, baseline_y),
        text,
        fontname="helv",
        fontsize=_LABEL_FONTSIZE,
        color=(0.0, 0.0, 0.0),
    )


def _draw_path(page, points: list, rgb: tuple) -> None:
    if len(points) < 2:
        if points:
            x, y = points[0]
            page.draw_rect(fitz.Rect(x - _BOX_PAD, y - _BOX_PAD, x + _BOX_PAD, y + _BOX_PAD),
                           color=rgb, width=_BOX_WIDTH)
        return
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        page.draw_line(fitz.Point(x0, y0), fitz.Point(x1, y1),
                       color=rgb, width=_PATH_WIDTH)


def _render_b64(page, matrix) -> str:
    pix = page.get_pixmap(matrix=matrix)
    return base64.b64encode(pix.tobytes("png")).decode()


def _render_b64_transparent(page, matrix) -> str:
    """Render as RGBA PNG; pixels where R,G,B all > 250 become transparent.

    Used for measurement label overlays so the white page background disappears
    while the light-yellow label boxes and black text remain fully opaque.
    The UI stacks this with mix-blend-mode: normal (transparent bg means no cover).
    PNG is hand-encoded with struct+zlib so no PIL dependency is needed.
    """
    pix = page.get_pixmap(matrix=matrix)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
    near_white = (rgb[:, :, 0] > 250) & (rgb[:, :, 1] > 250) & (rgb[:, :, 2] > 250)
    alpha = np.where(near_white, np.uint8(0), np.uint8(255))
    rgba = np.dstack([rgb, alpha])   # shape: (h, w, 4)

    h, w = rgba.shape[:2]
    # Build PNG raw image data: one filter byte (0 = None) per scanline
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def render_annotations(pdf_path: str, classifications: dict,
                       scale_factor: float = 1.0) -> dict:
    """Returns {pages, legend}.

    pages: list of {page, base_image_b64, task_layers: {task: png_b64}, measurement_layers: {task: png_b64}}
    legend: [{task, color}]
    """
    task_color = {}
    for idx, task in enumerate(t for t in classifications if _groups_of(classifications[t])):
        task_color[task] = _OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)]

    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(_RENDER_ZOOM, _RENDER_ZOOM)
    pages_out = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_no = page_idx + 1

        # Collect drawing commands per task for this page.
        # Tuples: ("path", pts, rgb) | ("label", cx, cy, text) | ("box", rect, rgb)
        task_cmds: dict[str, list] = {}

        for task, info in classifications.items():
            rgb = task_color.get(task)
            if rgb is None:
                continue
            for group in _groups_of(info):
                if group.get("page", 1) != page_no:
                    continue
                cmds = task_cmds.setdefault(task, [])
                if _is_per_meter(task):
                    is_door = any(kw in task.lower() for kw in ("door", "דלת"))
                    for pts, length_m, cx, cy in _paths_for_group(page, group, scale_factor, skip_curved=is_door):
                        cmds.append(("path", pts, rgb))
                        if length_m is not None:
                            cmds.append(("label", cx, cy, f"{length_m}m"))
                else:
                    for r in cluster_group_item_rects(pdf_path, group):
                        cmds.append(("box", r, rgb))

        if not task_cmds:
            continue

        # Base image: plain PDF page, no annotations.
        base_b64 = _render_b64(page, matrix)

        # Per-task overlay layers (paths/boxes only — no labels).
        # White background + mix-blend-mode: multiply in the UI = white is invisible.
        task_layers = {}
        measurement_layers = {}   # task -> label-only overlay, rendered separately so they
                                  # can be toggled independently of the segment highlights
        for task, cmds in task_cmds.items():
            seg_doc = fitz.open()
            seg_page = seg_doc.new_page(width=page.rect.width, height=page.rect.height)
            label_cmds = []
            for cmd in cmds:
                if cmd[0] == "path":
                    _, pts, rgb = cmd
                    _draw_path(seg_page, pts, rgb)
                elif cmd[0] == "label":
                    label_cmds.append(cmd)
                elif cmd[0] == "box":
                    _, r, rgb = cmd
                    box = fitz.Rect(r.x0 - _BOX_PAD, r.y0 - _BOX_PAD,
                                    r.x1 + _BOX_PAD, r.y1 + _BOX_PAD)
                    seg_page.draw_rect(box, color=rgb, width=_BOX_WIDTH)
            task_layers[task] = _render_b64(seg_page, matrix)
            seg_doc.close()

            if label_cmds:
                # Deduplicate labels whose centres are within 25 PDF points of each
                # other (the two parallel faces of the same wall both get a label at
                # nearly the same position — keep only the first one encountered).
                deduped: list = []
                for cmd in label_cmds:
                    _, cx, cy, _ = cmd
                    if not any(
                        abs(cx - ex) < 25 and abs(cy - ey) < 25
                        for _, ex, ey, _ in deduped
                    ):
                        deduped.append(cmd)

                lbl_doc = fitz.open()
                lbl_page = lbl_doc.new_page(width=page.rect.width, height=page.rect.height)
                for _, cx, cy, text in deduped:
                    _draw_label(lbl_page, cx, cy, text)
                measurement_layers[task] = _render_b64_transparent(lbl_page, matrix)
                lbl_doc.close()

        pages_out.append({
            "page": page_no,
            "base_image_b64": base_b64,
            "task_layers": task_layers,
            "measurement_layers": measurement_layers,
        })

    doc.close()

    legend = [
        {"task": t, "color": "#%02x%02x%02x" % tuple(round(c * 255) for c in rgb)}
        for t, rgb in task_color.items()
    ]
    return {"pages": pages_out, "legend": legend}
