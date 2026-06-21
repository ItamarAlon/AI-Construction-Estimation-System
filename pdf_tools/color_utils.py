"""Color matching primitives used throughout pdf_tools.

Defines hue targets, saturation thresholds, hex parsing, namespace generation,
and per-drawing color matching. All other modules import from here rather than
duplicating color logic.
"""
import colorsys

# Hue targets in degrees [0, 360). All shades sharing a hue match the same name.
_HUE_TARGETS: dict[str, float] = {
    "red":     0,
    "orange":  30,
    "yellow":  60,
    "green":   120,
    "cyan":    180,
    "blue":    240,
    "magenta": 300,
    "purple":  300,
}
_HUE_TOLERANCE = 30    # degrees — covers dark/light/desaturated shades of the same hue
_MIN_SATURATION = 0.25  # below this, no dominant hue → treat as achromatic

# Achromatic colors (black, gray, white) are the plan's base drawing — existing
# walls, text, dimensions, grid — not task-specific layers. When False,
# list_colored_segments refuses achromatic colors so the agent never classifies
# the whole base map. Flip to True for the rare plan where a gray/black layer is
# a genuine task color.
_LIST_BASE_COLORS = False

# Short, collision-free color codes used to namespace segment IDs (e.g. "R3-5"
# = red, page 3, index 5). "blue"/"black" and "gray"/"grey" must not collide.
_COLOR_CODES: dict[str, str] = {
    "red": "R", "orange": "O", "yellow": "Y", "green": "G",
    "cyan": "C", "blue": "B", "magenta": "M", "purple": "P",
    "black": "K", "white": "W", "gray": "GR", "grey": "GR",
}


def _parse_hex(color: str):
    """Parse '#RRGGBB' (or 'RRGGBB') into an (r, g, b) tuple of 0-1 floats, or None."""
    s = color.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r, g, b = (int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None
    return (r, g, b)


def _namespace(color: str, page_number: int) -> str:
    """Build the ID namespace for a (color, page) listing.

    Named colors get a short code ('red',3 -> 'R3'); a hex target gets a stable
    code from its digits ('#e6f00a',1 -> 'XE6F00A1') so IDs never collide with
    the single-letter named codes and stay consistent between list/measure.
    """
    hex_rgb = _parse_hex(color)
    if hex_rgb is not None:
        code = "X" + color.strip().lstrip("#").upper()
    else:
        code = _COLOR_CODES.get(color.lower(), color.lower())
    return f"{code}{page_number}"


def _matches_color(fill: tuple, color_name: str) -> bool:
    r, g, b = fill[0], fill[1], fill[2]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = h * 360

    hex_rgb = _parse_hex(color_name)
    if hex_rgb is not None:
        th, ts, tv = colorsys.rgb_to_hsv(*hex_rgb)
        if ts < _MIN_SATURATION:  # achromatic target (black/white/gray-ish)
            return s < _MIN_SATURATION and abs(v - tv) <= 0.2
        if s < _MIN_SATURATION:
            return False
        diff = min(abs(h_deg - th * 360), 360 - abs(h_deg - th * 360))
        return diff <= _HUE_TOLERANCE

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


def _is_base_color(color: str) -> bool:
    """True if color is achromatic (black/gray/white) — the plan's base drawing."""
    if color.lower() in ("black", "white", "gray", "grey"):
        return True
    rgb = _parse_hex(color)
    if rgb is None:
        return False
    return colorsys.rgb_to_hsv(*rgb)[1] < _MIN_SATURATION
