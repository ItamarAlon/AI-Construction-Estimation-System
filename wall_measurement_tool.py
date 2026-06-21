"""Backward-compatibility re-export stub.

All logic has moved to the pdf_tools/ package. This file exists so that any
code still importing from wall_measurement_tool continues to work unchanged.
"""
from pdf_tools.color_utils import (
    _HUE_TARGETS,
    _HUE_TOLERANCE,
    _MIN_SATURATION,
    _LIST_BASE_COLORS,
    _COLOR_CODES,
    _parse_hex,
    _namespace,
    _matches_color,
    _is_chromatic,
    _is_base_color,
)
from pdf_tools.calibration import (
    _MIN_UNITS,
    _PLAN_WIDTH_FRACTION,
    _calibrate,
)
from pdf_tools.segment_geometry import (
    _ON_WALL_ENABLED,
    _ON_WALL_THRESHOLD,
    _ON_WALL_SAMPLES,
    _DUP_ENABLED,
    _DUP_LENGTH_TOL,
    _DUP_CENTER_TOL,
    _COLOCATED_TOL,
    _CROP_PAD,
    _CROP_MIN_SIDE,
    _CROP_TARGET_PX,
    _CROP_MAX_ZOOM,
    _collect_colored_segments,
    _duplicate_canonical,
    _colocated_counts,
    _is_black,
    _collect_black_lines,
    _point_segment_distance,
    _on_wall_fraction,
    _crop_segment_png,
    _describe_segment,
)
from pdf_tools.agent_tools import (
    _merge_nearby_rects,
    count_outline_shapes_by_color,
    get_wall_lengths_by_color,
    measure_total_length_by_coordinates,
    list_colored_segments,
    measure_segments_by_id_tool,
    measure_segments_by_id,
)
from pdf_tools.measurement import (
    _collect_group_segments,
    measure_task_groups,
)
from pdf_tools.clustering import (
    _SYMBOL_GAP_FACTOR,
    _rect_gap,
    _collect_group_rects,
    _cluster_connected,
    _cluster_centers,
    cluster_group_item_rects,
    count_task_groups,
)

__all__ = [
    "_HUE_TARGETS", "_HUE_TOLERANCE", "_MIN_SATURATION", "_LIST_BASE_COLORS",
    "_COLOR_CODES", "_parse_hex", "_namespace", "_matches_color", "_is_chromatic",
    "_is_base_color", "_MIN_UNITS", "_PLAN_WIDTH_FRACTION", "_calibrate",
    "_ON_WALL_ENABLED", "_ON_WALL_THRESHOLD", "_ON_WALL_SAMPLES",
    "_DUP_ENABLED", "_DUP_LENGTH_TOL", "_DUP_CENTER_TOL", "_COLOCATED_TOL",
    "_CROP_PAD", "_CROP_MIN_SIDE", "_CROP_TARGET_PX", "_CROP_MAX_ZOOM",
    "_collect_colored_segments", "_duplicate_canonical", "_colocated_counts",
    "_is_black", "_collect_black_lines", "_point_segment_distance",
    "_on_wall_fraction", "_crop_segment_png", "_describe_segment",
    "_merge_nearby_rects", "count_outline_shapes_by_color", "get_wall_lengths_by_color",
    "measure_total_length_by_coordinates", "list_colored_segments",
    "measure_segments_by_id_tool", "measure_segments_by_id",
    "_collect_group_segments", "measure_task_groups",
    "_SYMBOL_GAP_FACTOR", "_rect_gap", "_collect_group_rects", "_cluster_connected",
    "_cluster_centers", "cluster_group_item_rects", "count_task_groups",
]
