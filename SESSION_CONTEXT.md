# Construction Estimation System — Session Context

This file summarizes the state of the project as of the last working session, for continuity if context is lost.

---

## Project Overview

A LangGraph-based AI system that:
1. Reads construction PDF plans
2. Uses a `task_finder_agent` to detect tasks on each page
3. Uses a `pdf_calculator_agent` to measure elements (lengths, counts) and price each task
4. Exposes results via a FastAPI backend + React/Vite frontend

---

## Active Architecture: Select-by-ID Approach

### How it works (current active approach — "Approach C")

1. Agent calls `list_colored_segments(pdf_path, color, page_number)` — the tool reads the PDF's raw vector geometry and returns every colored segment with a **namespaced deterministic ID** and rich attributes.
2. Agent reads the listing, reasons about which segments belong to the task, and selects IDs.
3. Agent calls `measure_segments_by_id(pdf_path, color, ids, page_number)` with those IDs — tool sums only the selected ones and returns total in meters.

### Why this approach (vs alternatives)

- **Coordinates approach (Approach A)**: Agent passes `[[x1,y1,x2,y2], ...]` normalized fracs. Tried and abandoned — GPT-4o cannot ground coordinates visually, always fabricates round placeholder values.
- **Color-only approach**: `get_wall_lengths_by_color` sums ALL elements of a color blindly — can't distinguish walls from door symbols or dimension lines.
- **Select-by-ID**: Agent only picks integers from a tool-provided list → reliable. Tool does all the geometry. No hallucinated coordinates.

---

## Key Files

| File | Purpose |
|------|---------|
| `wall_measurement_tool.py` | All measurement tools (core logic) |
| `pdf_calculator_agent.py` | Agent definition, system prompt, tools list |
| `task_finder_agent.py` | Upstream agent that detects tasks on each PDF page |
| `helpers/agent_wrap/AgentBuilder.py` | LangGraph agent builder |
| `helpers/skill_loader/` | Skill loader (moved from repo root) |
| `construction_tasks_prices/read_construction_tasks_prices.py` | Task price CRUD |
| `server.py` | FastAPI backend |
| `ui/src/` | React frontend |

---

## `wall_measurement_tool.py` — Key Internals

### Constants (top of file)
```python
_MIN_UNITS = 10
_PLAN_WIDTH_FRACTION = 0.70   # title block is the rightmost ~30% — exclude it
_ON_WALL_ENABLED = True        # flip to False to disable on-wall attribute
_ON_WALL_THRESHOLD = 8.0       # PDF units tolerance for wall proximity
_ON_WALL_SAMPLES = 20          # centerline sample points
_COLOR_CODES = {"red":"R","orange":"O",...,"gray":"GR","grey":"GR"}
```

### Namespace format
`{COLOR_CODE}{PAGE}-{INDEX}` — e.g. `R3-5` = red, page 3, segment index 5.  
This prevents the agent from mixing IDs across different color/page listings.

### Tools available
- `list_colored_segments(pdf_path, color, page_number)` — enumerates all segments with IDs + attributes
- `measure_segments_by_id(pdf_path, color, ids, page_number)` — sums selected IDs only; validates namespace prefix, rejects foreign/malformed tokens
- `get_wall_lengths_by_color(pdf_path, color, page_number, drawing_type)` — legacy; sums all segments of color blindly
- `count_outline_shapes_by_color(pdf_path, color, page_number)` — counts discrete outline shapes (doors, windows)
- `measure_total_length_by_coordinates(pdf_path, segments, page_number)` — coordinate-based (not in active tools, kept for reference)

### Per-segment attributes from `list_colored_segments`
Each line in the output:
```
ID R3-5 | 58.0 cm | vertical | FILLED | center (21%,51%) | on-wall 95%
```
- **length_cm**: exact length from vector geometry × calibrated scale
- **orientation**: horizontal / vertical / square (arc/symbol) / rectangular
- **style**: FILLED (solid interior = real wall) vs OUTLINE (stroke only = often door/symbol/leader line)
- **center**: position as % of page width/height
- **on-wall N%**: fraction of segment's centerline that runs along black architectural linework. High (~80-100%) = sits on a building wall; low (~0-20%) on a long segment = likely a dimension/leader line. Controlled by `_ON_WALL_ENABLED` flag.

### Scale calibration
- Strategy 1: reads `"1:N"` regex from title block text → `N * (2.54/72)` cm per PDF point
- Strategy 2 (fallback): matches colored segments to nearby numeric dimension annotations

---

## `pdf_calculator_agent.py` — Current State

Three prompt/tools variants defined, only Approach C active:

```python
# ACTIVE
SYSTEM_PROMPT_SELECT_IDS = "..."
TOOLS_SELECT_IDS = [get_task_price_tool, multiply_numbers, list_colored_segments,
                    measure_segments_by_id, count_outline_shapes_by_color]

# COMMENTED OUT — coordinates approach (GPT-4o hallucinates)
SYSTEM_PROMPT_COORDS_ONLY = "..."
# TOOLS_COORDS_ONLY = [...]

# COMMENTED OUT — both tools (color + coordinates) approach
# SYSTEM_PROMPT_BOTH_TOOLS = "..."
# TOOLS_BOTH = [...]

agent = AgentBuilder(model=model, tools=TOOLS_SELECT_IDS, system_prompt=SYSTEM_PROMPT_SELECT_IDS)
         .with_memory().pdf_reader().with_todos().build()
```

---

## Next Feature: Option 2 — Classify-then-Measure (NOT YET BUILT)

### What it is
Instead of the agent going directly from `list_colored_segments` output → picking IDs for a task, it must first **tag EVERY segment** in the listing to a task (or "ignore") with a one-line reason.

### Desired agent behavior
After calling `list_colored_segments`, the agent produces a classification table:

```
ID R2-0  → Building Wall (per meter)  [FILLED, 425cm, vertical, on-wall 100%]
ID R2-1  → ignore                      [OUTLINE, 158cm, horizontal, on-wall 5% — likely dimension line]
ID R2-7  → ignore                      [OUTLINE, 381cm, square — door arc, use count_outline_shapes]
ID R2-12 → Demo Wall (per meter)       [FILLED, 310cm, vertical, on-wall 88%]
...
```

Only after tagging every segment does it call `measure_segments_by_id` per task.

### Why this is better than the current approach
- Forces **deliberate per-segment reasoning** instead of a one-shot eyeball pick
- Makes mistakes **visible and debuggable** in the trace
- **Prevents omissions and double-counting** — every segment is accounted for exactly once
- The `on-wall%` attribute (already built) is the key input for this tagging
- **No new tools needed** — pure system prompt change in `SYSTEM_PROMPT_SELECT_IDS`

### How to implement
Change `SYSTEM_PROMPT_SELECT_IDS` to require:
1. Call `list_colored_segments` for each color on the page
2. Output a full classification table (every segment → task name or "ignore" + reason)
3. Only then call `measure_segments_by_id` per task with the tagged IDs

Keep the full table in the agent's output so errors are auditable.

---

## Known Issues / History

- **Namespace collision bug (fixed)**: Agent was mixing IDs from page-2 red listing into page-3 measurement calls. Fixed by namespacing IDs as `{CODE}{PAGE}-{IDX}` and validating prefix in `measure_segments_by_id`.
- **Text labels not extractable**: All descriptive labels on these PDFs are vector paths, not text — `get_text()` returns nothing useful. Only scale tags and some dimension numbers survive as real text.
- **Hebrew on Windows console**: UnicodeEncodeError when printing Hebrew text. Write output to UTF-8 files instead.
- **Door arc detection**: Yellow elements include both wall fills AND door/frame outline symbols. FILLED/OUTLINE distinction separates them correctly.

---

## Recent UI Changes

- `ui/src/components/TaskPanel.jsx` — inline price editing: pencil button → input → save/cancel
- `ui/src/api.js` — added `updateTaskPrice(taskName, newPrice)`
- `server.py` — added `PUT /tasks/{task_name}` endpoint with `UpdateTaskRequest(price: float)`
- `construction_tasks_prices/read_construction_tasks_prices.py` — added `update_task_price(name, price)`

---

## Test PDFs

- `הריסה (1).pdf` — demolition plan, scale 1:50, yellow walls + door symbols
- `סט תוכניות (1).pdf` — multi-page set; red segments on page 2 include both walls and leader lines where on-wall% is the key differentiator
