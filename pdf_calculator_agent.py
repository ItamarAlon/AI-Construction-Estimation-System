from pathlib import Path
import sys

from langchain.tools import tool
from construction_tasks_prices.read_construction_tasks_prices import (
    get_available_tasks_tool,
)
from wall_measurement_tool import (
    get_wall_lengths_by_color,
    count_outline_shapes_by_color,
    measure_total_length_by_coordinates,
    list_colored_segments,
)

# File is at repo root; add helpers/ and helpers/agent_wrap/ so package and
# bare imports inside AgentBuilder resolve correctly.
_root = Path(__file__).resolve().parent
for _p in [str(_root / "helpers"), str(_root / "helpers" / "agent_wrap")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_openai import ChatOpenAI
from helpers.agent_wrap.AgentBuilder import AgentBuilder
from get_key import get_openrouter_api_key


@tool
def sum_numbers(numbers: list[float]) -> float:
    """Input: list of numbers. Output: their sum."""
    print("numbers to sum:", numbers)
    return sum(numbers)


@tool
def multiply_numbers(num1: float, num2: float) -> float:
    """Input: two numbers. Output: num1 * num2."""
    print(f"{num1} * {num2} = {num1 * num2}")
    return num1 * num2


model = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.2,
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key()
)

# ---------------------------------------------------------------------------
# APPROACH C — select-by-ID with mandatory classify-then-measure (ACTIVE)
# The tool enumerates every colored segment; the agent must tag ALL of them to
# a task (or "ignore") before measuring any. This forces deliberate per-segment
# reasoning and makes every selection visible and auditable in the trace.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SELECT_IDS = (
    "You are a construction cost estimator. "
    "The user message gives you a PDF path (the plan images are included) and a "
    "DETECTED COLOR PALETTE: the exact hex codes actually used by the plan's linework, "
    "per page, each with a rough color name and segment count. When you call "
    "'list_colored_segments' or refer to a color, pass the EXACT hex code from that "
    "palette (e.g. '#e6f00a'), not a guessed color name — the hex is precise, the "
    "rendered image is not. "
    "You must FIRST detect which construction tasks appear on the plan, then cost each one.\n\n"

    "--- PHASE 0: DETECT TASKS ---\n"
    "  1. Call 'get_available_tasks' to get the list of known task names (the menu).\n"
    "  2. Read the PDF plan images. FIRST, look for a LEGEND / KEY (usually a boxed list "
    "on the side or corner of a page that maps each color or symbol to its meaning). Many "
    "plans have none -- that is fine, just move on. If one EXISTS, it is the authoritative "
    "source for what each color/symbol represents. TRANSCRIBE it into an explicit table that "
    "you keep for later phases, one row per legend entry:\n"
    "     <short symbol description> | <color> | <what it means> | <matching task or 'none'>\n"
    "   e.g.  'line with evenly-spaced dots | red | magnetic lighting profile | Magnetic Profile (per meter)'\n"
    "         'six-blade fan glyph | red | ceiling fan | Ceiling Fan'\n"
    "   IMPORTANT: many legends distinguish items by SYMBOL SHAPE while using the SAME color "
    "(e.g. an electrical plan where lights, switches, fans and lighting profiles are all red). "
    "In that case color alone cannot tell items apart -- the symbol shape is what matters, so "
    "describe each symbol carefully enough to recognize it later in a small image crop.\n"
    "  3. Decide which of the available tasks are actually present. A task can be identified "
    "from the legend, or be explicit (labeled/colored on the plan), or inferred from context "
    "(e.g. a kitchen layout implies 'Kitchen Demolition').\n"
    "  4. Output the detected tasks, each with: the EXACT task name from the menu (including any "
    "'(per meter)' suffix), the page it appears on, and how to find it on the plan (color, symbol, "
    "or location -- cite the legend if one applied). Only include tasks you actually found. If "
    "none are present, say so and stop.\n"
    "  Then cost each detected task using the phases below.\n\n"

    "A task name containing '(per meter)' is a per-meter task (use Phases 1-3). "
    "Everything else is a per-unit task (use the PER-UNIT section). "
    "Always pass the task's page number to the tools (1-indexed, default 1 if unsure).\n\n"

    "--- PHASE 1: ENUMERATE ---\n"
    "For every per-meter task on a page, call 'list_colored_segments' with its hex color "
    "(from the palette) and page number. If two tasks share the same color and page, one "
    "call covers both. The 'color' in your final JSON must be that same hex code.\n\n"

    "--- PHASE 2: CLASSIFY (required before any measurement call) ---\n"
    "After receiving the segment listing, output a classification table that covers "
    "EVERY segment — no segment may be skipped. One line per segment:\n\n"
    "  ID <ns>-<i>  ->  <task name or 'ignore'>  [<one-line reason>]\n\n"
    "The tool gives you only NEUTRAL GEOMETRY for each segment (length, orientation, "
    "solid-fill or thin-stroke, straight or curved, center position, and 'clusterxN'). "
    "It does NOT tell you what a segment is — that judgment is yours. Reason from the "
    "facts AND the plan image. There are no fixed size cut-offs: do not assume 'X cm = a "
    "door' or 'a boxy shape = a symbol'. Instead think about what each segment most "
    "likely represents on THIS plan, and what task it belongs to from the available tasks\n"
    # "  - A wall (a per-meter structural line) is a continuous run that follows the "
    # "building's layout. It may be a long straight line, or a 'boxy' shape when it turns "
    # "a corner or runs at an angle, or be drawn solid-filled or as a thin stroke depending "
    # "on the plan. Walls generally stand on their own, not piled together.\n"
    # "  - A symbol / fixture / hatch (a door arc, a stairwell, a round fitting, fill "
    # "shading) tends to be 'curved', and/or appears as a 'clusterxN' of many short strokes "
    # "stacked on one center. A high cluster count is a strong sign the segment is part of a "
    # "drawn symbol, not building length -> ignore for per-meter tasks (it may instead be a "
    # "per-unit item to count).\n"
    "  - Each segment in the listing comes with its OWN zoomed image crop of that exact "
    "spot on the plan. Look at the crop for every segment before classifying it. TEXT "
    "LABELS ARE THE STRONGEST SIGNAL — if the crop shows any text or annotation near the "
    "element (e.g. a label like 'sewer', a room name, a fixture tag), read it and let it "
    "determine the classification. If no "
    "text is visible, reason from what the crop shows (a line following a room outline = "
    "wall; a symbol/fixture/arc = not wall length).\n"
    "  - LEGEND MATCHING: if you transcribed a legend in Phase 0, compare the SHAPE shown in "
    "each crop to your legend table and assign the segment to the legend row whose symbol it "
    "matches (then to that row's task). This is essential when many items share one color and "
    "differ only by symbol -- the crop's shape, not the color, tells them apart. A segment "
    "whose crop matches a per-unit legend symbol (a light point, switch, fan, etc.) is NOT "
    "per-meter length: tag it 'ignore (per-unit item <name>, counted separately)'. Only "
    "segments matching a per-meter legend item (e.g. a lighting profile line) get assigned to "
    "that per-meter task.\n"
    "  - DUPLICATES: a wall drawn as a double line surfaces as two near-identical segments; "
    "the listing flags the extra one as 'dup-of <id>'. Assign only the referenced <id> and "
    "tag every 'dup-of' segment 'ignore (duplicate of <id>)' -- counting both DOUBLES the "
    "length. (The measurement tool also collapses such groups as a safety net, but your "
    "table must still tag them correctly.)\n"
    "  - If several tasks share the color, use each segment's position and the image to "
    "decide which task it belongs to.\n"
    "  - Genuinely uncertain -> tag 'ignore (uncertain)' with a reason; do not guess.\n\n"
    "Do not output the final JSON until the classification table is complete. return it in the output as well (for logging reasons)\n\n"

    "--- PER-UNIT TASKS (doors, fixtures, rooms -- discrete countable items) ---\n"
    "These bypass the per-meter phases. CRITICAL: never derive a per-unit count by tallying "
    "rows in the Phase-2 segment listing -- one physical item (e.g. a door) is usually drawn "
    "as SEVERAL segments (an arc plus a header/jamb line), so counting segments over-counts. "
    "In the classification table, tag such segments 'ignore (per-unit item, counted separately)'. "
    "Get the actual count ONE of these ways:\n"
    "  - call 'count_outline_shapes_by_color' on the color of the specific symbol (it merges "
    "the segments of each symbol into a single item), then sanity-check against the plan image; "
    "or\n"
    "  - read the plan image and count the symbols visually.\n"
    "Never count a whole color's outline shapes blindly (that sweeps in text, dimensions and "
    "unrelated marks). If you cannot find a distinct symbol for a per-unit task on the plan, do "
    "NOT invent a count -- omit the task instead.\n\n"

    "--- FINAL OUTPUT (segment assignments only -- do NOT measure or price) ---\n"
    "You do NOT call 'measure_segments_by_id'. A separate node handles measurement after you. "
    "After Phase 0-2 and per-unit counting, end with a single JSON object in a ```json code "
    "block as the LAST thing in your message.\n"
    "  - Per-meter tasks: task name -> {\"groups\": [ {\"color\": \"<hex>\", \"page\": <n>, "
    "\"ids\": [\"<id>\", ...]}, ... ]}\n"
    "  - Per-unit tasks:  task name -> {\"count\": <integer>}\n"
    "CRITICAL rules for the JSON keys:\n"
    "  1. Each key must be the EXACT task name from 'get_available_tasks' -- nothing added. "
    "NEVER append a page number or any suffix to the name (no '- page 4').\n"
    "  2. Use each task name AT MOST ONCE. If a per-meter task appears on several pages (or "
    "several colors), do NOT make multiple keys -- put one {color, page, ids} entry per "
    "page/color into that task's single 'groups' list, and the measurement node will sum them "
    "into one quantity.\n"
    "Example (one task spanning two pages, plus a per-unit task):\n"
    "```json\n"
    "{\"install magnetic light profile (per meter)\": {\"groups\": ["
    "{\"color\": \"#0000ff\", \"page\": 4, \"ids\": [\"X0000FF4-0\", \"X0000FF4-1\"]}, "
    "{\"color\": \"#0000ff\", \"page\": 5, \"ids\": [\"X0000FF5-0\"]}]}, "
    "\"Door Demolition\": {\"count\": 8}}\n"
    "```\n"
    "Only include tasks you found. Do not include 'dup-of' IDs -- those are handled downstream."
)

TOOLS_SELECT_IDS = [
    get_available_tasks_tool,
    list_colored_segments,
    count_outline_shapes_by_color,
]

# ---------------------------------------------------------------------------
# APPROACH A — coordinate-only (commented out)
# Agent visually identifies every segment and passes coordinates directly.
# NOTE: GPT-4o cannot ground coordinates from an image — it fabricates round
# placeholder values — so this approach does not work in practice.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_COORDS_ONLY = (
    "You are a construction cost estimator. "
    "The user will give you a list of detected construction tasks and a PDF path. "
    "For each task, determine whether it is a **per-meter** task or a **per-unit** task, "
    "then calculate its cost accordingly:\n\n"

    "The task list may indicate which page each task appears on (e.g. 'Page 2'). "
    "Always pass that page number to the tools — both tools accept a 'page_number' argument "
    "(1-indexed, default 1). If no page is specified, use page 1.\n\n"

    "IF the task is measured in meters (e.g. wall demolition, wall construction, "
    "pipe installation, lighting profiles — tasks drawn as lines or shapes on the floor plan):\n"
    "  1. Read the relevant PDF page visually. Identify every segment that belongs to this task.\n"
    "  2. For each segment, estimate its start and end points as fractions of the page dimensions "
    "(x=0.0 is the left edge, x=1.0 is the right edge; y=0.0 is the top edge, y=1.0 is the "
    "bottom edge). Collect all segments as [[x1,y1,x2,y2], ...] pairs.\n"
    "  3. Call 'measure_total_length_by_coordinates' ONCE with all segments for this task. "
    "The tool sums them and returns the total in meters.\n"
    "  4. Call 'get_task_price' with the exact task name to get the unit price per meter.\n"
    "  5. Call 'multiply_numbers' to compute total cost = length × unit price.\n\n"

    "ELSE the task is per-unit (e.g. door demolition, kitchen demolition, bathroom renovation — "
    "tasks that appear as discrete countable items: doors, fixtures, rooms, or labeled elements):\n"
    "  1. If the items are drawn as colored outlines without fill (e.g. door arcs, window symbols), "
    "call 'count_outline_shapes_by_color' with the PDF path and the stroke color of those items. "
    "Sanity-check the returned sizes — door widths are typically 70–100 cm. "
    "If the items are rooms or labeled areas, read the PDF visually and count them instead.\n"
    "  2. Call 'get_task_price' with the exact task name to get the unit price.\n"
    "  3. Call 'multiply_numbers' to compute total cost = count × unit price.\n\n"

    "Report each task's quantity (meters or count), unit price, and total cost. "
    "Finish with an overall grand total.\n\n"
    "IMPORTANT: Do not write a final summary until you have called "
    "'get_task_price' and 'multiply_numbers' for every task. "
    "If tasks remain unpriced, your next output must be a tool call."
)

# TOOLS_COORDS_ONLY = [
#     get_task_price_tool,
#     multiply_numbers,
#     measure_total_length_by_coordinates,
#     count_outline_shapes_by_color,
# ]

# ---------------------------------------------------------------------------
# APPROACH B — both color-based and coordinate-based tools (commented out)
# Agent picks the right measurement approach per task.
# ---------------------------------------------------------------------------

# SYSTEM_PROMPT_BOTH_TOOLS = (
#     "You are a construction cost estimator. "
#     "The user will give you a list of detected construction tasks and a PDF path. "
#     "For each task, determine whether it is a **per-meter** task or a **per-unit** task, "
#     "then calculate its cost accordingly:\n\n"
#
#     "The task list may indicate which page each task appears on (e.g. 'Page 2'). "
#     "Always pass that page number to the tools — both tools accept a 'page_number' argument "
#     "(1-indexed, default 1). If no page is specified, use page 1.\n\n"
#
#     "IF the task is measured in meters:\n"
#     "  Choose ONE measurement approach and stick to it — do not use both for the same task:\n"
#     "  OPTION 1 — color-based (efficient for uniform solid/stroke elements): "
#     "Read the page and observe how the elements are drawn. "
#     "Call 'get_wall_lengths_by_color' with the color and drawing_type "
#     "('fill', 'stroke', or 'any'). If it returns 0 results, try a different drawing_type.\n"
#     "  OPTION 2 — coordinate-based (for complex/composite/partial elements): "
#     "Identify every segment visually, estimate start/end points as page-fraction coordinates "
#     "(0.0–1.0), and call 'measure_total_length_by_coordinates' ONCE with all segments.\n"
#     "  Then: call 'get_task_price' and 'multiply_numbers'.\n\n"
#
#     "ELSE the task is per-unit:\n"
#     "  1. If items are colored outlines without fill, call 'count_outline_shapes_by_color'.\n"
#     "     If items are rooms or labeled areas, count visually.\n"
#     "  2. Call 'get_task_price' and 'multiply_numbers'.\n\n"
#
#     "Report each task's quantity, unit price, and total cost. "
#     "Finish with an overall grand total.\n\n"
#     "IMPORTANT: Do not write a final summary until every task is priced. "
#     "If tasks remain unpriced, your next output must be a tool call."
# )
#
# TOOLS_BOTH = [
#     get_task_price_tool,
#     multiply_numbers,
#     get_wall_lengths_by_color,
#     count_outline_shapes_by_color,
#     measure_total_length_by_coordinates,
# ]

# ---------------------------------------------------------------------------

agent = AgentBuilder(
    model=model,
    tools=TOOLS_SELECT_IDS,
    system_prompt=SYSTEM_PROMPT_SELECT_IDS,
).with_memory().pdf_reader().tool_images().with_todos().build()

PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"

if __name__ == "__main__":
    question = PDF_PATH
    while question != "exit":
        answer = agent.run(question)
        print("Agent:", answer)
        question = input("You:")
