from pathlib import Path
import sys

from langchain.tools import tool
from construction_tasks_prices.read_construction_tasks_prices import (
    get_task_price_tool,
    get_available_tasks_tool,
)
from wall_measurement_tool import (
    get_wall_lengths_by_color,
    count_outline_shapes_by_color,
    measure_total_length_by_coordinates,
    list_colored_segments,
    measure_segments_by_id,
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
    "The user will give you a PDF path (the plan images are included in the message). "
    "You must FIRST detect which construction tasks appear on the plan, then cost each one.\n\n"

    "--- PHASE 0: DETECT TASKS ---\n"
    "  1. Call 'get_available_tasks' to get the list of known task names (the menu).\n"
    "  2. Read the PDF plan images. Decide which of those tasks are actually present. "
    "A task can be explicit (labeled/colored on the plan) or inferred from context "
    "(e.g. a kitchen layout implies 'Kitchen Demolition'; door symbols imply 'Door Demolition').\n"
    "  3. Output the detected tasks, each with: the EXACT task name from the menu (including any "
    "'(per meter)' suffix), the page it appears on, and how to find it on the plan (color, symbol, "
    "or location). Only include tasks you actually found. If none are present, say so and stop.\n"
    "  Then cost each detected task using the phases below.\n\n"

    "A task name containing '(per meter)' is a per-meter task (use Phases 1-3). "
    "Everything else is a per-unit task (use the PER-UNIT section). "
    "Always pass the task's page number to the tools (1-indexed, default 1 if unsure).\n\n"

    "--- PHASE 1: ENUMERATE ---\n"
    "For every per-meter task on a page, call 'list_colored_segments' with its color "
    "and page number. If two tasks share the same color and page, one call covers both.\n\n"

    "--- PHASE 2: CLASSIFY (required before any measurement call) ---\n"
    "After receiving the segment listing, output a classification table that covers "
    "EVERY segment — no segment may be skipped. One line per segment:\n\n"
    "  ID <ns>-<i>  ->  <task name or 'ignore'>  [<one-line reason>]\n\n"
    "Example:\n"
    "  ID R2-0   ->  Wall Demolition  [vertical, on-wall 100% -- traces a real wall edge]\n"
    "  ID R2-11  ->  ignore           [duplicate of R2-0: same length & center -- 2nd edge of same wall]\n"
    "  ID R2-1   ->  ignore           [on-wall 10% -- floats off the walls, dimension line]\n\n"
    "How to read the attributes (they describe the vector geometry, not the task):\n"
    "  - 'on-wall N%' is the STRONGEST signal: the share of the segment running along the "
    "building's black walls. HIGH (>=60%) = it traces a real wall edge -> assign to the task. "
    "LOW (<25%) on a long segment = it floats in open space -> dimension/leader line -> ignore.\n"
    "  - Orientation: 'horizontal'/'vertical' is an elongated run (typical wall edge). "
    "'square (arc/symbol)' that is SMALL (~50-110 cm) is a door/window symbol -> ignore (it may "
    "be a separate per-unit task, not wall length). A LARGE 'square' (>150 cm) is usually a "
    "corner or diagonal wall run, not a symbol -- judge it by on-wall and the PDF.\n"
    "  - FILLED vs OUTLINE is only a WEAK hint. Walls may be drawn either way depending on the "
    "plan (some plans draw every wall as an OUTLINE), so do NOT reject a segment just because "
    "it is OUTLINE.\n"
    "  - DUPLICATES: a wall drawn as a double line surfaces as two near-identical segments. "
    "The listing flags the extra one as 'dup-of <id>'. Assign only the referenced <id> to the "
    "task and tag every 'dup-of' segment 'ignore (duplicate of <id>)' -- counting both DOUBLES "
    "the length. (The measurement tool also collapses such groups as a safety net, but your "
    "table must still tag them correctly.)\n"
    "  - If several tasks share the color, use each segment's position and the PDF to decide "
    "which task it belongs to.\n"
    "  - Genuinely uncertain -> tag 'ignore (uncertain)' with a reason; do not guess.\n\n"
    "Do not call 'measure_segments_by_id' until the classification table is complete.\n\n"

    "--- PHASE 3: MEASURE ---\n"
    "For each per-meter task that has segments assigned to it, call 'measure_segments_by_id' "
    "with (color, page, IDs tagged to that task) to get its total length in meters.\n\n"

    "--- PER-UNIT TASKS (doors, fixtures, rooms -- discrete countable items) ---\n"
    "These bypass Phases 1-3:\n"
    "  If items are colored outlines without fill (door arcs, window symbols), "
    "call 'count_outline_shapes_by_color'. Sanity-check sizes -- door widths are 70-100 cm. "
    "If items are rooms or labeled areas, read the PDF and count visually.\n\n"

    "--- FINAL OUTPUT (quantities only -- do NOT price anything) ---\n"
    "You do NOT have access to prices and you must NOT compute any cost. A separate program "
    "prices the quantities you report. After detecting and measuring every task:\n"
    "  1. Restate the Phase 0 detected-task list and the Phase 2 classification table(s).\n"
    "  2. End your message with a single JSON object mapping each EXACT task name (as returned "
    "by 'get_available_tasks', including any '(per meter)' suffix) to its measured quantity "
    "(meters for per-meter tasks, integer count for per-unit tasks). Put it in a ```json code "
    "block as the LAST thing in your message. Example:\n"
    "```json\n"
    "{\"Wall Demolition (per meter)\": 3.58, \"Door Demolition\": 8}\n"
    "```\n"
    "Only include tasks you actually found. Use numbers, not strings. Report the quantity once "
    "per task -- do not double-count duplicate ('dup-of') segments."
)

TOOLS_SELECT_IDS = [
    get_available_tasks_tool,
    list_colored_segments,
    measure_segments_by_id,
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
).with_memory().pdf_reader().with_todos().build()

PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"

if __name__ == "__main__":
    question = PDF_PATH
    while question != "exit":
        answer = agent.run(question)
        print("Agent:", answer)
        question = input("You:")
