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
    "The user message contains: the PDF plan images, the DETECTED COLOR PALETTE (exact hex "
    "codes per page), and the PRE-COMPUTED SEGMENT LISTINGS for every palette color — "
    "attribute lines plus zoomed crop images, one crop per segment. "
    "You do NOT need to call 'list_colored_segments'; all segment data is already in this "
    "message. Use the exact hex codes from the palette in your JSON output.\n\n"

    "--- PHASE 0: DETECT TASKS ---\n"
    "  1. Call 'get_available_tasks' to get the list of known task names (the menu).\n"
    "  2. Read the PDF plan images. FIRST, look for a LEGEND / KEY (usually a boxed list "
    "on the side or corner of a page that maps each color or symbol to its meaning). Many "
    "plans have none -- that is fine, just move on. If one EXISTS, it is the authoritative "
    "source for what each color/symbol represents. TRANSCRIBE it into an explicit table:\n"
    "     <short symbol description> | <color> | <what it means> | <matching task or 'none'>\n"
    "   e.g.  'solid line | yellow | wall demolition | Wall Demolition (per meter)'\n"
    "         'arc + short line | yellow | door | Door Demolition'\n"
    "   IMPORTANT: many legends distinguish items by SYMBOL SHAPE while using the SAME color. "
    "Describe each symbol carefully enough to recognise it later in a small image crop.\n"
    "  3. Decide which available tasks are actually present (from legend, explicit color "
    "labels, or context). Output each detected task with its EXACT name, page, and how to "
    "identify it. If none are present, say so and stop.\n\n"

    "--- PHASE 2: CLASSIFY ---\n"
    "The segment listings are already in this message (sections headed '=== #hex, page N ==='). "
    "Output a classification table covering EVERY segment — no segment may be skipped:\n\n"
    "  ID <ns>-<i>  ->  <task name or 'ignore'>  [<one-line reason>]\n\n"
    "Each segment has a text attribute line followed immediately by its own zoomed crop image. "
    "The attributes give NEUTRAL GEOMETRY only (length, orientation, solid-fill/thin-stroke, "
    "straight/curved, center, clusterxN). The crop shows the actual spot on the plan.\n"
    "  - Look at EVERY crop. TEXT LABELS IN THE CROP ARE THE STRONGEST SIGNAL.\n"
    "  - LEGEND MATCHING: compare each crop's shape to your Phase-0 legend table. When a color "
    "is used for multiple tasks (e.g. yellow = walls AND doors), the shape in the crop is the "
    "ONLY way to tell them apart. Assign the segment to the task whose legend symbol matches.\n"
    "  - Do NOT tag a segment 'ignore' just because it differs from other segments of the same "
    "color. Different shapes within one color are expected (walls vs. door arcs on the same "
    "yellow layer). Only ignore segments that genuinely belong to no task.\n"
    "  - DUPLICATES: 'dup-of <id>' means a second trace of the same element. Tag those "
    "'ignore (duplicate of <id>)' to avoid double-counting.\n"
    "  - Genuinely uncertain -> 'ignore (uncertain)' with a reason.\n\n"
    "Do not output the final JSON until the classification table is complete. "
    "Return it in the output too (for logging).\n\n"

    "--- PER-UNIT TASKS (doors, fixtures -- discrete countable items) ---\n"
    "Handle these the SAME way as per-meter tasks whenever the item is a drawn colored symbol: "
    "tag EVERY segment that makes up the item to that task in the classification table — a door "
    "is usually an arc PLUS a short header/jamb line, so tag both (all of them). Do NOT count "
    "the items yourself and do NOT tally table rows: a separate step clusters each per-unit "
    "task's tagged segments into discrete physical items (it merges the arc+header of one door "
    "into a single door) and counts them for you. Your only job is to tag every segment that "
    "belongs to the item to its task.\n"
    "ONLY fall back to a plain visual count when the item has NO distinct colored symbol to tag "
    "(e.g. 'Kitchen Demolition' inferred from a room layout). If you cannot find the item on the "
    "plan at all, omit the task.\n\n"

    "--- FINAL OUTPUT ---\n"
    "End with a single JSON object in a ```json code block as the LAST thing in your message.\n"
    "  - Tasks with tagged segments (per-meter AND per-unit): task name -> {\"groups\": "
    "[{\"color\": \"#hex\", \"page\": N, \"ids\": [\"ID\", ...]}, ...]}. Put EVERY segment you "
    "tagged to the task here (all door arcs + headers, all light symbols, etc.).\n"
    "  - Fallback only — a per-unit task with NO taggable symbol (counted visually): "
    "task name -> {\"count\": <integer>}.\n"
    "Rules:\n"
    "  1. EXACT task name from 'get_available_tasks' — never append page numbers or suffixes.\n"
    "  2. Each task name AT MOST ONCE. Multiple pages/colors -> multiple entries in 'groups'.\n"
    "Example (a per-meter task and a per-unit door task, both via tagged segments):\n"
    "```json\n"
    "{\"Wall Demolition (per meter)\": {\"groups\": ["
    "{\"color\": \"#ffff00\", \"page\": 1, \"ids\": [\"XFFFF001-0\", \"XFFFF001-1\"]}]}, "
    "\"Door Demolition\": {\"groups\": [{\"color\": \"#ffff00\", \"page\": 1, "
    "\"ids\": [\"XFFFF001-3\", \"XFFFF001-4\", \"XFFFF001-7\"]}]}}\n"
    "```\n"
    "Only include tasks you found. Do not include 'dup-of' IDs."
)

TOOLS_SELECT_IDS = [
    get_available_tasks_tool,
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
