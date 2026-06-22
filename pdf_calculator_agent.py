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


# Vision model: gpt-4o reads small Hebrew crop labels unreliably (non-deterministic
# OCR run-to-run), which caused tasks like Kitchen Removal to be missed. Claude reads
# the same crops reliably, so the classification agent uses Claude via OpenRouter.
# (Other Claude slugs available on this account: claude-sonnet-4 / -4.5, claude-opus-4.x.)
model = ChatOpenAI(
    model="anthropic/claude-sonnet-4.6",
    temperature=0.2,
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key()
)

SYSTEM_PROMPT_SELECT_IDS = (
    "You are a construction cost estimator. "
    "The user message contains: The PDF Plan, the DETECTED COLOR PALETTE (exact hex "
    "codes per page), and the PRE-COMPUTED SEGMENT LISTINGS for every palette color — "
    "attribute lines, optionally followed by a zoomed crop image per segment (crops may or "
    "may not be included). "
    "Use the exact hex codes from the palette in your JSON output.\n\n"

    "--- PHASE 1: DETECT TASKS ---\n"
    "  1. Call 'get_available_tasks' to get the list of known task names (the menu).\n"
    "  2. Read the PDF plan. FIRST, look for a LEGEND / KEY (usually a boxed list "
    "on the side or corner of a page that maps each color or symbol to its meaning). Many "
    "plans have none -- that is fine, just move on. If one EXISTS, it is the authoritative "
    "source for what each color/symbol represents. "
    "TRANSCRIBE it into an explicit table:\n"
    "     <short symbol description> | <color> | <what it means> | <matching task or 'none'>\n"
    #Might Overfit:
    "   e.g.  'solid line | yellow | wall demolition | Wall Demolition (per meter)'\n"
    "         'arc + short line | yellow | door | Door Demolition'\n"
    #   
    "   IMPORTANT: many legends distinguish items by SYMBOL SHAPE while using the SAME color. "
    "Describe each symbol carefully enough to recognise it later in a small crop or on the plan. "
    #Might Overfit:
    "   Also, many legends simply assign a specific color to multiple tasks of the same kind. "
    "For example red can be assigned for construction - meaning red items are for construction. "
    "(whether it's door construction/wall construction/window construction depends on it's appearance on the plan itself)."
    "In that case, don't immediately assume that it's a different task just because the color pattern is not the exact same as in the legend.\n"
    #
    "  3. Decide which available tasks are actually present (from legend, explicit color "
    "labels, or context). Output each detected task with its EXACT name, page, and how to "
    "identify it. If none are present, say so and stop."
    "Text Labels written in the PDF are also a good indication for a task appearing. Read those as well."
    "Don't invent new tasks. ONLY use the tasks given from 'get_available_tasks'.\n\n"

    "--- PHASE 2: CLASSIFY ---\n"
    "The segment listings are already in this message (sections headed '=== #hex, page N ==='). "
    "Output a classification table covering EVERY segment — no segment may be skipped. "
    "The table has THREE columns and you MUST fill all three for every segment:\n\n"
    "  ID <ns>-<i>  |  <exact text visible at THIS segment, transcribed verbatim, or '(none)'>  |  <task name or 'ignore'>\n\n"
    "Each segment has a text attribute line giving NEUTRAL GEOMETRY only (length, orientation, "
    "solid-fill/thin-stroke, straight/curved, center (x%,y%), clusterxN). "
    "EACH SEGMENT'S TEXT LINE MAY BE FOLLOWED BY ITS OWN ZOOMED CROP IMAGE. If crops are present, "
    "use the crop to see the segment and read any label. If NO crops are present, find the segment "
    "on the full-page plan image using its center (x%,y%) and read any label next to it there.\n"
    "  - TRANSCRIBE FIRST, CLASSIFY SECOND: for EVERY segment, look at it (its crop if one is "
    "given, otherwise its spot on the full-page plan) and write down the exact label text you "
    "actually see there (in the original language) BEFORE you pick a task. Do NOT copy the label "
    "from a previous segment and do NOT guess it from the shape — read the actual pixels at this "
    "specific segment. If you see no text, write '(none)'.\n"
    "  - Then pick the task FROM THAT TRANSCRIBED TEXT. Two segments of the same color and shape "
    "can carry different labels (e.g. one says 'wall', another says 'kitchen') — the label, not "
    "the shape, decides the task. If the transcribed text names a different element than nearby "
    "segments, it is a DIFFERENT task; never assume it is the same just because it looks similar.\n"
    "  - MATCH AGAINST THE FULL TASK MENU, NOT JUST PHASE-1: a transcribed label maps to ANY task "
    "from 'get_available_tasks' whose meaning it matches — even if you did NOT list that task in "
    "Phase 1. Phase 1 is only a first skim and routinely misses tasks; the per-segment label is "
    "the authoritative signal. Translate the label if needed (e.g. 'מטבח'/'kitchen' -> a Kitchen "
    "task; 'חלון'/'window' -> a Window task) and tag the segment to that available task. NEVER "
    "tag a segment 'ignore' when its transcribed label matches an available task just because the "
    "task was absent from your Phase-1 list — add the task instead.\n"
    "  - Never ignore a segment based on geometry alone — if it has a task-related label, tag it.\n"
    "  - LEGEND MATCHING: compare each segment's shape (from its crop if given, else from the "
    "full-page plan image at its center) to your Phase-1 legend table. When a color is used for "
    "multiple tasks (e.g. yellow = walls AND doors), the shape is the ONLY way to tell them apart. "
    "Assign the segment to the task whose legend symbol matches.\n"
    
    # "  - ROOM / AREA OUTLINES: a rectangle or L-shape surrounding a labeled room or area "                                                                                   
    # "(e.g. a box around text that says 'kitchen', 'bathroom') is a per-unit task "                                                                                                
    # "marker — tag it to the matching task (Kitchen Demolition, Kitchen Removal, etc.). "                                                                                     
    # "Do NOT call it a dimension line. Dimension lines are SHORT lines with arrowheads and a "
    # "nearby NUMBER;\n"

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

agent = AgentBuilder(
    model=model,
    tools=TOOLS_SELECT_IDS,
    system_prompt=SYSTEM_PROMPT_SELECT_IDS,
).pdf_reader().tool_images().with_memory().with_todos().build()

PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"

if __name__ == "__main__":
    question = PDF_PATH
    while question != "exit":
        answer = agent.run(question)
        print("Agent:", answer)
        question = input("You:")
