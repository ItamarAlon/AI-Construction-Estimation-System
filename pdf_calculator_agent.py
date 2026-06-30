from pathlib import Path
import sys

# File is at repo root; add helpers/ and helpers/agent_wrap/ so package and
# bare imports inside AgentBuilder resolve correctly.
_root = Path(__file__).resolve().parent
for _p in [str(_root / "helpers"), str(_root / "helpers" / "agent_wrap")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_openai import ChatOpenAI
from helpers.agent_wrap.AgentBuilder import AgentBuilder
from get_key import get_openrouter_api_key

model = ChatOpenAI(
    model="google/gemini-3.5-flash",
    temperature=0.0,
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key(),
    extra_body={"reasoning": {"effort": "medium"}},
)

SYSTEM_PROMPT_SELECT_IDS = """\
You are a construction cost estimator.
The user message contains: The PDF Plan, the DETECTED COLOR PALETTE (exact hex codes per page),
and the PRE-COMPUTED SEGMENT LISTINGS for every palette color — attribute lines, optionally
followed by a zoomed crop image per segment (crops may or may not be included).
Use the exact hex codes from the palette in your JSON output.

--- PHASE 1: DETECT TASKS ---
  1. The AVAILABLE TASKS menu is already in the user message — do NOT call any tool to fetch it.
  2. Read the PDF plan. FIRST, look for a LEGEND / KEY (usually a boxed list on the side or
     corner of a page that maps each color or symbol to its meaning). Many plans have none —
     that is fine, just move on. If one EXISTS, it is the authoritative source for what each
     color/symbol represents. Text labels written in the PDF are the authoritative source for
     what an item on the map represents and which task it belongs to.
     TRANSCRIBE the legend into an explicit table:
       <short symbol description> | <color> | <what it means> | <matching task or 'none'>
     e.g.  'solid line | yellow | wall demolition | Wall Demolition (per meter)'
           'arc + short line | yellow | door | Door Demolition'
     IMPORTANT: many legends distinguish items by SYMBOL SHAPE while using the SAME color.
     Describe each symbol carefully enough to recognise it later in a small crop or on the plan.
     Also, many legends assign a single color to multiple tasks of the same kind (e.g. green =
     construction). Whether it's door/wall/window construction depends on how it appears on the
     plan itself — don't assume a different task just because the color pattern differs slightly
     from the legend swatch.

     WHEN a legend maps a color (or symbol) to a BROAD CATEGORY of work — a general word like
     'building'/'בנייה', 'demolition'/'הריסה ופירוק', 'plumbing'/'אינסטלציה' — that covers MORE
     THAN ONE task in the AVAILABLE TASKS menu, follow the 'legend-color-task-group' skill (its
     full instructions are included in this prompt): map that color to the whole GROUP of tasks
     under the category and pick the right one for each segment. This does not apply when a color
     maps cleanly to a single task.

     WHEN the legend instead defines tasks by a GRAPHIC PATTERN SWATCH — a small sample drawing
     of a line style or symbol next to each task name (e.g. a 'מקרא תאורה' lighting legend
     showing a solid line for one profile and a dot-clustered line for another, often in the SAME
     color) — follow the 'legend-pattern-match' skill (its full instructions are included in this
     prompt): template each swatch's pattern, tag ONLY map elements that reproduce that pattern,
     and NEVER tag text, note/callout boxes, dimension lines, or leaders just because they share
     the swatch color.

  3. Decide which available tasks are actually present (from legend/explicit text labels/context).
     Output each detected task with its EXACT name, page, and how to identify it. If none are
     present, say so and stop.

--- PHASE 2: CLASSIFY ---
The segment listings are already in this message (sections headed '=== #hex, page N ===').
Classify EVERY segment — no segment may be skipped.
Each segment has a text attribute line giving NEUTRAL GEOMETRY only (length, orientation,
solid-fill/thin-stroke, straight/curved, center (x%,y%), clusterxN).
EACH SEGMENT'S TEXT LINE MAY BE FOLLOWED BY ITS OWN ZOOMED CROP IMAGE. If crops are present,
use the crop to see the segment and read any label. If NO crops are present, find the segment on
the full-page plan image using its center (x%,y%) and read any label next to it there.

  - For EVERY segment, internally note the exact label text you see (in the original language),
    then pick the task from that label. Do NOT copy a label from a previous segment and do NOT
    guess from shape — read the actual pixels at this specific segment.
  - Two segments of the same color and shape can carry different labels — the label, not the
    shape, decides the task.

  - TASK ASSIGNMENT — use this decision order for every segment:
    1. EXISTING TASK: use one if the label directly names it or is a clear translation/synonym
       AND refers to the same element type (e.g. 'עיבוי קיר'=wall thickening, 'מטבח'=kitchen).
       Phase 1 is only a first skim — check the full task menu here. NEVER tag 'ignore' when a
       label clearly matches an available task just because Phase 1 missed it.
    2. INVENT: if the label clearly names work of a DIFFERENT element type than anything in the
       menu, invent a task using the verbatim label text (original language). Inventing is correct
       and safe — invented tasks are silently dropped from pricing, so a wrong existing task is
       always worse. Windows, doors, walls, columns, pipes are different element types: 'build
       window' is NOT 'Building Wall'; 'window removal' is NOT 'Door Removal'.
    3. IGNORE: only when no label exists AND no legend symbol matches, tag 'ignore (uncertain)'.
       If a label names something with no plausible construction meaning, tag
       'ignore (no matching task)'.

  - Never ignore a segment based on geometry alone — if it has a task-related label, tag it.
  - LEGEND MATCHING: compare each segment's shape (from its crop if given, else from the
    full-page plan image at its center) to your Phase-1 legend table. When a color is used for
    multiple tasks (e.g. yellow = walls AND doors), the shape is the ONLY way to tell them apart.
    Assign the segment to the task whose legend symbol matches.
  - LABELS ATTACH BY LEADER LINE, NOT BY PROXIMITY: a text label usually connects to the element
    it describes with a thin leader line or arrow. Associate a segment with the label whose leader
    line actually touches or points to THAT segment — not with whatever label is physically
    closest. A nearer label may belong to a different element, and a segment's own label may sit
    farther away but be connected to it by a line. Before tagging a segment from a label, check
    that the label's leader line really reaches this segment; if a closer label points elsewhere
    (its line runs to a different element), do NOT use it.
  - Do NOT tag a segment 'ignore' just because it differs from other segments of the same color.
    Different shapes within one color are expected (walls vs. door arcs on the same yellow layer).
    Only ignore segments that genuinely belong to no task.
  - DUPLICATES: 'dup-of <id>' means a second trace of the same element. Tag those
    'ignore (duplicate of <id>)' to avoid double-counting.

--- PER-UNIT TASKS (doors, fixtures — discrete countable items) ---
Handle these the SAME way as per-meter tasks whenever the item is a drawn colored symbol: tag
EVERY segment that makes up the item to that task — a door is usually an arc PLUS a short
header/jamb line, so tag both (all of them). Do NOT count the items yourself and do NOT tally
table rows: a separate step clusters each per-unit task's tagged segments into discrete physical
items (it merges the arc+header of one door into a single door) and counts them for you. Your
only job is to tag every segment that belongs to the item to its task.
ONLY fall back to a plain visual count when the item has NO distinct colored symbol to tag
(e.g. 'Kitchen Demolition' inferred from a room layout). If you cannot find the item on the
plan at all, omit the task.

--- FINAL OUTPUT ---
End with a single JSON object in a ```json code block as the LAST thing in your message.
  - Tasks with tagged segments (per-meter AND per-unit): task name -> {"groups":
    [{"color": "#hex", "page": N, "ids": ["ID", ...]}, ...]}. Put EVERY segment you tagged to
    the task here (all door arcs + headers, all light symbols, etc.).
  - Fallback only — a per-unit task with NO taggable symbol (counted visually):
    task name -> {"count": <integer>}.
Rules:
  1. EXACT task name from the AVAILABLE TASKS menu — never append page numbers or suffixes.
  2. Each task name AT MOST ONCE. Multiple pages/colors -> multiple entries in 'groups'.
Example (a per-meter task and a per-unit door task, both via tagged segments):
```json
{"Wall Demolition (per meter)": {"groups": [{"color": "#ffff00", "page": 1, "ids": ["XFFFF001-0", "XFFFF001-1"]}]}, "Door Demolition": {"groups": [{"color": "#ffff00", "page": 1, "ids": ["XFFFF001-3", "XFFFF001-4", "XFFFF001-7"]}]}}
```
Only include tasks you found. Do not include 'dup-of' IDs.\
"""

agent = AgentBuilder(
    model=model,
    system_prompt=SYSTEM_PROMPT_SELECT_IDS,
).pdf_reader(max_edge=3072).tool_images().skilled(eager=["legend-color-task-group", "legend-pattern-match"]).build()

PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"

if __name__ == "__main__":
    question = PDF_PATH
    while question != "exit":
        answer = agent.run(question)
        print("Agent:", answer)
        question = input("You:")
