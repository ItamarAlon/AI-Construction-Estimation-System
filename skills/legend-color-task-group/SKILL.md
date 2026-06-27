---
name: legend-color-task-group
description: Use when a plan's legend maps a color (or symbol) to a broad work CATEGORY — a general word like "building"/"בנייה", "demolition"/"הריסה ופירוק", "plumbing"/"אינסטלציה" — instead of one specific task. Explains that such a color stands for a GROUP of related tasks, and how to pick the correct task per segment when classifying.
---

## Purpose

A legend rarely maps one color to one task. Often a single color is labeled with a
broad CATEGORY of work — e.g. "בנייה" (building/construction) or "הריסה ופירוק"
(demolition) — and every element drawn in that color belongs to that category, but to
a DIFFERENT specific task within it.

This skill tells you how to treat such a legend entry: map the color to the whole
GROUP of available tasks that fall under that category, then decide the exact task
for each individual segment later, when you classify it.

## When to Use

Load this skill in Phase 1, the moment a legend entry maps a color/symbol to a
general category word rather than to a single concrete task. Signs:

* The legend text is a broad heading like "building", "construction", "demolition",
  "plumbing", "electrical", "renovation" — not a precise item like "interior door".
* The AVAILABLE TASKS menu contains MORE THAN ONE task that plausibly falls under
  that category (e.g. "Wall Demolition" AND "Door Demolition" both under demolition).

If a legend color maps cleanly to exactly one task, you do not need this skill.

## Instructions

### Step 1 — Recognize a category legend entry

Read the legend label. If it names a general kind of work (not one specific
element), treat it as a CATEGORY, not a single task.

### Step 2 — Map the color to a GROUP of tasks

Scan the AVAILABLE TASKS menu and collect EVERY task that belongs to that category.
Record the color as mapping to that whole group. For example:

* `green | "הריסה ופירוק" (demolition)` → group = { Wall Demolition (per meter),
  Door Demolition }
* `pink | "בנייה" (building)` → group = { Building Wall (per meter),
  Wall thickening (per meter) }

Note this grouping in your Phase-1 output, e.g.:

    orange (#ff7f00) | demolition category | tasks: Wall Demolition, Door Demolition

### Step 3 — Pick the specific task per segment (during classification)

When you later classify each segment of that color, you already know it belongs to
the group. Now choose WHICH task in the group it is, using:

1. Its own text label, if any (the strongest signal).
2. Otherwise its shape/symbol: a long bar = a wall task; an arc + short jamb line =
   a door task; a fixture symbol = a fixture task; etc.

So a long orange bar → Wall Demolition; an orange door arc → Door Demolition — both
correct under the same "demolition" legend color.

## Rules

* A category color does NOT mean every segment is the SAME task — segments of one
  category color split across the several tasks in its group.
* Only pick tasks that are actually in the AVAILABLE TASKS menu. Never invent a task
  just because it fits the category.
* If a segment of a category color matches none of the group's tasks (and has no
  other matching label), ignore it — do not force it onto the nearest group task.
* A specific (non-category) legend entry overrides the group: if one color has both a
  broad category meaning and a precise label on a particular segment, the precise
  label wins for that segment.

## Examples

### Example: demolition category

Legend: `orange = "הריסה ופירוק קירות ודלתות"` (demolition of walls and doors).
Available tasks include "Wall Demolition (per meter)" and "Door Demolition".

→ Map orange to the group { Wall Demolition, Door Demolition }.
→ Long orange bars → Wall Demolition. Orange door arcs → Door Demolition.

### Example: building category

Legend: `blue = "בנייה"` (building). Available tasks include "Building Wall
(per meter)" and "Wall thickening (per meter)".

→ Map blue to the group { Building Wall, Wall thickening }.
→ Plain new wall segments → Building Wall. Segments whose label says "עיבוי קיר"
  (wall thickening) → Wall thickening.
