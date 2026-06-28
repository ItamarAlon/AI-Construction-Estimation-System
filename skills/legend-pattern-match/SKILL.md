---
name: legend-pattern-match
description: Use when a plan's legend defines each task by a GRAPHIC PATTERN SWATCH — a small sample drawing of a line style or symbol (e.g. a solid red line for "LED strip profile" vs. a red line with periodic dot/dash clusters for "magnetic profile") — rather than by a color name or a category word. Explains how to template each swatch and tag ONLY the map elements whose drawn pattern matches it, while never tagging text, dimensions, or callouts that merely share a color.
---

## Purpose

Some legends do not map a color (or a category word) to a task. Instead each legend
row shows a SAMPLE DRAWING — a swatch of the exact line style or symbol used on the
plan — next to the task name. The swatch IS the identity of the task. Two tasks can
share the same color and be told apart ONLY by their pattern.

This skill tells you to (1) build a precise visual template from each swatch, (2) tag
a map element ONLY when its actual drawn pattern matches a swatch, and (3) never tag
text, dimension lines, leaders, or callout boxes that happen to share the swatch
color but are not the pattern.

This is the STRICT counterpart to the `legend-color-task-group` skill:

* `legend-color-task-group` — legend maps a COLOR / category WORD to several tasks →
  be liberal, every element of that color belongs to the group.
* `legend-pattern-match` (this skill) — legend maps a graphic SWATCH to ONE specific
  task → be strict, only elements that reproduce the swatch pattern count; same-color
  noise is excluded.

## When to Use

Use this skill the moment a legend row contains a small SAMPLE DRAWING of a line/
symbol (not just a color chip with a word). Signs:

* The legend shows a drawn line segment or symbol next to each task name, e.g. a
  header like "מקרא תאורה" (lighting legend) listing fixture symbols and profile
  line styles.
* Two or more rows share the SAME color but show DIFFERENT line patterns (solid vs.
  dotted vs. dot-clustered), so color alone cannot distinguish them.

If the legend just maps a color/word to tasks with no sample drawing, use
`legend-color-task-group` instead.

## Instructions

### Step 1 — Template each swatch precisely

For every legend row, look at the sample drawing and write a precise description of
its pattern — enough to recognise it in a small crop. Capture:

* line vs. symbol; if a line: solid / dashed / dotted / line-with-periodic-marks;
* the color;
* any repeating decoration (dot clusters, ticks, an embedded symbol).

Example template table:

    swatch                              | color | task
    solid thick line                    | red   | install LED strip lighting profile (per meter)
    line with periodic dot/dash clusters| red   | install magnetic light profile (per meter)
    circle + cross (ceiling fixture)    | red   | (fixture — per-unit, if a task exists; else ignore)

### Step 2 — Match map elements to a swatch by PATTERN, not color

When classifying each segment, compare its ACTUAL drawn pattern (from its crop, or
its spot on the full-page image) to your swatch templates. Tag it to a task ONLY when
its pattern reproduces that task's swatch. A long solid red run → the solid-line task;
a red run carrying periodic dot clusters → the dot-cluster task — even though both are
red.

### Step 3 — Exclude same-color noise

A swatch color is also used for things that are NOT the task. Do NOT tag a segment
just because it is the swatch color. In particular, NEVER tag as a task:

* text, labels, or Hebrew/English note strings;
* callout / note boxes (a rectangle drawn around explanatory text);
* dimension lines, extension lines, witness lines, leader lines, arrowheads;
* title-block or legend graphics themselves.

If a segment is one of these, tag it `ignore` regardless of its color.

## Rules

* PATTERN decides the task, not color. Same-color rows are distinguished only by their
  drawn pattern.
* Tag ONLY elements that reproduce a swatch pattern. If a segment's pattern matches no
  swatch, ignore it — never force same-color noise onto the nearest task.
* Text, dimensions, leaders, and callout boxes are NEVER tasks, even in the swatch
  color. This is the most common mistake on pattern legends — exclude them explicitly.
* A per-meter profile task is a continuous LINE RUN that traces a route on the plan.
  A short colored stroke sitting next to text, or a box around text, is not a profile
  run — ignore it.

## Examples

### Example: lighting legend with two same-color profiles

Legend "מקרא תאורה" lists, among point-fixture symbols, two line-style rows, both red:

* `solid thick red line` → `פרופיל תאורה פס לד` = install LED strip lighting profile
* `red line with periodic dot/dash clusters` → `פרופיל תאורה מגנטי` = install magnetic
  light profile

On the plan:

* A long red run with no decoration tracing the ceiling → LED strip profile.
* A long red run carrying periodic dot clusters → magnetic profile.
* A blue (or red) rectangle around a Hebrew note like "הוצאת שקעים מרצפה…" → `ignore`
  (callout text, not a profile).
* Red dimension/extension lines with arrowheads → `ignore`.
