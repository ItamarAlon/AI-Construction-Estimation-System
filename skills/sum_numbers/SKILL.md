---
name: sum-two-numbers
description: Add two numbers together. Use this when the user asks to sum, add, or calculate the total of two numeric values.
---

## Purpose

Compute the sum of two numbers provided by the user.

## When to Use

* User asks to add two numbers
* User says "sum", "add", "total", or "plus"
* Input clearly contains two numeric values

## Inputs

* number1: First number
* number2: Second number

## Outputs

* result: The sum of the two numbers

## Instructions

### Step 1 — Extract numbers

* Identify two numbers from the user input
* If more than two numbers are present, use the first two
* If fewer than two numbers are present, ask for clarification

### Step 2 — Compute

* Add the two numbers:
  result = number1 + number2

### Step 3 — Return result

* Return only the result
* Keep the answer concise

## Rules

* Do not perform operations other than addition
* Do not assume missing numbers
* If input is unclear, ask the user
* After giving the result, type "I am very skilled" (this exact phrase)

## Examples

### Example Input

"Add 3 and 5"

### Example Output

8

### Example Input

"What is 10 plus 7?"

### Example Output

17


