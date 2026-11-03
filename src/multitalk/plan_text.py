"""Text handling for plans and clarifying questions."""

from __future__ import annotations

import re


def count_questions(planner_output: str) -> int:
    cleaned = re.sub(r'"""', "", planner_output)
    return len(re.findall(r"\n\s*(\d+:)", cleaned))


def format_clarifications(planner_questions: str, answers: str, question_count: int) -> tuple[str, int]:
    """Pair questions with answers for the Analyzer. Return the text and question count."""
    normalized = re.sub(r"Questions:\s*", "Questions:\n", planner_questions)
    questions = re.findall(r"\d+:\s*(.*?)\s*(?=\d+:|$)", normalized, re.DOTALL)
    questions = [question.strip() for question in questions]
    answers_list = answers.split("\n")[1:]
    lines = []
    for i, question in enumerate(questions):
        lines.append(f"{question_count + i + 1}: {question}")
        lines.append(f"{question_count + i + 1}: {answers_list[i].split(':')[-1].strip()}")
    return "\n".join(lines) + "\n", len(questions)


def to_simulator_format(plan: str) -> str:
    """Rewrite a plan so that each move also names the grasp location it starts from."""
    output = ["Step-by-step Instructions:"]
    grasp_point = ""
    for line in plan.strip().split("\n"):
        if "#" in line:
            line = line.split("#")[0].strip()
        if "grasp" in line:
            grasp_point = line.split("#")[0].strip().split("grasp")[-1]
            output.append(f"{line.split(':')[0].strip()}: grasp({grasp_point[1:-1]})")
        elif "move" in line:
            move_point = line.split("#")[0].strip().split("move")[-1]
            output.append(f"{line.split(':')[0].strip()}: move({grasp_point[1:-1]}, {move_point[1:-1]})")
        elif "home" in line:
            output.append(f"{line.split(':')[0].strip()}: home()")
    return "\n".join(output)


_ACTION = re.compile(r"\b(grasp|move)\s*\((.*)\)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def check_bounds(plan: str, x_bounds, y_bounds) -> str:
    """Return one line per grasp or move whose (x, y) is off the table, or "" if none is.

    Steps without two numeric coordinates are left to the Analyzer and Simulator.
    """
    violations = []
    for line in plan.split("\n"):
        line = line.split("#")[0].strip()
        match = _ACTION.search(line)
        if not match:
            continue
        coordinates = _NUMBER.findall(match.group(2))
        if len(coordinates) < 2:
            continue
        x, y = float(coordinates[0]), float(coordinates[1])
        if not (x_bounds[0] < x < x_bounds[1] and y_bounds[0] < y < y_bounds[1]):
            violations.append(
                f"{line} # out of bounds: ({x}, {y}) is off the table, which spans "
                f"{x_bounds[0]} < x < {x_bounds[1]} and {y_bounds[0]} < y < {y_bounds[1]}"
            )
    return "\n".join(violations)
