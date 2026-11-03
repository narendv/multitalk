"""The MultiTalk planning loop (Algorithm 1 in the paper)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .config import Config
from .llm.agents import Analyzer, Planner
from .llm.client import ChatClient, Usage
from .plan_text import check_bounds, count_questions, format_clarifications, to_simulator_format
from .types import PipelineResult, Scene


@dataclass
class Interfaces:
    """The pipeline's connections to the world outside the LLMs."""

    ask_human: Callable[[int], str]  # Takes the question's 1-based number.
    rescan: Callable[[], Scene]
    simulate: Callable[[str, Scene], str]
    log: Callable[[str], None] = print


def run_pipeline(
    task: str,
    scene: Scene,
    config: Config,
    client: ChatClient,
    interfaces: Interfaces,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """Run the dialogue. Conversations and the result are saved even if a call raises."""
    start = time.time()
    usage = Usage(config.llm)
    planner = Planner(client, usage)
    analyzer = Analyzer(client, usage)
    result = PipelineResult()
    try:
        _loop(task, scene, config, planner, analyzer, interfaces, result)
    finally:
        result.cost_usd = usage.cost_usd
        result.duration_s = round(time.time() - start, 3)
        if output_dir is not None:
            _save(Path(output_dir), result, planner, analyzer, config)
    return result


def _loop(task, scene, config, planner, analyzer, interfaces, result) -> None:
    log = interfaces.log
    checks = config.checks
    planner_environment = scene.to_prompt()
    analyzer_environment = scene.to_prompt()
    iteration = 0
    analyzer_calls = 0
    call_type = "first"
    feedback = None
    question_count = 0
    clarifications = ""
    analyzer_scene_update = None  # A rescan the Analyzer has not been sent yet.

    while iteration < config.max_iterations:
        result.iterations = iteration + 1
        log(f"Feedback Loop {iteration + 1}")
        plan = planner.call(call_type, task, planner_environment, feedback)
        log(f"Planner:\n{plan}")
        lowered = plan.lower()

        if "perception" in lowered:
            scene = interfaces.rescan()
            feedback = scene.to_prompt()
            log(f"Updated environment:\n{feedback}")
            if analyzer_calls == 0:
                analyzer_environment = feedback
            else:
                analyzer_scene_update = feedback
            call_type = "perception"
        elif "question" in lowered:
            answers = ["Answers:"]
            for number in range(1, count_questions(plan) + 1):
                answers.append(f"{number}: {interfaces.ask_human(number)}")
            feedback = "\n".join(answers)
            call_type = "human"
            added, question_count = format_clarifications(plan, feedback, question_count)
            clarifications += added
        elif "step" in lowered:
            # Bounds violations go back to the Planner as Analyzer feedback, and
            # use up an iteration like any other feedback.
            call_type = "analyzer"
            if checks.bounds:
                feedback = check_bounds(plan, config.sim.x_bounds, config.sim.y_bounds)
                if "out of bounds" in feedback:
                    log(f"Bounds check:\n{feedback}")
                    iteration += 1
                    continue
            if checks.analyzer:
                analyzer_call = "first" if analyzer_calls == 0 else "feedback"
                feedback = analyzer.call(
                    analyzer_call, task, analyzer_environment, plan, clarifications, analyzer_scene_update
                )
                analyzer_scene_update = None
                log(f"Analyzer:\n{feedback}")
            else:
                feedback = "feasible plan"
            if "feasible" in feedback.lower() and "plan" in feedback.lower():
                result.final_plan = to_simulator_format(plan)
                if not checks.simulator:
                    break
                feedback = interfaces.simulate(result.final_plan, scene)
                if feedback != "success":
                    log(f"Simulator: {feedback}")
                    call_type = "simulator"
                else:
                    log("The plan is feasible!")
                    result.simulator_passed = True
                    break
            analyzer_calls += 1
        else:
            break
        iteration += 1


def _save(output_dir: Path, result, planner, analyzer, config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    planner.conversation.save(output_dir / "planner_conversation.txt")
    analyzer.conversation.save(output_dir / "analyzer_conversation.txt")
    if result.final_plan:
        (output_dir / "final_plan.txt").write_text(result.final_plan, encoding="utf-8")
    summary = {**asdict(result), "model": config.llm.model}
    summary.pop("log")
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
