"""Command line interface: `multitalk plan`, `simulate` and `detect`."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .config import ASSETS_DIR, PROJECT_ROOT, Config, load_config, resolve_path
from .types import PipelineResult, Scene


def _resolve_scene(value: str) -> Path:
    """Accept a file path or a bundled scene name such as 'exp8/new_objects'."""
    path = Path(value).expanduser()
    if path.is_file():
        return path
    bundled = ASSETS_DIR / "scenes" / f"{value}.json"
    if bundled.is_file():
        return bundled
    raise SystemExit(f"Scene not found: {value}")


def _make_rescan(args, config: Config):
    """Return the rescan callback. Moving the camera is a manual step, as before."""
    if args.capture_dir:

        def rescan_capture() -> Scene:
            from .perception.perceptor import Perceptor

            input(f"Update the capture in {args.capture_dir}, then press Enter: ")
            return Perceptor(config.perception).detect(args.capture_dir)

        return rescan_capture

    def rescan_file() -> Scene:
        input(f"Update {args.scene_path}, then press Enter: ")
        return Scene.from_file(args.scene_path)

    return rescan_file


def _plan(args, config: Config) -> int:
    from .llm.client import MissingAPIKeyError, make_client
    from .pipeline import Interfaces, run_pipeline
    from .sim.simulator import simulate_plan

    if bool(args.scene) == bool(args.capture_dir):
        raise SystemExit("Pass exactly one of --scene or --capture-dir.")

    try:
        client = make_client(config.llm)
    except MissingAPIKeyError as error:
        raise SystemExit(str(error)) from error

    if args.scene:
        args.scene_path = _resolve_scene(args.scene)
        scene = Scene.from_file(args.scene_path)
    else:
        from .perception.perceptor import Perceptor

        scene = Perceptor(config.perception).detect(args.capture_dir)
    print(scene.objects)

    task = args.task or input("Enter the task: ")
    output_dir = args.output_dir or resolve_path(config.output_dir) / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    result = run_pipeline(
        task,
        scene,
        config,
        client,
        Interfaces(
            ask_human=lambda number: input(f"Enter the answer to the question {number}: "),
            rescan=_make_rescan(args, config),
            simulate=lambda plan, current: simulate_plan(plan, current, config.sim, render=args.render),
        ),
        output_dir,
    )
    _print_result(result, output_dir)
    return 0 if result.simulator_passed else 1


def _print_result(result: PipelineResult, output_dir) -> None:
    print(f"\nPassed the simulator: {result.simulator_passed}")
    if result.final_plan:
        print(f"Final plan:\n{result.final_plan}\n")
    print(f"Total number of loops: {result.iterations}")
    print(f"Total cost: {result.cost_usd}")
    print(f"Time taken: {result.duration_s} seconds")
    print(f"Saved to: {output_dir}")


def _simulate(args, config: Config) -> int:
    from .plan_text import to_simulator_format
    from .sim.simulator import simulate_plan

    scene = Scene.from_file(_resolve_scene(args.scene))
    plan = to_simulator_format(Path(args.plan).read_text(encoding="utf-8"))
    outcome = simulate_plan(plan, scene, config.sim, render=args.render)
    print(outcome)
    return 0 if outcome == "success" else 1


def _detect(args, config: Config) -> int:
    from .perception.perceptor import Perceptor

    scene = Perceptor(config.perception).detect(args.capture_dir)
    text = json.dumps(scene.objects, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(prog="multitalk", description=__doc__)
    parser.add_argument("--config", help="YAML config file (default: built-in settings)")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="Generate a plan for a task")
    plan.add_argument("--task", help="Task description (prompted for if omitted)")
    plan.add_argument("--scene", help="Scene file or bundled name, e.g. exp8/new_objects")
    plan.add_argument("--capture-dir", help="Detect the scene from this RGB-D capture")
    plan.add_argument("--output-dir", help="Where to save the run")
    plan.add_argument("--render", action="store_true", help="Show the simulator")
    plan.set_defaults(handler=_plan)

    simulate = commands.add_parser("simulate", help="Check a saved plan in the simulator")
    simulate.add_argument("--scene", required=True)
    simulate.add_argument("--plan", required=True, help="Text file containing the plan")
    simulate.add_argument("--render", action="store_true")
    simulate.set_defaults(handler=_simulate)

    detect = commands.add_parser("detect", help="Detect objects in an RGB-D capture")
    detect.add_argument("--capture-dir", required=True)
    detect.add_argument("--output", help="Save the scene to this JSON file")
    detect.set_defaults(handler=_detect)

    args = parser.parse_args(argv)
    return args.handler(args, load_config(args.config))


if __name__ == "__main__":
    sys.exit(main())
