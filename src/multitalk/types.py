"""Data types shared by the pipeline, simulator and perception."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Scene:
    """Objects on the table as {name: [dimensions, center]} in the robot base frame."""

    objects: dict

    @classmethod
    def from_file(cls, path: str | Path) -> "Scene":
        with open(path, encoding="utf-8") as file:
            return cls(json.load(file))

    def to_prompt(self) -> str:
        """Format the scene as the Python-like dictionary the prompts describe."""
        return json.dumps(self.objects).replace("[", "(").replace("]", ")")


@dataclass
class PipelineResult:
    """The outcome of a run.

    `final_plan` is the last plan the Analyzer accepted, as in the original
    implementation. It has only passed the simulator if `simulator_passed` is True.
    """

    simulator_passed: bool = False
    final_plan: str | None = None
    iterations: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    log: list[str] = field(default_factory=list)
