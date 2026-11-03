"""Configuration, loaded from YAML with paths resolved from the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
PROMPTS_DIR = PACKAGE_DIR / "prompts"
PROJECT_ROOT = PACKAGE_DIR.parent.parent


@dataclass
class LLMConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512


@dataclass
class ChecksConfig:
    """Disabling checks gives the ablated variants of the framework."""

    bounds: bool = True
    analyzer: bool = True
    simulator: bool = True


@dataclass
class SimConfig:
    x_bounds: tuple[float, float] = (-0.15, 0.45)
    y_bounds: tuple[float, float] = (-0.62, 0.62)
    min_place_radius: float = 0.254
    max_place_radius: float = 0.762
    max_jacobian_condition: float = 40.0
    max_target_error: float = 0.15
    max_steps: int = 1_000_000


@dataclass
class PerceptionConfig:
    target_objects: str = "sugar box . mustard bottle . food can . apple . wooden cube ."
    box_threshold: float = 0.50
    text_threshold: float = 0.35
    min_object_points: int = 700
    dino_weights: str = "weights/groundingdino_swint_ogc.pth"
    sam_weights: str = "weights/mobile_sam.pt"
    sam_backbone: str = "vit_t"


@dataclass
class Config:
    max_iterations: int = 10
    output_dir: str = "runs"
    llm: LLMConfig = field(default_factory=LLMConfig)
    checks: ChecksConfig = field(default_factory=ChecksConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)


def resolve_path(path: str | Path) -> Path:
    """Resolve a relative config path against the project root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> Config:
    config = Config()
    if path is None:
        return config
    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    sections = {
        "llm": config.llm,
        "checks": config.checks,
        "sim": config.sim,
        "perception": config.perception,
    }
    for key, value in data.items():
        target = sections.get(key)
        if target is not None:
            _apply(target, value, prefix=f"{key}.")
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return config


def _apply(target, values: dict, prefix: str) -> None:
    for key, value in (values or {}).items():
        if not hasattr(target, key):
            raise ValueError(f"Unknown config key: {prefix}{key}")
        setattr(target, key, tuple(value) if isinstance(value, list) else value)
