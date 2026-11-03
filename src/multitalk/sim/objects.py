"""Simulation models and grasp parameters for the supported object classes.

Collision and visual parameters come from the YCB_sim body definitions and
FR3Env's cube. Heights are end-effector z-coordinates in the robot base frame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectModel:
    name: str
    grasp_height: float
    place_height: float
    grasp_close_time: float
    collision_size: str
    collision_pos: str
    visual: str = ""
    asset: str = ""


def _ycb_asset(mesh: str) -> str:
    return (
        f'<mesh name="{mesh}" file="{{ycb}}/meshes/{mesh}.msh" scale="1 1 1"/>'
        f'<texture name="{mesh}" type="2d" file="{{ycb}}/textures/{mesh}.png"/>'
        f'<material name="{mesh}" texture="{mesh}"/>'
    )


_APPLE_ASSET = (
    '<mesh name="013_apple" file="{ycb}/apple/textured.obj" scale="1 1 1"/>'
    '<texture name="013_apple" type="2d" file="{ycb}/apple/texture_map.png"/>'
    '<material name="013_apple" texture="013_apple"/>'
)
_FRICTION = 'friction="2 0.005 0.0001"'


def _visual(mesh: str, placement: str, friction: bool = True) -> str:
    extra = _FRICTION if friction else ""
    return (
        f'<geom class="ycb_viz" mesh="{mesh}" {placement} material="{mesh}" '
        f'mass="0.00" {extra}/>'
    )


# Order matters: the first class whose name appears in an object's name wins.
OBJECT_MODELS = (
    ObjectModel("cube", 0.0255, 0.0255, 2.0, "0.025 0.025 0.025", "0 0 0"),
    ObjectModel(
        "sugar box", 0.155, 0.160, 2.0, ".021 .045 .088", "0 0 .088",
        _visual("004_sugar_box", 'pos="0.005 0.018 0" euler="0.01 .02 0"'),
        _ycb_asset("004_sugar_box"),
    ),
    ObjectModel(
        "soup can", 0.07, 0.07, 2.0, ".033 .033 .05", "0 0 .05",
        _visual("005_tomato_soup_can", 'pos="0.01 -.084 -.002"'),
        _ycb_asset("005_tomato_soup_can"),
    ),
    ObjectModel(
        "tuna can", 0.024, 0.024, 2.0, ".03 .03 .016", "0 0 .016",
        _visual("007_tuna_fish_can", 'pos="0.0182 0.0168 0.002"'),
        _ycb_asset("007_tuna_fish_can"),
    ),
    ObjectModel(
        "mustard bottle", 0.145, 0.160, 4.0, ".022 .045 .088", "0 0 .088",
        _visual(
            "006_mustard_bottle",
            'pos="-0.028 0.014 0.002" quat="0.6428 0.0 0.0 0.7660"',
            friction=False,
        ),
        _ycb_asset("006_mustard_bottle"),
    ),
    ObjectModel(
        "apple", 0.035, 0.035, 4.0, "0.030 0.030 0.032", "0 0 0.032",
        _visual("013_apple", 'pos="0.0 0.0 0.0"'),
        _APPLE_ASSET,
    ),
)


def model_for(object_name: str) -> ObjectModel | None:
    for model in OBJECT_MODELS:
        if re.search(model.name, object_name):
            return model
    return None
