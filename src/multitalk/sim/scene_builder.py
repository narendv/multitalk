"""Build the MuJoCo model for a scene."""

from __future__ import annotations

import mujoco

from ..config import ASSETS_DIR
from .objects import _FRICTION, ObjectModel, model_for

ROBOT_DIR = ASSETS_DIR / "robot"
YCB_DIR = ASSETS_DIR / "ycb"
OBJECT_SPAWN_HEIGHT = 0.85


def _object_xml(name: str, center, model: ObjectModel, geom_name: str) -> str:
    # The world frame is rotated and offset from the robot base frame, which
    # all scene and plan coordinates use.
    world_x = -center[1]
    world_y = center[0] - 0.25
    if model.name == "cube":
        geoms = (
            '<inertial mass="0.01" pos="0 0 0" fullinertia="1.04e-6 1.04e-6 1.04e-6 0.0 0.0 0.0"/>'
            f'<geom name="{geom_name}" type="box" size="{model.collision_size}" '
            f'pos="{model.collision_pos}" material="block"/>'
        )
    else:
        geoms = (
            model.visual
            + f'<geom name="{geom_name}" class="ycb_col" type="box" '
            f'size="{model.collision_size}" pos="{model.collision_pos}" '
            f'mass="0.014" {_FRICTION}/>'
        )
    return (
        f'<body name="{name}" pos="{world_x} {world_y} {OBJECT_SPAWN_HEIGHT}">'
        f"{geoms}<freejoint/></body>"
    )


def build_model(objects: dict) -> tuple[mujoco.MjModel, dict[str, str]]:
    """Return the model and a mapping from collision geom name to object name."""
    counts: dict[str, int] = {}
    geom_names: dict[str, str] = {}
    bodies = []
    assets: dict[str, str] = {}
    for name, (_, center) in objects.items():
        model = model_for(name)
        if model is None:
            raise Exception(f"Object type is not defined for {name}")
        counts[model.name] = counts.get(model.name, 0) + 1
        geom_name = f"{model.name}{counts[model.name]}"
        geom_names[geom_name] = name
        assets[model.name] = model.asset.replace("{ycb}", str(YCB_DIR))
        bodies.append(_object_xml(name, center, model, geom_name))

    xml = f"""
<mujoco>
    <compiler angle="radian" meshdir="{ROBOT_DIR}/mj_assets" autolimits="true"/>
    <statistic center="0.3 0 0.4" extent="1"/>
    <option integrator="implicitfast"/>
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
    </visual>
    <default>
        <include file="{ROBOT_DIR}/table_default.xml"/>
        <include file="{ROBOT_DIR}/fr3_default.xml"/>
        <default class="ycb_col">
            <geom contype="1" conaffinity="1" group="3" rgba=".4 .5 .6 1"/>
        </default>
        <default class="ycb_viz">
            <geom contype="0" conaffinity="0" type="mesh"/>
        </default>
    </default>
    <asset>
        <include file="{ROBOT_DIR}/table_assets.xml"/>
        <include file="{ROBOT_DIR}/fr3_assets.xml"/>
        <material name="block" rgba="1 0 0 1" reflectance="1" shininess=".5"/>
        {''.join(assets.values())}
    </asset>
    <worldbody>
        <light name="top" pos="0 0 2" mode="trackcom"/>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <include file="{ROBOT_DIR}/table_body.xml"/>
        <include file="{ROBOT_DIR}/fr3_on_table_body.xml"/>
        {''.join(bodies)}
    </worldbody>
    <include file="{ROBOT_DIR}/fr3_motors.xml"/>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.impratio = 2.0
    return model, geom_names
