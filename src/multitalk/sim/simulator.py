"""Run a plan in MuJoCo and report the first failure.

The environment and control law are adapted from FR3Env
(MIT, (c) 2022 Bolun Dai).
"""

from __future__ import annotations

import re

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
from ndcurves import SE3Curve

from ..config import SimConfig
from ..types import Scene
from .controller import Q_NOMINAL, DiffIK
from .objects import model_for
from .scene_builder import ROBOT_DIR, build_model

CONTROL_DT = 2e-3
TRAVEL_HEIGHT = 0.376
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
EE_ROTATION = np.diag([1.0, -1.0, -1.0])
_STEP = re.compile(r"\d+: (\w+)\((.*)\)")
_COORDINATES = re.compile(r"\(([^)]+)\)")


def _waypoint(x, y, z, gripper, duration):
    return np.array([[x], [y], [z], [gripper]]), duration


def _coordinates(arguments: str) -> list[tuple[float, ...]]:
    return [tuple(map(float, m.group(1).split(","))) for m in _COORDINATES.finditer(arguments)]


def plan_to_waypoints(plan: str, objects: dict, config: SimConfig) -> list:
    """Expand a plan in simulator format into (target, duration) waypoints."""
    locations = {tuple(value[1][0:2]): name for name, value in objects.items()}
    waypoints = []
    for line in plan.split("\n"):
        match = _STEP.search(line)
        if not match:
            continue
        command = match.group(1).strip().lower()
        arguments = match.group(2)
        if command == "grasp":
            x, y = _coordinates(arguments)[0][0:2]
            name = locations.get((x, y))
            if name is None:
                raise Exception(f"There is not graspable object at {(x, y)}")
            model = model_for(name)
            waypoints += [
                _waypoint(x, y, TRAVEL_HEIGHT, GRIPPER_OPEN, 2),
                _waypoint(x, y, model.grasp_height, GRIPPER_OPEN, 2),
                _waypoint(x, y, model.grasp_height, GRIPPER_CLOSED, model.grasp_close_time),
                _waypoint(x, y, TRAVEL_HEIGHT, GRIPPER_CLOSED, 2),
            ]
        elif command == "move":
            coordinates = _coordinates(arguments)
            grasp_position = coordinates[0][0:2]
            place_position = coordinates[1][0:2]
            radius = np.linalg.norm(np.array(place_position))
            if radius < config.min_place_radius:
                raise Exception(
                    f"Place position {place_position} is too close to the robot base. Placing "
                    f"position should be at least outside the radius of "
                    f"{config.min_place_radius} from the origin."
                )
            if radius > config.max_place_radius:
                raise Exception(
                    f"Place position {place_position} is too far to the robot base. Placing "
                    f"position should be at most {config.max_place_radius} away from the robot base."
                )
            name = locations.pop(grasp_position)
            locations[place_position] = name
            model = model_for(name)
            x, y = place_position
            waypoints += [
                _waypoint(x, y, TRAVEL_HEIGHT, GRIPPER_CLOSED, 2),
                _waypoint(x, y, model.place_height, GRIPPER_CLOSED, 1),
                _waypoint(x, y, model.place_height, GRIPPER_OPEN, 1),
                _waypoint(x, y, TRAVEL_HEIGHT, GRIPPER_OPEN, 1),
            ]
        elif command == "home":
            waypoints.append(_waypoint(0.3, 0.0, 0.4, GRIPPER_OPEN, 1.0))
        else:
            raise Exception(f"'{command}' is not available in robot defined commands.")
    return waypoints


class FR3Simulation:
    def __init__(self, scene: Scene, plan: str, config: SimConfig, render: bool = False) -> None:
        self.config = config
        self.controller = DiffIK()
        self.model, geom_names = build_model(scene.objects)
        self.data = mujoco.MjData(self.model)
        self.object_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom): obj
            for geom, obj in geom_names.items()
        }
        self.pin_model = pin.buildModelFromUrdf(str(ROBOT_DIR / "fr3.urdf"))
        self.pin_data = self.pin_model.createData()
        self.ee_frame = self.pin_model.getFrameId("fr3_hand_tcp")
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 3.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
            self.viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
        self.model.opt.gravity[2] = -9.81
        self.data.qpos[:9] = Q_NOMINAL
        self.info = self._robot_state()
        self.waypoints = plan_to_waypoints(plan, scene.objects, config)
        self.t = 0.0
        self.target, duration = self.waypoints.pop(0)
        self._start_segment(duration)

    def _robot_state(self) -> dict:
        q = self.data.qpos[:9].copy()
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
        jacobian = pin.getFrameJacobian(
            self.pin_model, self.pin_data, self.ee_frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        pose = self.pin_data.oMf[self.ee_frame]
        return {
            "q": q,
            "dq": self.data.qvel[:9].copy(),
            "jacobian": jacobian,
            "gravity": pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q).copy(),
            "rotation": pose.rotation.copy(),
            "position": pose.translation.copy(),
        }

    def _start_segment(self, duration: float) -> None:
        start = pin.SE3(self.info["rotation"], self.info["position"])
        end = pin.SE3(EE_ROTATION, self.target[:3, 0])
        self.segment_end = self.t + duration
        self.curve = SE3Curve(start, end, self.t, self.segment_end)

    def step(self) -> bool:
        """Advance one control step. Return True once the plan has finished."""
        if self.viewer is not None and not self.viewer.is_running():
            raise Exception("Env Viewer is not running!")
        pose = pin.SE3(self.info["rotation"], self.info["position"])
        self.t = np.clip(self.t + CONTROL_DT, 0.0, self.segment_end)
        error = pin.log6(self.curve(self.t) @ pose.inverse().homogeneous).vector
        v_des = self.curve.derivate(self.t, 1) + 20.0 * error
        delta_q = self.controller(v_des, self.info["q"], self.info["jacobian"])
        tau = 10.0 * delta_q + self.info["gravity"] - 20.0 * self.info["dq"]

        self.data.ctrl[:] = np.append(tau[:7], self.target[3, 0])
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        self.info = self._robot_state()

        self._check_workspace()
        self._check_collisions()
        if self.t >= self.segment_end:
            if not self.waypoints:
                return True
            self._check_target()
            self.target, duration = self.waypoints.pop(0)
            self._start_segment(duration)
        return False

    def _check_collisions(self) -> None:
        for geom1, geom2 in np.unique(self.data.contact.geom, axis=0):
            if geom1 in self.object_geoms and geom2 in self.object_geoms:
                raise Exception(
                    f"Collision detected between {self.object_geoms[geom1]} "
                    f"and {self.object_geoms[geom2]}"
                )

    def _check_workspace(self) -> None:
        if np.linalg.cond(np.linalg.pinv(self.info["jacobian"])) > self.config.max_jacobian_condition:
            raise Exception(
                f"Robot is close to out of its workspace while approaching target "
                f"{self.target[0:3, 0]}. It is not safe to operate the robot in this region."
            )

    def _check_target(self) -> None:
        error = np.linalg.norm(self.info["position"] - self.target[:3, 0])
        if error > self.config.max_target_error:
            raise Exception(f"Robot could not reach the target {self.target[:3, 0]}")

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


def simulate_plan(plan: str, scene: Scene, config: SimConfig, render: bool = False) -> str:
    """Return 'success', or the failure text to send back to the Planner."""
    simulation = None
    try:
        simulation = FR3Simulation(scene, plan, config, render=render)
        for _ in range(config.max_steps):
            if simulation.step():
                break
        return "success"
    except Exception as failure:  # Any failure becomes feedback for the Planner.
        return " ".join(str(failure).replace("\n", "").split())
    finally:
        if simulation is not None:
            simulation.close()
