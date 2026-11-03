"""Detect objects in an RGB-D capture with GroundingDINO, MobileSAM and depth.

A capture directory holds rgb.png, depth.png (16-bit, millimeters), and the
hand pose at capture time as R_hand.npy and p_hand.npy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ASSETS_DIR, PerceptionConfig, resolve_path
from ..types import Scene

CALIBRATION_DIR = ASSETS_DIR / "calibration"
LABEL_TO_CLASS = {"sugar box": "sugar box", "mustard bottle": "mustard bottle", "metal food": "soup can", "apple": "apple"}
HAND_T_CAMERA = np.array(
    [
        [-0.028, 1.0, -0.001, 0.05],
        [-1.0, -0.028, -0.005, -0.008],
        [-0.005, 0.001, 1.0, 0.006],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def camera_to_world(capture_dir: Path) -> np.ndarray:
    world_t_hand = np.eye(4)
    world_t_hand[:3, :3] = np.load(capture_dir / "R_hand.npy")
    world_t_hand[:3, 3] = np.load(capture_dir / "p_hand.npy")
    return world_t_hand @ HAND_T_CAMERA


def depth_to_points(depth_mm: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Return an (H, W, 3) point map in meters, aligned to the RGB camera."""
    calibration = json.loads((CALIBRATION_DIR / "calibration.json").read_text())
    depth_to_rgb = np.array(
        json.loads((CALIBRATION_DIR / "depth_to_rgb_transform.json").read_text())["transform"]
    )
    depth_settings = calibration["depth"]
    depth = depth_mm.astype(np.float64)
    depth[depth > depth_settings["depth_high_threshold"] * 1000] = 0
    depth[depth < depth_settings["depth_low_threshold"] * 1000] = 0

    points = _back_project(depth / 1000.0, depth_settings["intrinsics"]).reshape(-1, 3)
    points = np.hstack((points, np.ones((len(points), 1))))
    points = (depth_to_rgb @ points.T).T[:, :3]

    # Reproject into the undistorted RGB image to align depth with the masks.
    rgb_intrinsics = calibration["rgb"]["new_intrinsics"]
    aligned = np.zeros(image_shape, dtype=np.float32)
    valid = points[:, 2] > 0
    x, y, z = points[valid].T
    u = np.clip(np.round(rgb_intrinsics["fx"] * x / z + rgb_intrinsics["cx"]).astype(int), 0, image_shape[1] - 1)
    v = np.clip(np.round(rgb_intrinsics["fy"] * y / z + rgb_intrinsics["cy"]).astype(int), 0, image_shape[0] - 1)
    aligned[v, u] = z
    return _back_project(aligned, rgb_intrinsics)


def _back_project(depth_m: np.ndarray, intrinsics: dict) -> np.ndarray:
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    return np.stack((x, y, depth_m), axis=-1)


class Perceptor:
    def __init__(self, config: PerceptionConfig | None = None, device: str | None = None) -> None:
        import groundingdino
        import torch
        from groundingdino.util.inference import load_model
        from mobile_sam import SamPredictor, sam_model_registry

        self.config = config or PerceptionConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dino_config = Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
        self._dino = load_model(
            str(dino_config), str(resolve_path(self.config.dino_weights)), device=self.device
        )
        sam = sam_model_registry[self.config.sam_backbone](
            checkpoint=str(resolve_path(self.config.sam_weights))
        )
        sam.to(device=self.device).eval()
        self._sam = SamPredictor(sam)

    def detect(self, capture_dir: str | Path) -> Scene:
        import cv2
        import open3d as o3d
        import torch
        from groundingdino.util.inference import load_image, predict
        from torchvision.ops import box_convert

        capture_dir = Path(capture_dir)
        depth = cv2.imread(str(capture_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read {capture_dir / 'depth.png'}")
        image, image_tensor = load_image(str(capture_dir / "rgb.png"))
        height, width = image.shape[:2]
        points = depth_to_points(depth, (height, width))

        boxes, _, labels = predict(
            model=self._dino,
            image=image_tensor,
            caption=self.config.target_objects,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            device=self.device,
        )
        if len(boxes) == 0:
            return Scene({})

        with torch.no_grad():
            pixel_boxes = boxes * torch.Tensor([width, height, width, height])
            xyxy = box_convert(pixel_boxes, in_fmt="cxcywh", out_fmt="xyxy").to(self.device)
            self._sam.set_image(image)
            transformed = self._sam.transform.apply_boxes_torch(xyxy, image.shape[:2])
            masks, _, _ = self._sam.predict_torch(
                point_coords=None, point_labels=None, boxes=transformed, multimask_output=False
            )
        masks = masks.cpu().numpy()

        world_t_camera = camera_to_world(capture_dir)
        objects = {}
        for index, (mask, label) in enumerate(zip(masks, labels)):
            object_points = points[mask.reshape(height, width)]
            if len(object_points) < self.config.min_object_points:
                continue
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(object_points)
            cloud = cloud.voxel_down_sample(voxel_size=0.01)
            cloud, _ = cloud.remove_radius_outlier(nb_points=16, radius=0.04)
            filtered = np.asarray(cloud.points)
            if len(filtered) == 0:
                continue
            low, high = filtered.min(axis=0), filtered.max(axis=0)
            center = world_t_camera @ np.array([*((low + high) / 2), 1.0])
            name = f"{LABEL_TO_CLASS.get(label, label)} {index}"
            objects[name] = (np.round(high - low, 3).tolist(), np.round(center[:3], 3).tolist())
        return Scene(objects)
