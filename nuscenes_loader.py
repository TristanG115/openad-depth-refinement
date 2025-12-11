import os

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset


class NuScenesDepthDataset(Dataset):
    """
    Load data directly from nuScenes format
    No OpenAD annotations needed!
    """

    def __init__(
        self, nuscenes_root: str, split: str = "train", version: str = "v1.0-trainval"
    ):
        """
        Args:
            nuscenes_root: Path to nuScenes data
            split: 'train' or 'val'
            version: 'v1.0-mini' or 'v1.0-trainval'
        """
        self.nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
        self.nuscenes_root = nuscenes_root

        # Get samples for the split
        self.samples = self._get_samples(split)

        print(f"Loaded {len(self.samples)} samples from nuScenes {split} split")

    def _get_samples(self, split):
        """Get sample tokens for train/val split, filtering for available images"""
        samples = []

        for scene in self.nusc.scene:
            # Simple split: odd scenes for val, even for train
            scene_token = scene["token"]
            scene_name = scene["name"]

            # Get first sample in scene
            sample_token = scene["first_sample_token"]

            # Determine split based on scene number
            scene_num = int(scene_name.split("-")[-1])

            # Check if this scene has images available
            # Test first sample to see if images exist
            test_sample = self.nusc.get("sample", sample_token)
            cam_token = test_sample["data"]["CAM_FRONT"]
            cam = self.nusc.get("sample_data", cam_token)
            test_path = os.path.normpath(
                os.path.join(self.nuscenes_root, cam["filename"])
            )

            # Skip this entire scene if first image doesn't exist
            if not os.path.exists(test_path):
                continue

            if split == "train" and scene_num % 5 != 0:
                # 80% train
                while sample_token:
                    samples.append(sample_token)
                    sample = self.nusc.get("sample", sample_token)
                    sample_token = sample["next"]
            elif split == "val" and scene_num % 5 == 0:
                # 20% val
                while sample_token:
                    samples.append(sample_token)
                    sample = self.nusc.get("sample", sample_token)
                    sample_token = sample["next"]

        return samples

    def __len__(self):
        return len(self.samples)

    def _get_camera_intrinsic(self, cam_token):
        """Get camera intrinsic matrix"""
        cam = self.nusc.get("sample_data", cam_token)
        cs_record = self.nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])

        # Camera intrinsic matrix
        K = np.array(cs_record["camera_intrinsic"], dtype=np.float32)
        return K

    def _get_boxes_in_camera(self, sample_token, cam_token):
        """Get 3D boxes in camera coordinates with 2D projections"""
        sample = self.nusc.get("sample", sample_token)
        cam = self.nusc.get("sample_data", cam_token)

        # Get camera calibration
        cs_record = self.nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])

        # Get pose
        pose_record = self.nusc.get("ego_pose", cam["ego_pose_token"])

        # Get all boxes in this sample
        boxes_2d = []
        boxes_3d = []
        labels = []

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            # Get box in global coordinates
            box_global = np.array(ann["translation"])
            size = ann["size"]  # [width, length, height] in nuScenes
            rotation = Quaternion(ann["rotation"])

            # Transform to ego vehicle frame
            box_ego = box_global - np.array(pose_record["translation"])
            rot_ego = Quaternion(pose_record["rotation"]).inverse
            box_ego = rot_ego.rotate(box_ego)

            # Transform to camera frame
            box_cam = box_ego - np.array(cs_record["translation"])
            rot_cam = Quaternion(cs_record["rotation"]).inverse
            box_cam = rot_cam.rotate(box_cam)

            # Check if box is in front of camera
            if box_cam[2] < 1.0:  # Too close or behind
                continue

            # Project center to image
            center_2d = view_points(
                box_cam.reshape(3, 1), cam_intrinsic, normalize=True
            )
            center_x = center_2d[0, 0]
            center_y = center_2d[1, 0]

            # Get image size
            img_width = cam["width"]
            img_height = cam["height"]

            # Simple 2D box estimation (can be improved)
            # Use projected center and approximate size
            proj_size = (size[0] + size[1]) / 2  # Average of width and length
            pixel_size = proj_size * cam_intrinsic[0, 0] / box_cam[2]  # Approximate

            x1 = max(0, center_x - pixel_size / 2)
            y1 = max(0, center_y - pixel_size / 2)
            x2 = min(img_width, center_x + pixel_size / 2)
            y2 = min(img_height, center_y + pixel_size / 2)

            # Skip if box is outside image
            if x2 <= x1 or y2 <= y1:
                continue
            if x1 >= img_width or y1 >= img_height:
                continue

            # Convert rotation to angle
            # nuScenes uses quaternions, convert to yaw angle
            yaw = rotation.yaw_pitch_roll[0]

            # Store boxes
            boxes_2d.append([x1, y1, x2, y2])
            # nuScenes format: [x, y, z, width, length, height]
            # We want: [x, y, z, height, width, length, yaw]
            boxes_3d.append(
                [
                    box_cam[0],
                    box_cam[1],
                    box_cam[2],  # x, y, z in camera frame
                    size[2],
                    size[0],
                    size[1],  # h, w, l
                    yaw,
                ]
            )
            labels.append(ann["category_name"])

        return {
            "boxes_2d": (
                np.array(boxes_2d, dtype=np.float32)
                if boxes_2d
                else np.zeros((0, 4), dtype=np.float32)
            ),
            "boxes_3d": (
                np.array(boxes_3d, dtype=np.float32)
                if boxes_3d
                else np.zeros((0, 7), dtype=np.float32)
            ),
            "labels": labels,
        }

    def __getitem__(self, idx):
        """Get a training sample"""
        sample_token = self.samples[idx]
        sample = self.nusc.get("sample", sample_token)

        # Use front camera
        cam_token = sample["data"]["CAM_FRONT"]
        cam = self.nusc.get("sample_data", cam_token)

        # Load image - use proper path normalization for Windows
        img_filename = cam["filename"]
        img_path = os.path.normpath(os.path.join(self.nuscenes_root, img_filename))

        # Check if file exists (shouldn't happen if filtering worked)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            # Return a dummy sample rather than recursing
            # This shouldn't happen if _get_samples filtered correctly
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Get camera intrinsic
        camera_intrinsic = self._get_camera_intrinsic(cam_token)

        # Get boxes
        boxes = self._get_boxes_in_camera(sample_token, cam_token)

        return {
            "image": image,
            "boxes_2d": torch.from_numpy(boxes["boxes_2d"]),
            "boxes_3d": torch.from_numpy(boxes["boxes_3d"]),
            "camera_internal": torch.from_numpy(camera_intrinsic),
            "camera_external": torch.eye(4),  # Already in camera frame
            "labels": boxes["labels"],
            "sample_token": sample_token,
        }


def collate_fn(batch):
    """Custom collate for variable number of boxes"""
    images = torch.stack([item["image"] for item in batch])
    boxes_2d = [item["boxes_2d"] for item in batch]
    boxes_3d = [item["boxes_3d"] for item in batch]
    camera_internal = torch.stack([item["camera_internal"] for item in batch])
    camera_external = torch.stack([item["camera_external"] for item in batch])
    labels = [item["labels"] for item in batch]
    sample_tokens = [item["sample_token"] for item in batch]

    return {
        "image": images,
        "boxes_2d": boxes_2d,
        "boxes_3d": boxes_3d,
        "camera_internal": camera_internal,
        "camera_external": camera_external,
        "labels": labels,
        "sample_token": sample_tokens,
    }


if __name__ == "__main__":
    # Test the dataset
    print("Testing NuScenes Direct Loader...")

    dataset = NuScenesDepthDataset(
        nuscenes_root="./data/nuscenes", split="train", version="v1.0-trainval"
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Number of boxes: {len(sample['boxes_2d'])}")
    if len(sample["boxes_2d"]) > 0:
        print(f"  First box 2D: {sample['boxes_2d'][0]}")
        print(f"  First box 3D: {sample['boxes_3d'][0]}")
        print(f"  Depth (z): {sample['boxes_3d'][0][2]:.2f} meters")
        print(f"  Label: {sample['labels'][0]}")

    print("\n✓ Dataset working!")
