from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return F.relu(x + residual)


class DepthNetwork(nn.Module):
    """
    Trainable depth estimation network.
    Takes 2D bounding box features and predicts depth.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [256, 512, 512, 256],
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout)
        )

        # Middle layers with optional residual connections
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i + 1]:
                self.layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                )

        # Output head - predicts depth value
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus(),  # Ensures positive depth
        )

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, N, input_dim] - Features from 2D bounding boxes

        Returns:
            depth: [B, N, 1] - Predicted depth values
            uncertainty: [B, N, 1] - Prediction uncertainty
        """
        x = self.input_proj(features)

        for layer in self.layers:
            x = layer(x)

        depth = self.depth_head(x) + 1.0  # Minimum depth of 1 meter
        uncertainty = self.uncertainty_head(x)

        return depth, uncertainty


class BBox3DHead(nn.Module):
    """
    Converts 2D box + depth to 3D bounding box parameters.
    Predicts: height, width, length, rotation
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim + 1  # +1 for depth

        for i in range(num_layers):
            layers.extend(
                [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Separate heads for different properties
        self.size_head = nn.Linear(hidden_dim, 3)  # h, w, l
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, 2), nn.Tanh()  # sin, cos of theta
        )

    def forward(
        self, features: torch.Tensor, depth: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, N, input_dim] - 2D box features
            depth: [B, N, 1] - Predicted depth

        Returns:
            Dictionary with 3D box parameters
        """
        # Concatenate features with depth
        x = torch.cat([features, depth], dim=-1)
        x = self.backbone(x)

        # Predict size (ensure positive values)
        size = F.softplus(self.size_head(x)) + 0.5  # Min size 0.5m

        # Predict rotation
        rotation = self.rotation_head(x)  # [B, N, 2] - sin, cos

        return {"size": size, "rotation": rotation}


class SimpleFeatureExtractor(nn.Module):
    """
    Simple feature extractor from 2D bounding boxes.
    Uses box coordinates and appearance features.
    """

    def __init__(self, output_dim: int = 256):
        super().__init__()

        # Box geometry encoder (4 coords -> features)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim // 2),
        )

        # Spatial encoder (position in image)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, output_dim // 2)
        )

    def forward(
        self, boxes_2d: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
            boxes_2d: [B, N, 4] - 2D boxes (x1, y1, x2, y2) in pixels
            image_size: (height, width)

        Returns:
            features: [B, N, output_dim]
        """
        B, N, _ = boxes_2d.shape
        H, W = image_size

        # Normalize boxes to [0, 1]
        boxes_norm = boxes_2d.clone()
        boxes_norm[:, :, [0, 2]] /= W
        boxes_norm[:, :, [1, 3]] /= H

        # Extract geometry features
        geometry_feat = self.geometry_encoder(boxes_norm)

        # Extract spatial features (box centers)
        centers = torch.stack(
            [
                (boxes_norm[:, :, 0] + boxes_norm[:, :, 2]) / 2,
                (boxes_norm[:, :, 1] + boxes_norm[:, :, 3]) / 2,
            ],
            dim=-1,
        )
        spatial_feat = self.spatial_encoder(centers)

        # Concatenate features
        features = torch.cat([geometry_feat, spatial_feat], dim=-1)

        return features


class DepthRefinementModel(nn.Module):
    """
    Complete model for depth refinement on OpenAD
    """

    def __init__(self, depth_config: Dict = None, box_head_config: Dict = None):
        super().__init__()

        # Feature extractor (replaces frozen 2D detector for now)
        self.feature_extractor = SimpleFeatureExtractor(output_dim=256)

        # Trainable components
        depth_config = depth_config or {}

        # Extract freeze flag before passing to DepthNetwork
        freeze_depth = depth_config.pop("freeze", False)

        self.depth_net = DepthNetwork(**depth_config)

        # FREEZE DEPTH NETWORK IF SPECIFIED
        if freeze_depth:
            print("🔒 FREEZING DEPTH NETWORK - Baseline Mode")
            for param in self.depth_net.parameters():
                param.requires_grad = False
        else:
            print("✓ Depth network is trainable")

        box_head_config = box_head_config or {}
        self.box_3d_head = BBox3DHead(**box_head_config)

    def forward(
        self,
        images: torch.Tensor,
        boxes_2d: List[torch.Tensor],
        camera_intrinsics: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass through the model

        Args:
            images: [B, 3, H, W] - Input images
            boxes_2d: List of [N_i, 4] tensors - 2D boxes for each image
            camera_intrinsics: [B, 3, 3] - Camera parameters

        Returns:
            Dictionary with predictions for each image in batch
        """
        B, _, H, W = images.shape

        # Process each image separately (variable number of boxes)
        all_predictions = {
            "depth": [],
            "uncertainty": [],
            "center_3d": [],
            "size": [],
            "rotation": [],
        }

        for i in range(B):
            if len(boxes_2d[i]) == 0:
                # No boxes in this image
                all_predictions["depth"].append(torch.empty(0, 1, device=images.device))
                all_predictions["uncertainty"].append(
                    torch.empty(0, 1, device=images.device)
                )
                all_predictions["center_3d"].append(
                    torch.empty(0, 3, device=images.device)
                )
                all_predictions["size"].append(torch.empty(0, 3, device=images.device))
                all_predictions["rotation"].append(torch.empty(0, device=images.device))
                continue

            # Get boxes for this image [N, 4]
            boxes = boxes_2d[i].unsqueeze(0)  # [1, N, 4]

            # Extract features
            features = self.feature_extractor(boxes, (H, W))  # [1, N, 256]

            # Predict depth
            depth, uncertainty = self.depth_net(features)  # [1, N, 1]

            # Predict 3D box parameters
            box_params = self.box_3d_head(features, depth)

            # Convert 2D center to 3D using depth and camera intrinsics
            K = camera_intrinsics[i]  # [3, 3]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # Get box centers in image coordinates
            boxes_single = boxes.squeeze(0)  # [N, 4]
            center_x = (boxes_single[:, 0] + boxes_single[:, 2]) / 2
            center_y = (boxes_single[:, 1] + boxes_single[:, 3]) / 2

            # Unproject to 3D
            depth_vals = depth.squeeze(0).squeeze(-1)  # [N]
            x_3d = (center_x - cx) * depth_vals / fx
            y_3d = (center_y - cy) * depth_vals / fy
            z_3d = depth_vals

            center_3d = torch.stack([x_3d, y_3d, z_3d], dim=-1)  # [N, 3]

            # Extract theta from sin/cos
            sin_theta = box_params["rotation"].squeeze(0)[:, 0]
            cos_theta = box_params["rotation"].squeeze(0)[:, 1]
            theta = torch.atan2(sin_theta, cos_theta)

            # Store predictions
            all_predictions["depth"].append(depth.squeeze(0))
            all_predictions["uncertainty"].append(uncertainty.squeeze(0))
            all_predictions["center_3d"].append(center_3d)
            all_predictions["size"].append(box_params["size"].squeeze(0))
            all_predictions["rotation"].append(theta)

        return all_predictions


def build_model(config: Dict) -> DepthRefinementModel:
    """Build model from config dictionary"""
    model_config = config.get("model", {})

    depth_config = model_config.get(
        "depth_network", {}
    ).copy()  # Copy to avoid modifying original
    box_head_config = model_config.get("box_head", {})

    model = DepthRefinementModel(
        depth_config=depth_config, box_head_config=box_head_config
    )

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Depth Refinement Model...")

    model = DepthRefinementModel()

    # Dummy inputs
    images = torch.randn(2, 3, 640, 480)
    boxes_2d = [
        torch.rand(5, 4) * 400,  # 5 boxes in first image
        torch.rand(3, 4) * 400,  # 3 boxes in second image
    ]
    camera_intrinsics = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    camera_intrinsics[:, 0, 0] = 1000  # fx
    camera_intrinsics[:, 1, 1] = 1000  # fy
    camera_intrinsics[:, 0, 2] = 320  # cx
    camera_intrinsics[:, 1, 2] = 240  # cy

    # Forward pass
    predictions = model(images, boxes_2d, camera_intrinsics)

    print("\nOutput:")
    for key, values in predictions.items():
        print(f"  {key}: {len(values)} images")
        for i, v in enumerate(values):
            print(f"    Image {i}: {v.shape}")

    print("\nModel trainable parameters:")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  Percentage trainable: {100 * trainable / total:.2f}%")
