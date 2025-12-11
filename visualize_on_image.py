import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import FancyBboxPatch

from model_real import build_model
from nuscenes_loader import NuScenesDepthDataset


def load_model(checkpoint_path, config_path, device="cpu"):
    """Load trained model"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    return model, checkpoint


def visualize_single_model(
    model, checkpoint, model_name, dataset, idx, output_path, device="cpu"
):
    """Visualize one model's predictions on an image"""

    # Get sample
    sample = dataset[idx]
    boxes_2d_np = sample["boxes_2d"].numpy()
    boxes_3d_gt = sample["boxes_3d"].numpy()

    if len(boxes_2d_np) == 0:
        print(f"  ⚠ Skipping sample {idx} (no objects)")
        return

    # Prepare batch
    image = sample["image"].unsqueeze(0).to(device)
    boxes_2d = [sample["boxes_2d"].to(device)]
    camera_intrinsics = sample["camera_internal"].unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        predictions = model(image, boxes_2d, camera_intrinsics)

    pred_centers = predictions["center_3d"][0].cpu().numpy()
    pred_depths = pred_centers[:, 2] if len(pred_centers) > 0 else []

    # Get GT
    gt_centers = boxes_3d_gt[:, :3]
    gt_depths = gt_centers[:, 2]

    # Match predictions to GT
    matched_info = []
    if len(pred_depths) > 0:
        for i, pred_center in enumerate(pred_centers):
            distances = np.linalg.norm(gt_centers - pred_center, axis=1)
            closest_idx = distances.argmin()

            matched_info.append(
                {
                    "pred_depth": pred_depths[i],
                    "gt_depth": gt_depths[closest_idx],
                    "error": abs(pred_depths[i] - gt_depths[closest_idx]),
                    "box_2d": boxes_2d_np[closest_idx],
                }
            )

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Display image
    img_np = sample["image"].permute(1, 2, 0).numpy()
    ax.imshow(img_np)

    # Model info
    epoch = checkpoint.get("epoch", "?")
    overall_ate = checkpoint.get(
        "best_metric", checkpoint.get("metrics", {}).get("ATE", "?")
    )

    # Compute sample stats
    if matched_info:
        errors = [m["error"] for m in matched_info]
        avg_error = np.mean(errors)
        max_error = np.max(errors)

        title = f"{model_name}\n"
        title += f"Epoch {epoch} | Overall ATE: {overall_ate:.2f}m | "
        title += f"Sample {idx}: Avg Error={avg_error:.2f}m, Max Error={max_error:.2f}m"
    else:
        title = f"{model_name}\nEpoch {epoch} | Overall ATE: {overall_ate:.2f}m | Sample {idx}: No Detections"

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Draw boxes with predictions
    colors = plt.cm.Set3(np.linspace(0, 1, 12))

    for i, info in enumerate(matched_info):
        x1, y1, x2, y2 = info["box_2d"]
        pred_d = info["pred_depth"]
        gt_d = info["gt_depth"]
        error = info["error"]

        color = colors[i % len(colors)]

        # Determine error severity color
        if error < 2.0:
            error_color = "green"
            status = "✓"
        elif error < 5.0:
            error_color = "orange"
            status = "~"
        else:
            error_color = "red"
            status = "✗"

        # Draw box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Info box background
        info_width = 140
        info_height = 80
        info_bg = FancyBboxPatch(
            (x1, y1 - info_height - 5),
            info_width,
            info_height,
            boxstyle="round,pad=5",
            facecolor="white",
            edgecolor=color,
            linewidth=3,
            alpha=0.95,
        )
        ax.add_patch(info_bg)

        # Text content
        text_x = x1 + 10
        text_y = y1 - info_height + 15

        # Object number
        ax.text(
            text_x,
            text_y,
            f"Object #{i+1}",
            fontsize=11,
            fontweight="bold",
            color="black",
        )

        # Predicted depth
        ax.text(
            text_x,
            text_y + 20,
            f"Pred: {pred_d:.1f}m",
            fontsize=10,
            color="blue",
            fontweight="bold",
        )

        # GT depth
        ax.text(
            text_x,
            text_y + 35,
            f"GT: {gt_d:.1f}m",
            fontsize=10,
            color="green",
            fontweight="bold",
        )

        # Error
        ax.text(
            text_x,
            text_y + 50,
            f"{status} Error: {error:.1f}m",
            fontsize=10,
            color=error_color,
            fontweight="bold",
        )

    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  ✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help='Model name (e.g., "Ours", "Frozen", "Ablation")',
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Config file for this model"
    )
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 20, 30, 40])
    parser.add_argument("--output", type=str, default="./predictions")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Create output directory with model name
    output_dir = os.path.join(args.output, args.name.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"VISUALIZING MODEL: {args.name}")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    print(f"Using config: {args.config}...")
    model, checkpoint = load_model(args.checkpoint, args.config, args.device)
    epoch = checkpoint.get("epoch", "?")
    ate = checkpoint.get("best_metric", checkpoint.get("metrics", {}).get("ATE", "?"))
    print(f"✓ Loaded: Epoch {epoch}, ATE: {ate:.2f}m")

    # Load dataset
    print("\nLoading validation dataset...")
    dataset = NuScenesDepthDataset(
        nuscenes_root="./data/nuscenes", split="val", version="v1.0-trainval"
    )
    print(f"✓ Loaded {len(dataset)} samples")

    # Generate visualizations
    print(f"\nGenerating predictions for {len(args.samples)} samples...")
    for idx in args.samples:
        if idx >= len(dataset):
            print(f"  ⚠ Skipping sample {idx} (out of range)")
            continue

        output_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
        print(f"\nSample {idx}...")
        visualize_single_model(
            model, checkpoint, args.name, dataset, idx, output_path, args.device
        )

    print("\n" + "=" * 70)
    print(f"✓ All predictions saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
