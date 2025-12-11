import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

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


def create_bar_comparison(models_info, dataset, idx, output_path, device="cpu"):
    """Create clean bar chart comparing all models"""

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

    # Get GT depths
    gt_centers = boxes_3d_gt[:, :3]
    gt_depths = gt_centers[:, 2]

    # Get predictions from all models
    all_predictions = []
    for model_info in models_info:
        model = model_info["model"]
        with torch.no_grad():
            predictions = model(image, boxes_2d, camera_intrinsics)

        pred_centers = predictions["center_3d"][0].cpu().numpy()
        pred_depths = pred_centers[:, 2] if len(pred_centers) > 0 else []

        # Match to GT
        matched_pred = []
        if len(pred_depths) > 0:
            for pred_center in pred_centers:
                distances = np.linalg.norm(gt_centers - pred_center, axis=1)
                closest_idx = distances.argmin()
                matched_pred.append(pred_depths[closest_idx])

        all_predictions.append(matched_pred)

    # Create figure
    n_models = len(models_info)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f"Sample {idx}: Depth Prediction Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Plot each model
    for ax, model_info, preds in zip(axes, models_info, all_predictions):
        name = model_info["name"]
        checkpoint = model_info["checkpoint"]

        if len(preds) > 0:
            n_objects = len(preds)
            x = np.arange(n_objects)
            width = 0.35

            # Use distinct colors
            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            bar_colors = [colors[i % len(colors)] for i in range(n_objects)]

            # Plot bars
            bars_pred = ax.bar(
                x - width / 2,
                preds,
                width,
                label="Predicted",
                color=bar_colors,
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
            )

            bars_gt = ax.bar(
                x + width / 2,
                gt_depths[:n_objects],
                width,
                label="Ground Truth",
                color=bar_colors,
                alpha=0.4,
                edgecolor="black",
                linewidth=2,
                hatch="///",
            )

            # Add error annotations
            for i, (pred, gt) in enumerate(zip(preds, gt_depths[:n_objects])):
                error = abs(pred - gt)
                y_pos = max(pred, gt) + max(gt_depths) * 0.05

                color = (
                    "darkgreen" if error < 2.0 else "orange" if error < 5.0 else "red"
                )
                ax.text(
                    i,
                    y_pos,
                    f"Δ{error:.1f}m",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                    color=color,
                )

                # Object number
                ax.text(
                    i,
                    -max(gt_depths) * 0.08,
                    f"#{i+1}",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                    color=bar_colors[i],
                )

            # Compute stats
            errors = np.abs(np.array(preds) - gt_depths[:n_objects])
            avg_error = np.mean(errors)
            max_error = np.max(errors)

            # Model info
            epoch = checkpoint.get("epoch", "?")
            overall_ate = checkpoint.get(
                "best_metric", checkpoint.get("metrics", {}).get("ATE", "?")
            )

            # Title
            title = f"{name}\n"
            title += f"Epoch {epoch}\n"
            title += f"Overall ATE: {overall_ate:.2f}m\n"
            title += f"This Sample: Avg={avg_error:.2f}m, Max={max_error:.2f}m"
            ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

            # Styling
            ax.set_xlabel("Object Number", fontsize=12, fontweight="bold")
            ax.set_ylabel("Depth (meters)", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_ylim(bottom=-max(gt_depths) * 0.1)

            # Thicker spines
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        else:
            ax.text(
                0.5,
                0.5,
                "No Detections",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="red",
                fontweight="bold",
            )
            ax.set_title(
                f"{name}\n(No Objects Detected)", fontsize=13, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  ✓ Saved bar comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="+", required=True)
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help="Config files for each model (same order as checkpoints)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Default config if --configs not provided",
    )
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 20, 30, 40])
    parser.add_argument("--output", type=str, default="./bar_comparisons")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Use per-model configs if provided, otherwise use default for all
    if args.configs:
        if len(args.configs) != len(args.checkpoints):
            print("ERROR: Number of configs must match number of checkpoints!")
            return
        configs = args.configs
    else:
        configs = [args.config] * len(args.checkpoints)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("BAR CHART COMPARISON")
    print("=" * 70)

    # Load models
    models_info = []
    for ckpt_path, name, cfg in zip(args.checkpoints, args.names, configs):
        print(f"\nLoading {name} with config {cfg}...")
        model, checkpoint = load_model(ckpt_path, cfg, args.device)
        models_info.append({"model": model, "name": name, "checkpoint": checkpoint})
        print(
            f"  ✓ Epoch {checkpoint.get('epoch', '?')}, ATE: {checkpoint.get('best_metric', '?'):.2f}m"
        )

    # Load dataset
    print("\nLoading validation dataset...")
    dataset = NuScenesDepthDataset(
        nuscenes_root="./data/nuscenes", split="val", version="v1.0-trainval"
    )
    print(f"✓ Loaded {len(dataset)} samples")

    # Generate comparisons
    print(f"\nGenerating bar charts for {len(args.samples)} samples...")
    for idx in args.samples:
        if idx >= len(dataset):
            continue

        output_path = os.path.join(args.output, f"bars_sample_{idx:04d}.png")
        print(f"\nSample {idx}...")
        create_bar_comparison(models_info, dataset, idx, output_path, args.device)

    print("\n" + "=" * 70)
    print(f"✓ All bar charts saved to {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
