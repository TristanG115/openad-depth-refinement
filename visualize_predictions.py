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


def visualize_comparison(models_info, dataset, idx, output_path, device="cpu"):
    """
    Visualize predictions vs ground truth for multiple models

    models_info: list of dicts with 'model', 'name', 'checkpoint' keys
    """

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

    # Create figure: 1 image column + N model columns
    n_models = len(models_info)
    fig = plt.figure(figsize=(6 + n_models * 5, 8))

    # Create grid: top row for image, bottom row for depth comparisons
    gs = fig.add_gridspec(2, 1 + n_models, height_ratios=[1, 1], hspace=0.3)

    # --- Image with labeled boxes (spans both rows) ---
    ax_img = fig.add_subplot(gs[:, 0])
    img_np = sample["image"].permute(1, 2, 0).numpy()
    ax_img.imshow(img_np)
    ax_img.set_title(
        f"Sample {idx}: Ground Truth Boxes", fontsize=14, fontweight="bold"
    )
    ax_img.axis("off")

    # Draw boxes with numbers
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(boxes_2d_np))))

    for i, box in enumerate(boxes_2d_np):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]

        # Draw box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=color, facecolor="none"
        )
        ax_img.add_patch(rect)

        # Add label with number
        label_bg = FancyBboxPatch(
            (x1, y1 - 25),
            35,
            25,
            boxstyle="round,pad=3",
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            alpha=0.9,
        )
        ax_img.add_patch(label_bg)

        ax_img.text(
            x1 + 17,
            y1 - 12,
            f"{i+1}",
            fontsize=14,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )

        # Add GT depth annotation
        gt_depth = gt_depths[i]
        ax_img.text(
            x2 + 5,
            (y1 + y2) / 2,
            f"GT: {gt_depth:.1f}m",
            fontsize=10,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # --- Depth comparisons for each model ---
    for model_idx, model_info in enumerate(models_info):
        ax = fig.add_subplot(gs[1, 1 + model_idx])

        model = model_info["model"]
        name = model_info["name"]
        checkpoint = model_info["checkpoint"]

        # Get predictions
        with torch.no_grad():
            predictions = model(image, boxes_2d, camera_intrinsics)

        pred_centers = predictions["center_3d"][0].cpu().numpy()
        pred_depths = pred_centers[:, 2] if len(pred_centers) > 0 else []

        # Match predictions to GT
        if len(pred_depths) > 0 and len(gt_depths) > 0:
            matched_pred = []
            matched_gt = []
            matched_indices = []

            for i, pred_center in enumerate(pred_centers):
                distances = np.linalg.norm(gt_centers - pred_center, axis=1)
                closest_idx = distances.argmin()

                matched_pred.append(pred_depths[i])
                matched_gt.append(gt_depths[closest_idx])
                matched_indices.append(closest_idx)

            # Plot
            x = np.arange(len(matched_pred))
            width = 0.35

            # Use same colors as boxes
            bar_colors = [colors[idx % len(colors)] for idx in matched_indices]

            bars_pred = ax.bar(
                x - width / 2,
                matched_pred,
                width,
                label="Predicted",
                color=bar_colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add error annotations
            for i, (pred, gt, idx) in enumerate(
                zip(matched_pred, matched_gt, matched_indices)
            ):
                error = abs(pred - gt)
                y_pos = max(pred, gt) + 1

                # Error text
                ax.text(
                    i,
                    y_pos,
                    f"Δ{error:.1f}m",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color="red" if error > 2.0 else "darkgreen",
                )

                # Box number
                ax.text(
                    i,
                    -2,
                    f"#{idx+1}",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    color=colors[idx % len(colors)],
                )

            # Compute metrics
            errors = np.abs(np.array(matched_pred) - np.array(matched_gt))
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)

            # Model info from checkpoint
            epoch = checkpoint.get("epoch", "?")
            best_ate = checkpoint.get(
                "best_metric", checkpoint.get("metrics", {}).get("ATE", "?")
            )

            # Title with model name and metrics
            title = f"{name}\n"
            title += f"Epoch {epoch} | Overall ATE: {best_ate:.2f}m\n"
            title += f"This Sample: Avg={avg_error:.2f}m, Max={max_error:.2f}m"
            ax.set_title(title, fontsize=11, fontweight="bold")

            ax.set_xlabel("Object Index", fontsize=10)
            ax.set_ylabel("Depth (meters)", fontsize=10)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_ylim(bottom=-3)

        else:
            ax.text(
                0.5,
                0.5,
                "No objects detected",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            ax.set_title(f"{name}\nNo Detections", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model checkpoints (space-separated)",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        required=True,
        help='Model names (e.g., "Ours" "Ablation" "Frozen")',
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30, 40],
        help="Sample indices to visualize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./visualizations_comparison",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if len(args.checkpoints) != len(args.names):
        print("ERROR: Number of checkpoints must match number of names!")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("MULTI-MODEL COMPARISON VISUALIZATION")
    print("=" * 70)

    # Load all models
    models_info = []
    for ckpt_path, name in zip(args.checkpoints, args.names):
        print(f"\nLoading {name} from {ckpt_path}...")
        model, checkpoint = load_model(ckpt_path, args.config, args.device)
        models_info.append({"model": model, "name": name, "checkpoint": checkpoint})
        print(
            f"  ✓ Loaded (Epoch {checkpoint.get('epoch', '?')}, "
            f"ATE: {checkpoint.get('best_metric', '?'):.2f}m)"
        )

    # Load validation dataset
    print("\nLoading validation dataset...")
    dataset = NuScenesDepthDataset(
        nuscenes_root="./data/nuscenes", split="val", version="v1.0-trainval"
    )
    print(f"✓ Loaded {len(dataset)} validation samples")

    # Visualize samples
    print(f"\nGenerating comparisons for {len(args.samples)} samples...")
    for idx in args.samples:
        if idx >= len(dataset):
            print(f"  ⚠ Skipping sample {idx} (out of range)")
            continue

        output_path = os.path.join(args.output, f"comparison_sample_{idx:04d}.png")
        print(f"\nProcessing sample {idx}...")
        visualize_comparison(models_info, dataset, idx, output_path, args.device)

    print("\n" + "=" * 70)
    print(f"✓ All comparisons saved to {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
