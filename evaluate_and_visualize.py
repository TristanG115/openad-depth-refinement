import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_real import build_model
from nuscenes_loader import NuScenesDepthDataset, collate_fn


def load_model_checkpoint(checkpoint_path, config_path):
    """Load a trained model"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, checkpoint


@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu"):
    """Evaluate model on dataset"""
    model.to(device)
    model.eval()

    all_errors = []
    depth_errors = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        boxes_2d = [b.to(device) for b in batch["boxes_2d"]]
        boxes_3d = batch["boxes_3d"]
        camera_intrinsics = batch["camera_internal"].to(device)

        predictions = model(images, boxes_2d, camera_intrinsics)

        # Compute errors
        for i in range(len(predictions["center_3d"])):
            if len(predictions["center_3d"][i]) > 0 and len(boxes_3d[i]) > 0:
                pred_center = predictions["center_3d"][i]
                gt_center = boxes_3d[i][:, :3].to(device)

                # Translation errors (ATE)
                distances = torch.norm(
                    pred_center.unsqueeze(1) - gt_center.unsqueeze(0), dim=2
                )
                min_distances = distances.min(dim=1)[0]
                all_errors.extend(min_distances.cpu().numpy())

                # Depth errors specifically
                pred_depths = pred_center[:, 2]
                gt_depths = gt_center[:, 2]
                depth_dist = torch.abs(
                    pred_depths.unsqueeze(1) - gt_depths.unsqueeze(0)
                )
                min_depth_errors = depth_dist.min(dim=1)[0]
                depth_errors.extend(min_depth_errors.cpu().numpy())

    results = {
        "ATE": float(np.mean(all_errors)) if all_errors else 0.0,
        "ATE_std": float(np.std(all_errors)) if all_errors else 0.0,
        "depth_error": float(np.mean(depth_errors)) if depth_errors else 0.0,
        "depth_error_std": float(np.std(depth_errors)) if depth_errors else 0.0,
        "num_predictions": len(all_errors),
    }

    return results, all_errors, depth_errors


def plot_training_curves(experiments, save_dir):
    """Plot training curves from tensorboard logs"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for exp_name, exp_data in experiments.items():
        output_dir = exp_data["output_dir"]

        # Try to load metrics from checkpoint
        checkpoint_path = os.path.join(output_dir, "checkpoints", "best.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            best_ate = checkpoint.get("metrics", {}).get("ATE", 0)
            best_epoch = checkpoint.get("epoch", 0)

            exp_data["best_ate"] = best_ate
            exp_data["best_epoch"] = best_epoch

    # Plot comparison
    names = list(experiments.keys())
    ates = [experiments[name].get("best_ate", 0) for name in names]
    epochs = [experiments[name].get("best_epoch", 0) for name in names]

    # Bar plot of final ATEs
    axes[0].bar(range(len(names)), ates, color=["#2ecc71", "#e74c3c", "#3498db"])
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_ylabel("ATE (meters)")
    axes[0].set_title("Average Translation Error (Lower is Better)")
    axes[0].grid(axis="y", alpha=0.3)

    # Add values on bars
    for i, (ate, epoch) in enumerate(zip(ates, epochs)):
        axes[0].text(
            i,
            ate + 0.1,
            f"{ate:.2f}m\n(epoch {epoch})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Improvement percentages
    if len(ates) >= 2:
        baseline_ate = ates[1]  # Assume frozen is baseline
        improvements = [((baseline_ate - ate) / baseline_ate) * 100 for ate in ates]

        axes[1].bar(
            range(len(names)), improvements, color=["#2ecc71", "#e74c3c", "#3498db"]
        )
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha="right")
        axes[1].set_ylabel("Improvement (%)")
        axes[1].set_title("Improvement Over Frozen Baseline")
        axes[1].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        axes[1].grid(axis="y", alpha=0.3)

        for i, imp in enumerate(improvements):
            axes[1].text(
                i, imp + 1, f"{imp:.1f}%", ha="center", va="bottom", fontsize=9
            )

    # Training time / efficiency
    axes[2].text(
        0.5,
        0.5,
        "Results Summary\n\n"
        + "\n".join([f"{name}: {ate:.2f}m" for name, ate in zip(names, ates)]),
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[2].axis("off")
    axes[2].set_title("Summary")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "comparison_plot.png"), dpi=300, bbox_inches="tight"
    )
    print(f"Saved comparison plot to {save_dir}/comparison_plot.png")
    plt.close()


def plot_error_distributions(all_results, save_dir):
    """Plot error distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    # Translation error distribution
    for i, (name, data) in enumerate(all_results.items()):
        if "errors" in data:
            axes[0].hist(
                data["errors"],
                bins=50,
                alpha=0.5,
                label=name,
                color=colors[i % len(colors)],
                edgecolor="black",
            )

    axes[0].set_xlabel("Translation Error (meters)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Translation Errors")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Depth error distribution
    for i, (name, data) in enumerate(all_results.items()):
        if "depth_errors" in data:
            axes[1].hist(
                data["depth_errors"],
                bins=50,
                alpha=0.5,
                label=name,
                color=colors[i % len(colors)],
                edgecolor="black",
            )

    axes[1].set_xlabel("Depth Error (meters)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Depth Errors")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "error_distributions.png"), dpi=300, bbox_inches="tight"
    )
    print(f"Saved error distributions to {save_dir}/error_distributions.png")
    plt.close()


def generate_results_table(all_results, save_dir):
    """Generate LaTeX table of results"""

    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\caption{Comparison of Depth Refinement Approaches}\n"
    table += "\\begin{tabular}{lccc}\n"
    table += "\\hline\n"
    table += "Method & ATE (m) $\\downarrow$ & Depth Error (m) $\\downarrow$ & Improvement (\\%) $\\uparrow$ \\\\\n"
    table += "\\hline\n"

    # Get baseline ATE (frozen depth)
    baseline_ate = None
    for name, data in all_results.items():
        if "frozen" in name.lower():
            baseline_ate = data["results"]["ATE"]
            break

    # Sort by ATE
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["results"]["ATE"])

    for name, data in sorted_results:
        ate = data["results"]["ATE"]
        ate_std = data["results"]["ATE_std"]
        depth_err = data["results"]["depth_error"]

        improvement = 0.0
        if baseline_ate and baseline_ate > 0:
            improvement = ((baseline_ate - ate) / baseline_ate) * 100

        table += f"{name} & {ate:.2f} $\\pm$ {ate_std:.2f} & {depth_err:.2f} & {improvement:+.1f}\\% \\\\\n"

    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"

    # Save table
    with open(os.path.join(save_dir, "results_table.tex"), "w") as f:
        f.write(table)
    print(f"Saved LaTeX table to {save_dir}/results_table.tex")

    # Also save as markdown
    md_table = "| Method | ATE (m) | Depth Error (m) | Improvement (%) |\n"
    md_table += "|--------|---------|-----------------|------------------|\n"

    for name, data in sorted_results:
        ate = data["results"]["ATE"]
        ate_std = data["results"]["ATE_std"]
        depth_err = data["results"]["depth_error"]

        improvement = 0.0
        if baseline_ate and baseline_ate > 0:
            improvement = ((baseline_ate - ate) / baseline_ate) * 100

        md_table += f"| {name} | {ate:.2f} ± {ate_std:.2f} | {depth_err:.2f} | {improvement:+.1f}% |\n"

    with open(os.path.join(save_dir, "results_table.md"), "w") as f:
        f.write(md_table)
    print(f"Saved markdown table to {save_dir}/results_table.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments", nargs="+", required=True, help="List of experiment directories"
    )
    parser.add_argument(
        "--names", nargs="+", required=True, help="Names for each experiment"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Setup dataset
    print("Loading validation dataset...")
    val_dataset = NuScenesDepthDataset(
        nuscenes_root="./data/nuscenes", split="val", version="v1.0-trainval"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, num_workers=0, collate_fn=collate_fn
    )

    # Evaluate each experiment
    all_results = {}
    experiments_data = {}

    for exp_dir, name in zip(args.experiments, args.names):
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        checkpoint_path = os.path.join(exp_dir, "checkpoints", "best.pth")
        config_path = os.path.join(exp_dir, "config.yaml")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            continue

        # Load model
        model, config, checkpoint = load_model_checkpoint(checkpoint_path, config_path)

        # Evaluate
        results, errors, depth_errors = evaluate_model(model, val_loader, args.device)

        print(f"\nResults for {name}:")
        print(f"  ATE: {results['ATE']:.4f} ± {results['ATE_std']:.4f} meters")
        print(
            f"  Depth Error: {results['depth_error']:.4f} ± {results['depth_error_std']:.4f} meters"
        )
        print(f"  Predictions: {results['num_predictions']}")

        all_results[name] = {
            "results": results,
            "errors": errors,
            "depth_errors": depth_errors,
            "checkpoint": checkpoint,
        }

        experiments_data[name] = {
            "output_dir": exp_dir,
            "best_ate": results["ATE"],
            "best_epoch": checkpoint.get("epoch", 0),
        }

    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")

    plot_training_curves(experiments_data, args.output)
    plot_error_distributions(all_results, args.output)
    generate_results_table(all_results, args.output)

    # Save results as JSON
    results_json = {
        name: {
            "ATE": data["results"]["ATE"],
            "ATE_std": data["results"]["ATE_std"],
            "depth_error": data["results"]["depth_error"],
            "num_predictions": data["results"]["num_predictions"],
        }
        for name, data in all_results.items()
    }

    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ All results saved to {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
