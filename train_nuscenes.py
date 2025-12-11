import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_real import build_model
from nuscenes_loader import NuScenesDepthDataset, collate_fn


class DepthRefinementLoss(nn.Module):
    """Combined loss for depth refinement"""

    def __init__(self, weights: dict):
        super().__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        """Compute losses"""
        losses = {}

        # Collect predictions and targets
        all_pred_centers = []
        all_gt_centers = []
        all_pred_sizes = []
        all_gt_sizes = []
        all_pred_depths = []
        all_gt_depths = []

        batch_size = len(predictions["center_3d"])

        for i in range(batch_size):
            if len(predictions["center_3d"][i]) > 0 and len(targets["boxes_3d"][i]) > 0:
                pred_centers = predictions["center_3d"][i]
                gt_boxes = targets["boxes_3d"][i]
                gt_centers = gt_boxes[:, :3]

                # Match predictions to GT (nearest neighbor)
                for j, pred_center in enumerate(pred_centers):
                    distances = torch.norm(gt_centers - pred_center, dim=1)
                    closest_idx = distances.argmin()

                    all_pred_centers.append(pred_center)
                    all_gt_centers.append(gt_centers[closest_idx])

                    if j < len(predictions["size"][i]):
                        pred_size = predictions["size"][i][j]
                        gt_size = gt_boxes[closest_idx, 3:6]
                        all_pred_sizes.append(pred_size)
                        all_gt_sizes.append(gt_size)

                    all_pred_depths.append(pred_center[2])
                    all_gt_depths.append(gt_centers[closest_idx, 2])

        # Compute losses
        if len(all_pred_centers) > 0:
            pred_centers_t = torch.stack(all_pred_centers)
            gt_centers_t = torch.stack(all_gt_centers)

            translation_error = torch.norm(pred_centers_t - gt_centers_t, dim=1).mean()
            losses["translation"] = translation_error

            if len(all_pred_sizes) > 0:
                pred_sizes_t = torch.stack(all_pred_sizes)
                gt_sizes_t = torch.stack(all_gt_sizes)
                scale_error = F.smooth_l1_loss(pred_sizes_t, gt_sizes_t)
                losses["scale"] = scale_error
            else:
                losses["scale"] = torch.tensor(0.0, device=pred_centers_t.device)

            if len(all_pred_depths) > 0:
                pred_depths_t = torch.stack(all_pred_depths)
                gt_depths_t = torch.stack(all_gt_depths)
                depth_error = F.smooth_l1_loss(pred_depths_t, gt_depths_t)
                losses["depth"] = depth_error
            else:
                losses["depth"] = torch.tensor(0.0, device=pred_centers_t.device)
        else:
            device = (
                predictions["center_3d"][0].device
                if len(predictions["center_3d"]) > 0
                else torch.device("cpu")
            )
            losses["translation"] = torch.tensor(0.0, device=device)
            losses["scale"] = torch.tensor(0.0, device=device)
            losses["depth"] = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
        losses["total"] = total_loss

        return total_loss, losses


class Trainer:
    """Main training class"""

    def __init__(self, config: dict, args):
        self.config = config
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.set_seed(config.get("seed", 42))

        # Setup directories first (needed for resume)
        self.setup_output_dir()

        # Build model
        print("\nBuilding model...")
        self.model = build_model(config)
        self.model.to(self.device)
        self.print_model_info()

        # Setup data
        print("\nLoading dataset...")
        self.setup_data()

        # Setup training
        self.setup_training()

        # Setup logging
        self.setup_logging()

        # Tracking
        self.best_metric = float("inf")
        self.epochs_without_improvement = 0
        self.global_step = 0
        self.start_epoch = 0

        # Load checkpoint if resuming (must be after model/optimizer setup)
        if args.resume:
            self.load_checkpoint(args.resume)

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup_output_dir(self):
        log_config = self.config["logging"]

        # If resuming, use existing directory
        if self.args.resume:
            # Extract directory from checkpoint path
            checkpoint_path = Path(self.args.resume)
            self.output_dir = checkpoint_path.parent.parent
            self.checkpoint_dir = self.output_dir / "checkpoints"
            self.log_dir = self.output_dir / "logs"
            print(f"Resuming in existing directory: {self.output_dir}")
        else:
            # Create new directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{log_config['experiment_name']}_{timestamp}"

            self.output_dir = Path(log_config["output_dir"]) / exp_name
            self.checkpoint_dir = self.output_dir / "checkpoints"
            self.log_dir = self.output_dir / "logs"

            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.log_dir.mkdir(exist_ok=True)

            with open(self.output_dir / "config.yaml", "w") as f:
                yaml.dump(self.config, f)

            print(f"Output directory: {self.output_dir}")

    def print_model_info(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        print(f"\nModel Parameters:")
        print(f"  Trainable: {trainable:,}")
        print(f"  Total: {total:,}")
        print(f"  Percentage trainable: {100 * trainable / total:.2f}%")

    def setup_data(self):
        dataset_config = self.config["dataset"]

        # Determine version
        version = "v1.0-trainval"
        if "mini" in dataset_config["nuscenes_root"].lower():
            version = "v1.0-mini"

        # Load datasets
        self.train_dataset = NuScenesDepthDataset(
            nuscenes_root=dataset_config["nuscenes_root"],
            split="train",
            version=version,
        )

        self.val_dataset = NuScenesDepthDataset(
            nuscenes_root=dataset_config["nuscenes_root"], split="val", version=version
        )

        # Test mode
        if self.args.test:
            print("\n⚠️  TEST MODE: Using only 32 samples")
            self.train_dataset.samples = self.train_dataset.samples[:32]
            self.val_dataset.samples = self.val_dataset.samples[:16]

        # Dataloaders
        train_config = self.config["training"]
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=train_config["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
            num_workers=train_config["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
        print(f"  Batch size: {train_config['batch_size']}")

    def setup_training(self):
        train_config = self.config["training"]

        if train_config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=train_config["learning_rate"],
                weight_decay=train_config["weight_decay"],
                betas=train_config.get("betas", [0.9, 0.999]),
            )

        self.scheduler = None
        if train_config["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config["num_epochs"],
                eta_min=train_config.get("min_lr", 0.000001),
            )

        self.criterion = DepthRefinementLoss(weights=train_config["loss_weights"])

        self.warmup_epochs = train_config.get("warmup_epochs", 0)

    def setup_logging(self):
        if self.config["logging"]["use_tensorboard"]:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

    def train_epoch(self, epoch: int):
        self.model.train()

        if epoch < self.warmup_epochs:
            lr = (
                self.config["training"]["learning_rate"]
                * (epoch + 1)
                / self.warmup_epochs
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        epoch_losses = {}
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}",
        )

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            boxes_2d = [b.to(self.device) for b in batch["boxes_2d"]]
            boxes_3d = batch["boxes_3d"]
            camera_intrinsics = batch["camera_internal"].to(self.device)

            predictions = self.model(images, boxes_2d, camera_intrinsics)

            targets = {"boxes_3d": boxes_3d}
            loss, losses = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()

            if self.config["training"].get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["grad_clip"]
                )

            self.optimizer.step()

            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v.item())

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            if self.writer and batch_idx % self.config["logging"]["log_every"] == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}", v.item(), self.global_step)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )

            self.global_step += 1

        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()

        val_losses = {}
        all_translation_errors = []

        pbar = tqdm(self.val_loader, desc="Validating")

        for batch in pbar:
            images = batch["image"].to(self.device)
            boxes_2d = [b.to(self.device) for b in batch["boxes_2d"]]
            boxes_3d = batch["boxes_3d"]
            camera_intrinsics = batch["camera_internal"].to(self.device)

            predictions = self.model(images, boxes_2d, camera_intrinsics)

            targets = {"boxes_3d": boxes_3d}
            loss, losses = self.criterion(predictions, targets)

            for k, v in losses.items():
                if k not in val_losses:
                    val_losses[k] = []
                val_losses[k].append(v.item())

            for i in range(len(predictions["center_3d"])):
                if len(predictions["center_3d"][i]) > 0 and len(boxes_3d[i]) > 0:
                    pred_center = predictions["center_3d"][i]
                    gt_center = boxes_3d[i][:, :3]
                    distances = torch.norm(
                        pred_center.unsqueeze(1) - gt_center.unsqueeze(0), dim=2
                    )
                    min_distances = distances.min(dim=1)[0]
                    all_translation_errors.extend(min_distances.cpu().numpy())

            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}

        metrics = {
            "ATE": np.mean(all_translation_errors) if all_translation_errors else 0.0,
        }

        if self.writer:
            for k, v in avg_losses.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            for k, v in metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

        return avg_losses, metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "metrics": metrics,
            "best_metric": self.best_metric,
            "global_step": self.global_step,
            "config": self.config,
        }

        if self.config["logging"]["save_last"]:
            torch.save(checkpoint, self.checkpoint_dir / "last.pth")

        if is_best and self.config["logging"]["save_best"]:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")

        if epoch % self.config["logging"]["checkpoint_every"] == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch}.pth")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        # PyTorch 2.6+ requires weights_only=False for full checkpoints
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Load model and optimizer states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        self.start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
        self.best_metric = checkpoint.get("best_metric", float("inf"))
        self.global_step = checkpoint.get("global_step", 0)

        print(f"✓ Resumed from epoch {checkpoint['epoch']}")
        print(f"  Best ATE so far: {self.best_metric:.4f}")
        print(
            f"  Continuing from epoch {self.start_epoch}/{self.config['training']['num_epochs']}"
        )

    def train(self):
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        for epoch in range(self.start_epoch, self.config["training"]["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            print("-" * 60)

            train_losses = self.train_epoch(epoch)
            print(f"Train Loss: {train_losses['total']:.4f}")

            # MODIFIED: Always compute ATE after every epoch
            val_losses, metrics = self.validate(epoch)
            print(f"Val Loss: {val_losses['total']:.4f}")
            print(f"Metrics: ATE={metrics['ATE']:.4f}")

            current_metric = metrics["ATE"]
            is_best = current_metric < self.best_metric

            if is_best:
                print(
                    f"🎉 New best ATE: {current_metric:.4f} (prev: {self.best_metric:.4f})"
                )
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, metrics, is_best)

            patience = self.config["training"]["patience"]
            if self.epochs_without_improvement >= patience:
                print(f"\n⚠️  Early stopping after {epoch+1} epochs")
                break

            if self.scheduler and epoch >= self.warmup_epochs:
                self.scheduler.step()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best ATE: {self.best_metric:.4f}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print("=" * 60)

        if self.writer:
            self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
