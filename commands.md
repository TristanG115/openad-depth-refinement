# OpenAD Depth Refinement - Quick Commands

## Training Commands

### 1. Train Main Model (with residuals, trainable depth)
```bash
python train_nuscenes.py --config config.yaml
```

### 2. Train Ablation Model (no residuals)
```bash
python train_nuscenes.py --config config_ablation.yaml
```

### 3. Train Frozen Baseline (frozen depth network)
```bash
python train_nuscenes.py --config config_frozen.yaml
```

### 4. Resume Training from Checkpoint
```bash
python train_nuscenes.py --config config.yaml --resume ./outputs/depth_refinement_TIMESTAMP/checkpoints/last.pth
```

### 5. Test Mode (quick training with small dataset)
```bash
python train_nuscenes.py --config config.yaml --test
```

##  Directory Structure

After training and evaluation, you'll have:
```
openad-depth-refinement/
├── data/
│   └── nuscenes/          # Dataset (ignored by git)
├── outputs/               # Model checkpoints (ignored by git)
│   ├── depth_refinement_TIMESTAMP/
│   │   ├── checkpoints/
│   │   │   ├── best.pth
│   │   │   ├── last.pth
│   │   │   └── epoch_*.pth
│   │   ├── logs/          # Tensorboard logs
│   │   └── config.yaml
│   ├── frozen_depth_baseline_TIMESTAMP/
│   └── ablation_no_residuals_TIMESTAMP/
├── evaluation_results/    # Comparison plots (ignored by git)
│   ├── comparison_plot.png
│   ├── error_distributions.png
│   ├── results_table.tex
│   ├── results_table.md
│   └── results.json
├── visualizations_comparison/  # Multi-model viz (ignored by git)
├── bar_comparisons/       # Bar charts (ignored by git)
└── predictions/           # Individual predictions (ignored by git)
```

---

## 📈 Monitoring Training

### View Tensorboard Logs
```bash
tensorboard --logdir ./outputs/depth_refinement_TIMESTAMP/logs
```

Then open browser to: http://localhost:6006

---

## Quick commands

1. **Train all three models:**
   ```bash
   python train_nuscenes.py --config config.yaml
   python train_nuscenes.py --config config_frozen.yaml
   python train_nuscenes.py --config config_ablation.yaml
   ```

2. **Wait for training to complete** (check early stopping or max epochs)

3. **Evaluate all models:**
   ```bash
   python evaluate_and_visualize.py \
       --experiments ./outputs/depth_refinement_* ./outputs/frozen_* ./outputs/ablation_* \
       --names "Ours" "Frozen" "Ablation" \
       --output ./evaluation_results
   ```

4. **Generate visualizations:**
   ```bash
   # Multi-model comparison
   python visualize_predictions.py --checkpoints [paths] --names [names] --samples 0 10 20 30 40

   # Clean bar charts
   python visualize_bars.py --checkpoints [paths] --names [names] --samples 0 10 20 30 40

   # Individual model predictions
   python visualize_on_image.py --checkpoint [path] --name "Ours" --samples 0 10 20 30 40
   ```

5. **Check results:**
   - Tables: `./evaluation_results/results_table.md`
   - Plots: `./evaluation_results/comparison_plot.png`
   - Visualizations: `./visualizations_comparison/`, `./bar_comparisons/`, `./predictions/`

---

## Saving Outputs for Git

Since data and outputs are git-ignored, to share results:

1. **Keep**: Configuration files, Python scripts, requirements.txt
2. **Share separately**:
   - Trained checkpoints (upload to cloud storage)
   - Evaluation results (can commit final plots/tables if small)
   - Dataset (provide download link)

---

**Last Updated:** December 2024
**Project:** OpenAD Depth Refinement with nuScenes
