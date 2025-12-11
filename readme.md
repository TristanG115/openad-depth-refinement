# OpenAD Depth Refinement

> **Trainable depth refinement for 3D object detection from 2D bounding boxes**

This project implements a depth refinement network that improves 3D object localization by learning to predict accurate depth from 2D bounding boxes on autonomous driving datasets (nuScenes).


---

##  What Does This Do?

Given a 2D bounding box around an object in an image, this model:
1. **Predicts accurate depth** to that object
2. **Converts 2D boxes to 3D boxes** in camera coordinates
3. **Learns from data** instead of using fixed depth estimation

**Key Approach:** Unlike frozen approaches, our model has a **trainable depth network** that learns from the dataset, leading to better 3D localization.

## Project Structure

```
openad-depth-refinement/
├── train_nuscenes.py           # Main training script
├── model_real.py               # Depth refinement model
├── nuscenes_loader.py          # Data loader
├── evaluate_and_visualize.py  # Evaluation & comparison
├── visualize_predictions.py    # Multi-model visualization
├── visualize_bars.py           # Bar chart comparisons
├── visualize_on_image.py       # Single model predictions
├── config.yaml                 # Main model config
├── config_frozen.yaml          # Frozen baseline config
├── config_ablation.yaml        # Ablation config
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── COMMANDS.md                 # Detailed command reference
└── README.md                   # This file
```

### Training

- **Loss Functions:**
  - Translation Error (ATE)
  - Scale Error (ASE)
  - Direct Depth Supervision
- **Optimizer:** Adam with cosine learning rate scheduling
- **Early Stopping:** Stops if no improvement for 12 epochs
- **Metrics:** ATE (Average Translation Error) computed after every epoch

## Monitoring Training

### View Live Progress

Training automatically logs to TensorBoard:

```bash
tensorboard --logdir ./outputs/depth_refinement_TIMESTAMP/logs
```

Open http://localhost:6006 in your browser.

### Check Console Output

Training prints progress after each epoch:

```
Epoch 15/75
------------------------------------------------------------
Train Loss: 2.3456
Val Loss: 2.8901
Metrics: ATE=3.2534
🎉 New best ATE: 3.2534 (prev: 3.4102)
```

---


## Additional Resources

- **Detailed Commands:** See `COMMANDS.md` for all commands with examples
- **nuScenes Dataset:** https://www.nuscenes.org/
- **PyTorch Docs:** https://pytorch.org/docs/

---

## Contributing

This is a research project. Feel free to:
- Report issues
- Suggest improvements
- Extend to other datasets

---

## Acknowledgments

- **nuScenes Dataset:** Provided by Motional
- **OpenAD Framework:** For the depth refinement approach
- Built with PyTorch

---

## Contact

tdgoodin@purdu.edu

** Good luck training! *
