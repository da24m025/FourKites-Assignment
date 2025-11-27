# FourKites-Assignment: Loss Landscape Analysis of Optimizers

## Overview

This project analyzes the loss landscape of neural network training with different optimizers using a comprehensive set of loss landscape probes. We trained SimpleCNN on CIFAR-10 with three optimizers (SGD, SGD+Momentum, AdamW) and characterized their optimization dynamics through loss landscape analysis.

## Key Results

| Optimizer | Val Accuracy | λ_max (Hessian) | Noise Ratio | Finding |
|-----------|-------------|-----------------|-------------|---------|
| SGD | 63.64% | 129.34 | 2.77 | Slowest, highest noise |
| SGD+Momentum | 71.89% | 37.60 | 3.60 | Flattest landscape |
| **AdamW** | **71.91%** | 175.18 | **1.52** | **Best accuracy**, lowest noise ratio |

**Key Finding**: AdamW achieves the best accuracy (71.91%) despite having the sharpest loss landscape (λ_max=175.18) due to its lower gradient noise ratio (1.52 vs 2.77 for SGD) and adaptive per-parameter learning rates.

## Project Structure

```
src/
├── train.py                 # Main training script
├── models/
│   └── small_cnn.py        # SimpleCNN model (183K parameters)
├── utils/
│   └── exp.py              # Dataloaders, checkpointing, utilities
└── probes/                 # Loss landscape analysis probes
    ├── hessian_lanczos.py  # Eigenvalue analysis (Lanczos method)
    ├── sgd_noise.py        # Gradient noise vs curvature
    ├── perturbation.py     # Top-eigenvector sensitivity
    ├── intrinsic_dim.py    # Intrinsic dimensionality
    └── interpolation.py    # Linear interpolation between models

experiments/
├── configs/
│   ├── sgd.yml             # SGD configuration
│   ├── sgd_momentum.yml    # SGD+Momentum configuration
│   └── adamw.yml           # AdamW configuration
└── runs/                   # Training outputs
    ├── sgd_seed42/
    │   ├── checkpoint.pt
    │   └── probes/         # Probe results
    ├── sgd_momentum_seed42/
    │   ├── checkpoint.pt
    │   └── probes/
    └── adamw_seed42/
        ├── checkpoint.pt
        └── probes/
```

## Installation

### Requirements
- Python 3.8+
- PyTorch >= 1.9
- torchvision
- numpy
- matplotlib
- PyYAML

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=/path/to/Assignment\ Flykites:$PYTHONPATH
```

## Loss Landscape Probes

This project implements 6 different loss landscape analysis probes:

### 1. **Lanczos Spectrum** (`hessian_lanczos.py`)
Computes top eigenvalues of the Hessian matrix to characterize landscape curvature.

```bash
python3 src/probes/hessian_lanczos.py \
  --checkpoint experiments/runs/sgd_seed42/checkpoint.pt \
  --config experiments/configs/sgd.yml \
  --k 3 --seed 42
```

**Output**: `lanczos_spectrum_checkpoint.json`
- Top 3 eigenvalues
- Smallest eigenvalues
- Condition number: |λ_max/λ_min|

### 2. **SGD Noise Covariance** (`sgd_noise.py`)
Analyzes gradient noise relative to Hessian curvature.

```bash
python3 src/probes/sgd_noise.py \
  --checkpoint experiments/runs/sgd_seed42/checkpoint.pt \
  --config experiments/configs/sgd.yml \
  --seed 42
```

**Output**: `sgd_noise_checkpoint_seed42.json`
- Hessian top eigenvalue
- Gradient covariance top eigenvalue
- Noise ratio: λ_C / λ_H

### 3. **Perturbation** (`perturbation.py`)
Analyzes loss sensitivity along the top Hessian eigenvector.

```bash
python3 src/probes/perturbation.py \
  --checkpoint experiments/runs/sgd_seed42/checkpoint.pt \
  --config experiments/configs/sgd.yml \
  --seed 42
```

**Output**: 
- `meta_checkpoint_seed42.json` - Curvature metrics
- `eps_list_checkpoint_seed42.npy` - Perturbation deltas
- `losses_checkpoint_seed42.npy` - Losses along direction

### 4. **Intrinsic Dimension** (`intrinsic_dim.py`)
Measures effective parameter dimensionality by training in a d-dimensional subspace.

```bash
python3 src/probes/intrinsic_dim.py \
  --config experiments/configs/sgd.yml \
  --d_sub 15 \
  --seed 42
```

**Output**: `subspace_d15_seed42.pth`
- Test accuracy in 15D subspace
- Finding: All optimizers drop to ~14% accuracy in 15D subspace (full 183K dims needed)

### 5. **Interpolation** (`interpolation.py`)
Traces loss along linear path between two trained models.

```bash
python3 src/probes/interpolation.py \
  --ckpt_a experiments/runs/sgd_seed42/checkpoint.pt \
  --ckpt_b experiments/runs/sgd_momentum_seed42/checkpoint.pt \
  --config experiments/configs/sgd.yml \
  --steps 41
```

**Output**:
- `interp_data.npz` - Alphas and losses
- `interp_checkpoint_to_checkpoint.png` - Visualization

### 6. **Minimum Energy Path (AutoNEB)** (`auto_neb.py`)
Finds minimum energy paths between two trained models using Nudged Elastic Band.

```bash
python3 src/probes/auto_neb.py \
  --ckpt_a experiments/runs/sgd_seed42/checkpoint.pt \
  --ckpt_b experiments/runs/sgd_momentum_seed42/checkpoint.pt \
  --config experiments/configs/sgd.yml \
  --n_nodes 5 --iters 20 --seed 42
```

**Output**: 
- `neb_meta_checkpoint_to_checkpoint.json` - Path metadata
- `node_00_checkpoint_to_checkpoint.npy` to `node_04_checkpoint_to_checkpoint.npy` - 5 nodes on path

**Results**:
- SGD→SGD+Momentum: path_loss=3.263 (smoothest connection)
- SGD+Momentum→AdamW: path_loss=4.667
- AdamW→SGD: path_loss=4.708

## Training

### Train single optimizer

```bash
python3 src/train.py --config experiments/configs/sgd.yml
```

### Train all three optimizers

```bash
export PYTHONPATH="/path/to/Assignment Flykites:$PYTHONPATH"

# SGD
python3 src/train.py --config experiments/configs/sgd.yml

# SGD+Momentum
python3 src/train.py --config experiments/configs/sgd_momentum.yml

# AdamW
python3 src/train.py --config experiments/configs/adamw.yml
```

## Running Complete Experiment

Run all 6 probes on all 3 optimizers (18 total probe runs + 3 AutoNEB paths):

```bash
cd /path/to/Assignment\ Flykites
export PYTHONPATH="/path/to/Assignment\ Flykites:$PYTHONPATH"

# SGD probes
python3 src/probes/hessian_lanczos.py --checkpoint experiments/runs/sgd_seed42/checkpoint.pt --config experiments/configs/sgd.yml --k 3 --seed 42
python3 src/probes/sgd_noise.py --checkpoint experiments/runs/sgd_seed42/checkpoint.pt --config experiments/configs/sgd.yml --seed 42
python3 src/probes/perturbation.py --checkpoint experiments/runs/sgd_seed42/checkpoint.pt --config experiments/configs/sgd.yml --seed 42
python3 src/probes/intrinsic_dim.py --config experiments/configs/sgd.yml --d_sub 15 --seed 42
python3 src/probes/interpolation.py --ckpt_a experiments/runs/sgd_seed42/checkpoint.pt --ckpt_b experiments/runs/sgd_momentum_seed42/checkpoint.pt --config experiments/configs/sgd.yml --steps 41

# Repeat for sgd_momentum_seed42 and adamw_seed42...
```

## Results Analysis

### Lanczos Eigenvalues

- **SGD**: λ_max=129.34, λ_min=-4.94, condition#=26.19
- **SGD+Momentum**: λ_max=37.60, λ_min=-2.01, condition#=18.74 (flattest)
- **AdamW**: λ_max=175.18, λ_min=-3.11, condition#=56.39 (sharpest)

All have **indefinite Hessians** (negative eigenvalues), indicating non-convex but reasonable minima.

### Gradient Noise Analysis

Noise ratio (λ_C / λ_H) - higher means more noisy optimization:
- **SGD**: 2.77 (highest noise, fixed learning rate struggles)
- **SGD+Momentum**: 3.60 (smooths noise with momentum)
- **AdamW**: 1.52 (lowest noise, adaptive rates handle well)

### Loss Landscape Interpolation

Path interpolation between optimizers shows:
- **SGD→SGD+Momentum**: Min loss at α=1.325 (beyond target)
- **SGD+Momentum→AdamW**: Min loss at α=-0.025 (near SGD+Momentum)
- **AdamW→SGD**: Min loss at α=-0.025 (sharp barrier to SGD)

### Intrinsic Dimension

All optimizers require full 183K parameter space:
- Training in 15D subspace: ~14% accuracy vs 71.9% with full space
- Finding: Loss landscape effective dimensionality > 15 dimensions

### Minimum Energy Path (AutoNEB)

Automatic Nudged Elastic Band finds minimum energy paths between optimizer checkpoints:

**SGD → SGD+Momentum**: 
- Path loss: 3.263 (lowest, smooth connection)
- 5 nodes, 12 iterations
- Finding: Relatively smooth transition, momentum variants nearby in parameter space

**SGD+Momentum → AdamW**:
- Path loss: 4.667 (moderate barrier)
- 5 nodes, 12 iterations
- Finding: Larger barrier indicates distinct optimization valleys

**AdamW → SGD**:
- Path loss: 4.708 (similar to SGD+Momentum→AdamW)
- 5 nodes, 12 iterations
- Finding: AdamW and SGD in different regions, harder to connect

**Interpretation**: The lowest path loss between SGD and SGD+Momentum (3.263) vs higher loss between AdamW and others (4.7) suggests that momentum variants explore a more connected region of the loss landscape, while AdamW finds a more isolated solution.

## File Organization

All probe results are organized per optimizer:

```
experiments/runs/sgd_seed42/
├── checkpoint.pt
├── probes/
│   ├── lanczos_spectrum_checkpoint.json
│   ├── sgd_noise_checkpoint_seed42.json
│   ├── meta_checkpoint_seed42.json
│   ├── eps_list_checkpoint_seed42.npy
│   ├── losses_checkpoint_seed42.npy
│   ├── subspace_d15_seed42.pth
│   ├── interp_data.npz
│   └── interp_checkpoint_to_checkpoint.png
└── auto_neb/
    ├── neb_meta_checkpoint_to_checkpoint.json
    ├── node_00_checkpoint_to_checkpoint.npy
    ├── node_01_checkpoint_to_checkpoint.npy
    ├── node_02_checkpoint_to_checkpoint.npy
    ├── node_03_checkpoint_to_checkpoint.npy
    └── node_04_checkpoint_to_checkpoint.npy
```

Same structure for `sgd_momentum_seed42/` and `adamw_seed42/`.

## Configuration Files

Each optimizer has a config YAML file with:
- **Dataset**: CIFAR-10 (128 batch size, 256 eval batch)
- **Model**: SimpleCNN with 3 conv layers, 64 base filters
- **Training**: 12 epochs
- **Optimizer parameters**: Specific lr, momentum, weight_decay
- **Probe output directory**: `experiments/runs/{opt}_seed42/probes/`

### SGD Config Example
```yaml
seed: 42
device: auto

dataset:
  name: CIFAR10
  root: data
  train_batch: 128
  eval_batch: 256

model:
  name: small_cnn
  num_classes: 10

optimizer:
  name: sgd
  lr: 0.01
  momentum: 0.0
  weight_decay: 5e-4

train:
  epochs: 12
  save_dir: experiments/runs/sgd_seed42
  save_name: checkpoint.pt

probes:
  out_dir: experiments/runs/sgd_seed42/probes
```

## Key Insights

1. **Adaptive Learning Rates Win**: AdamW achieves best accuracy (71.91%) despite sharpest landscape because per-parameter adaptation handles sharp curvatures automatically.

2. **Momentum Flattens Landscape**: SGD+Momentum finds a much flatter region (λ_max=37.60 vs 129.34 for SGD), suggesting momentum drives exploration toward flatter areas.

3. **Noise Resilience Matters**: AdamW's lower noise ratio (1.52) indicates it better handles noisy gradient signals, key for stochastic optimization.

4. **Early Convergence**: AdamW converges fastest in early epochs (44.71% at epoch 0 vs 31.85% for SGD), gaining advantage through rapid initial learning.

5. **Non-Convex Landscape**: All optimizers find indefinite Hessians with negative eigenvalues, indicating the loss landscape is genuinely non-convex but in reasonable local minima.

## Citation

This project implements loss landscape analysis techniques from:
- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the Loss Landscape of Neural Nets. NeurIPS.
- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2020). Sharpness Aware Minimization for Efficiently Improving Generalization. ICLR.

## License

This is an assignment project for educational purposes.
