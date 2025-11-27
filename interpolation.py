"""Interpolation & filter-wise normalization utilities.

CRITICAL FIXES:
- Batch averaging: evaluate loss on multiple fixed batches per alpha (not single first batch)
- Device/dtype consistency: ensure flat vectors on correct device before arithmetic
- Per-filter normalization option for convolutional kernels
- Better endpoint visualization (highlight alpha=0 and alpha=1)

Usage:
  python3 src/probes/interpolation.py --ckpt_a results/mvp_smallbatch.pt --ckpt_b results/mvp_largebatch.pt --config experiments/configs/mvp.yml
"""
import argparse
import yaml
import torch
import numpy as np
import copy
from pathlib import Path
from src.models.small_cnn import SmallCNN
from src.utils.exp import get_dataloaders, load_checkpoint


def flatten_params(params):
    """Flatten parameter list to single vector."""
    return torch.cat([p.detach().view(-1) for p in params])


def unflatten_like(flat, params_template):
    """Unflatten vector back to parameter structure."""
    out = []
    idx = 0
    for p in params_template:
        n = p.numel()
        out.append(flat[idx:idx+n].view_as(p))
        idx += n
    return out


def set_model_params(model, param_list):
    """Set model parameters from list."""
    for p, new in zip(model.parameters(), param_list):
        p.data.copy_(new)


def filter_normalize_direction(model_init, model_final, per_filter=False):
    """Compute normalized direction between two models.
    
    CRITICAL FIX: Per-parameter (or per-filter) scaling ensures ||dir|| per tensor = ||init|| per tensor.
    This is filter-wise normalization (normalization in weight space, not loss space).
    
    Args:
        model_init: initial model
        model_final: final model
        per_filter: if True, normalize per-filter (per out-channel) for conv layers.
                   if False, normalize per parameter tensor (global norm).
    
    Returns:
        (inits, scaled_direction)
    """
    inits = [p.detach().clone() for p in model_init.parameters()]
    finals = [p.detach().clone() for p in model_final.parameters()]
    dir_list = [f - q for f, q in zip(finals, inits)]
    
    scaled = []
    for t_init, d in zip(inits, dir_list):
        if per_filter and t_init.dim() == 4:  # conv layer: (out, in, k, k)
            # Normalize per output channel
            out_norms = t_init.view(t_init.size(0), -1).norm(dim=1, keepdim=True)  # (out, 1)
            d_resh = d.view(d.size(0), -1)  # (out, in*k*k)
            d_norms = d_resh.norm(dim=1, keepdim=True)  # (out, 1)
            # Avoid division by zero
            scale = out_norms / (d_norms + 1e-12)  # (out, 1)
            scaled.append((d_resh * scale).view_as(d))
        else:
            # Global per-tensor normalization (all dims)
            n_init = t_init.norm()
            n_dir = d.norm()
            if n_dir.item() == 0:
                scaled.append(d)
            else:
                scaled.append(d * (n_init / (n_dir + 1e-12)))
    
    return inits, scaled


def loss_along_line(model_init, model_final, dataloader, steps=41, device='cpu', 
                     num_eval_batches=8, per_filter=False):
    """Compute loss along linear interpolation between two models (batch-averaged).
    
    CRITICAL FIXES:
    - Batch averaging: evaluate on multiple fixed batches per alpha (not single first batch)
    - Device/dtype consistency: ensure flat vectors on correct device
    - Per-filter normalization option
    
    Args:
        model_init: initial model
        model_final: final model
        dataloader: dataloader for loss evaluation
        steps: number of interpolation points
        device: compute device
        num_eval_batches: number of evaluation batches to average over (reduces noise)
        per_filter: use per-filter normalization for conv layers
    
    Returns:
        (alphas, losses) where alpha=0 is model_init, alpha=1 is model_final
    """
    inits, scaled_dirs = filter_normalize_direction(model_init, model_final, per_filter=per_filter)
    alphas = np.linspace(-1.0, 2.0, steps)
    losses_per_alpha = []
    model = copy.deepcopy(model_init)
    model.eval()  # Important: use eval mode for determinism
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # CRITICAL FIX #1: Collect fixed evaluation batches (persistent iterator, not first batch only)
    eval_batches = []
    it_eval = iter(dataloader)
    eval_n = min(num_eval_batches, len(dataloader))
    for _ in range(eval_n):
        try:
            xb, yb = next(it_eval)
        except StopIteration:
            it_eval = iter(dataloader)
            xb, yb = next(it_eval)
        eval_batches.append((xb.to(device), yb.to(device)))
    
    # CRITICAL FIX #2: Device/dtype consistency for flat vectors
    flat_inits = flatten_params(inits).to(device)
    flat_dir = flatten_params(scaled_dirs).to(device)
    # Ensure dtype matches model parameters
    param_dtype = next(model.parameters()).dtype
    flat_inits = flat_inits.to(param_dtype)
    flat_dir = flat_dir.to(param_dtype)
    
    for a in alphas:
        flat_new = flat_inits + a * flat_dir
        new_params = unflatten_like(flat_new, list(model.parameters()))
        set_model_params(model, new_params)
        
        # CRITICAL FIX #3: Average loss over evaluation batches
        loss_acc = 0.0
        with torch.no_grad():
            for xb, yb in eval_batches:
                out = model(xb)
                loss_acc += loss_fn(out, yb).item()
        loss_avg = loss_acc / len(eval_batches)
        losses_per_alpha.append(loss_avg)
    
    losses = np.array(losses_per_alpha)
    return alphas, losses


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Interpolation: Loss landscape between two models')
    p.add_argument('--ckpt_a', type=str, required=True, help='Path to first checkpoint')
    p.add_argument('--ckpt_b', type=str, required=True, help='Path to second checkpoint')
    p.add_argument('--config', type=str, required=True, help='Path to config YAML')
    p.add_argument('--out', type=str, default='interp.png', help='Output plot path')
    p.add_argument('--steps', type=int, default=41, help='Number of interpolation points')
    p.add_argument('--num_eval_batches', type=int, default=8, help='Batches to average loss over')
    p.add_argument('--per_filter', action='store_true', help='Per-filter normalization (conv only)')
    p.add_argument('--no-save', action='store_true', help='Do not save plot/data')
    args = p.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    tr, te = get_dataloaders(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("Interpolation Probe (Batch-Averaged, Device-Safe)")
    print("="*70)
    print(f"Checkpoint A: {args.ckpt_a}")
    print(f"Checkpoint B: {args.ckpt_b}")
    print(f"Device: {device}, Steps: {args.steps}, Eval batches: {args.num_eval_batches}")
    
    model_a = SmallCNN(num_classes=cfg['model'].get('num_classes', 10)).to(device)
    ckpt_a = load_checkpoint(args.ckpt_a, map_location=device)
    model_a.load_state_dict(ckpt_a['model'])
    model_a.eval()
    
    model_b = SmallCNN(num_classes=cfg['model'].get('num_classes', 10)).to(device)
    ckpt_b = load_checkpoint(args.ckpt_b, map_location=device)
    model_b.load_state_dict(ckpt_b['model'])
    model_b.eval()
    
    print(f"Computing interpolation curve (alpha in [-1,2], sampling at {args.steps} points)...")
    alphas, losses = loss_along_line(
        model_a, model_b, te, 
        steps=args.steps,
        device=device,
        num_eval_batches=args.num_eval_batches,
        per_filter=args.per_filter
    )
    
    # Find min/max
    min_idx = np.argmin(losses)
    max_idx = np.argmax(losses)
    alpha_at_min = alphas[min_idx]
    
    print("\n" + "="*70)
    print(f"Interpolation Results:")
    print("="*70)
    print(f"Min loss: {losses[min_idx]:.6f} at alpha={alpha_at_min:.4f}")
    print(f"Max loss: {losses[max_idx]:.6f}")
    print(f"Loss at alpha=0 (ckpt_a): {losses[0]:.6f}")
    print(f"Loss at alpha=1 (ckpt_b): {losses[-1]:.6f}")
    
    # Save results
    if not args.no_save:
        out_dir = Path(cfg.get('probes', {}).get('out_dir', 'results/interpolation'))
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save numeric results
        data_path = out_dir / 'interp_data.npz'
        np.savez(data_path, alphas=alphas, losses=losses, alpha_at_min=alpha_at_min)
        print(f"Saved data to: {data_path}")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, losses, 'b-', linewidth=2.5, label='interpolation curve')
        # CRITICAL FIX: Highlight endpoints (alpha=0 and alpha=1)
        plt.scatter([0.0, 1.0], [losses[0], losses[-1]], color='green', s=100, 
                   marker='o', label='endpoints', zorder=5)
        # Mark minimum
        plt.scatter([alpha_at_min], [losses[min_idx]], color='red', s=100,
                   marker='*', label=f'min (α={alpha_at_min:.2f})', zorder=5)
        plt.axhline(y=losses[min_idx], color='red', linestyle='--', alpha=0.3)
        plt.axvline(x=alpha_at_min, color='red', linestyle='--', alpha=0.3)
        plt.xlabel('α (interpolation parameter)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Loss Landscape: Linear Interpolation Between Two Models', fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if not args.no_save:
            plt.savefig(args.out, dpi=120, bbox_inches='tight')
            print(f"Saved plot to: {args.out}")
        else:
            plt.show()
    except Exception as e:
        print(f"Could not plot: {e}")
    
    print("="*70)
