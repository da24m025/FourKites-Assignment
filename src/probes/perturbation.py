"""Perturbation analysis: move along Hessian top eigenvector and measure loss.

CRITICAL FIXES:
- Fixed data iterator (uses fixed eval batch, not repeatedly first batch)
- Explicit device/dtype consistency
- Proper parameter restore after probe
- Better curvature estimation (central finite difference, not asymmetric)
- Saves results to disk (eps_list, losses, metadata)
- Multi-seed support with proper reproducibility
"""
import argparse
import yaml
import torch
import numpy as np
import json
from pathlib import Path
import time

from src.models.small_cnn import SmallCNN
from src.utils.exp import get_dataloaders, load_checkpoint
from src.probes.hessian import power_top_eig


def flatten_params(params):
    """Flatten parameters."""
    return torch.cat([p.detach().view(-1) for p in params])


def unflatten_like(flat, params_template):
    """Unflatten vector to parameter structure."""
    out = []
    idx = 0
    for p in params_template:
        n = p.numel()
        out.append(flat[idx:idx+n].view_as(p))
        idx += n
    return out


def set_model_params(model, param_list):
    """Set model parameters."""
    for p, new in zip(model.parameters(), param_list):
        p.data.copy_(new)


def perturbation_probe(ckpt_path, cfg, eps_scale=1.0, steps=41, num_eval_batches=4, 
                       seed=42, verbose=True, save_outputs=True):
    """Perturbation analysis along top Hessian eigenvector.
    
    CRITICAL FIXES:
    - Uses fixed eval batches (averaged for noise reduction)
    - Proper device/dtype consistency
    - Restores model params after probe
    - Better curvature estimation (central finite difference)
    - Saves results to disk
    
    Args:
        ckpt_path: path to model checkpoint
        cfg: config dict
        eps_scale: perturbation range (symmetric, ±eps_scale)
        steps: number of evaluation points
        num_eval_batches: number of batches to average loss over (reduces noise)
        seed: random seed
        verbose: if True, print progress
        save_outputs: if True, save results to disk
    
    Returns:
        dict with keys: eigenvalue, eps_list, losses, curvature, seed, ckpt_name
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"[Perturbation] device={device}, seed={seed}, eps_scale={eps_scale}")
    
    tr, te = get_dataloaders(cfg)
    
    model = SmallCNN(num_classes=cfg['model'].get('num_classes', 10))
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Get top Hessian eigenvector
    if verbose:
        print("  Computing top Hessian eigenvector (power iteration)...")
    eig, eig_std, vec = power_top_eig(model, loss_fn, te, device, power_iters=30)
    
    flat_theta = flatten_params(list(model.parameters())).to(device)
    
    # CRITICAL FIX: Ensure device/dtype consistency for vec
    vec = vec.to(device).to(flat_theta.dtype)
    vec = vec / (vec.norm() + 1e-12)
    
    # CRITICAL FIX #1: Collect fixed eval batches (not fresh ones each time)
    if verbose:
        print(f"  Collecting {num_eval_batches} evaluation batches...")
    eval_batches = []
    it_eval = iter(te)
    for _ in range(num_eval_batches):
        try:
            xb, yb = next(it_eval)
        except StopIteration:
            it_eval = iter(te)
            xb, yb = next(it_eval)
        eval_batches.append((xb.to(device), yb.to(device)))
    
    # CRITICAL FIX #2: Save original params to restore later
    orig_state = {name: p.detach().clone() for name, p in model.named_parameters()}
    
    # Perturbation along top eigenvector (symmetric)
    eps_list = torch.linspace(-eps_scale, eps_scale, steps, device=device, dtype=flat_theta.dtype)
    losses = []
    
    if verbose:
        print(f"  Evaluating at {steps} perturbation points...")
    
    for eps in eps_list:
        new_flat = flat_theta + eps * vec
        new_params = unflatten_like(new_flat, list(model.parameters()))
        set_model_params(model, new_params)
        
        # CRITICAL FIX #3: Average loss over eval_batches to reduce noise
        loss_acc = 0.0
        with torch.no_grad():
            for xb, yb in eval_batches:
                loss_acc += loss_fn(model(xb), yb).item()
        losses.append(loss_acc / len(eval_batches))
    
    # CRITICAL FIX #4: Restore original params
    for name, p in model.named_parameters():
        p.data.copy_(orig_state[name])
    
    if verbose:
        print("  Model parameters restored")
    
    # CRITICAL FIX #5: Better curvature estimation (central finite difference)
    # Using central difference: f''(0) ≈ (f(h) - 2f(0) + f(-h)) / h^2
    h = eps_list[1].item() - eps_list[0].item()  # spacing
    center_idx = (len(eps_list) - 1) // 2  # index of zero
    
    if center_idx > 0 and center_idx < len(eps_list) - 1:
        f_plus = losses[center_idx + 1]
        f_center = losses[center_idx]
        f_minus = losses[center_idx - 1]
        
        # Central finite difference: second derivative
        second_deriv = (f_plus - 2 * f_center + f_minus) / (h ** 2)
    else:
        second_deriv = np.nan
    
    if verbose:
        print(f"  Top eigenvalue: {eig:.6f}")
        print(f"  Curvature (central diff): {second_deriv:.6f}")
        print(f"  Loss range: [{min(losses):.6f}, {max(losses):.6f}]")
    
    result = {
        'eigenvalue': float(eig),
        'eps_list': eps_list.cpu().numpy(),
        'losses': np.array(losses),
        'curvature': float(second_deriv),
        'seed': seed,
        'ckpt_name': Path(ckpt_path).stem,
        'num_eval_batches': num_eval_batches,
    }
    
    # CRITICAL FIX #6: Save results to disk
    if save_outputs:
        _save_perturbation_results(result, cfg, verbose=verbose)
    
    return result


def _save_perturbation_results(result, cfg, verbose=True):
    """Save perturbation analysis results to disk."""
    out_dir = Path(cfg.get('probes', {}).get('out_dir', 'results/perturbation'))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_name = result['ckpt_name']
    seed = result['seed']
    
    # Save numpy arrays
    np.save(out_dir / f"eps_list_{ckpt_name}_seed{seed}.npy", result['eps_list'])
    np.save(out_dir / f"losses_{ckpt_name}_seed{seed}.npy", result['losses'])
    
    # Save metadata as JSON
    meta = {
        'eigenvalue': result['eigenvalue'],
        'curvature': result['curvature'],
        'seed': result['seed'],
        'ckpt_name': result['ckpt_name'],
        'num_eval_batches': result['num_eval_batches'],
        'eps_scale': float(result['eps_list'][0]).__abs__(),
        'num_points': len(result['eps_list']),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    meta_path = out_dir / f"meta_{ckpt_name}_seed{seed}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    if verbose:
        print(f"  Saved results to {out_dir}/")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Perturbation Probe: Top-eigenvector Sensitivity')
    p.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    p.add_argument('--config', type=str, required=True, help='Path to config YAML')
    p.add_argument('--eps_scale', type=float, default=1.0, help='Perturbation range (±)')
    p.add_argument('--steps', type=int, default=41, help='Number of evaluation points')
    p.add_argument('--num_eval_batches', type=int, default=4, help='Batches to average loss over')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--no-save', action='store_true', help='Do not save outputs')
    args = p.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    
    print("="*70)
    print("Perturbation Probe: Top-Eigenvector Sensitivity Analysis")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Hyperparams: eps_scale={args.eps_scale}, steps={args.steps}, seed={args.seed}")
    
    try:
        result = perturbation_probe(
            args.checkpoint, cfg,
            eps_scale=args.eps_scale,
            steps=args.steps,
            num_eval_batches=args.num_eval_batches,
            seed=args.seed,
            verbose=True,
            save_outputs=not args.no_save
        )
        
        print("\n" + "="*70)
        print(f"✓ Results:")
        print(f"  - Top eigenvalue: {result['eigenvalue']:.6f}")
        print(f"  - Curvature: {result['curvature']:.6f}")
        print(f"  - Loss range: [{result['losses'].min():.6f}, {result['losses'].max():.6f}]")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
