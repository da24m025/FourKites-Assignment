"""Random subspace training for intrinsic dimension estimation.

CRITICAL FIXES: requires stateless, explicit dtype/device, saves outputs, AdamW optimizer.

The idea: theta = theta0 + P @ z, train only z in a d_sub dimensional subspace.
If training in d_sub-dim subspace achieves high accuracy, the effective dimension is ≤ d_sub.

Usage:
  python src/probes/intrinsic_dim.py --config config.yaml --d_sub 50 --seed 42
"""
import argparse
import yaml
import torch
import numpy as np
import json
from pathlib import Path
import time

# REQUIRED for gradient flow to z
try:
    from torch.nn.utils import stateless
except ImportError:
    raise RuntimeError("stateless.functional_call required (PyTorch >= 1.9)")

from src.utils.exp import get_dataloaders, set_seed
from src.models.small_cnn import SmallCNN


def flatten_params(params):
    """Flatten parameters to vector."""
    return torch.cat([p.detach().view(-1) for p in params])


def unflatten_like(flat, params_template):
    """Unflatten vector to parameter structure, preserving device/dtype."""
    out = []
    idx = 0
    for p in params_template:
        n = p.numel()
        out.append(flat[idx:idx+n].view_as(p))
        idx += n
    return out


def to_device_and_dtype(tensor, ref):
    """Move tensor to same device and dtype as reference tensor."""
    return tensor.to(ref.device).to(ref.dtype)


def _save_subspace_results(P, z, theta0, param_names, cfg, seed, test_acc, verbose=True):
    """Save subspace training results to disk for reproducibility."""
    d_sub = z.numel()
    out_dir = Path(cfg.get('probes', {}).get('out_dir', 'results/intrinsic_dim'))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'P': P.cpu(), 'z': z.detach().cpu(), 'theta0': theta0.cpu(),
        'param_names': param_names, 'd_sub': d_sub,
        'test_accuracy': float(test_acc), 'seed': seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_path = out_dir / f"subspace_d{d_sub}_seed{seed}.pth"
    torch.save(save_dict, save_path)
    
    if verbose:
        print(f"  Saved to {save_path}")


def train_subspace(cfg, d_sub, seed=42, verbose=True, save_outputs=True):
    """Train in random d_sub-dimensional subspace via functional_call.
    
    Args:
        cfg: configuration dict
        d_sub: subspace dimension
        seed: random seed for reproducibility
        verbose: if True, print progress
        save_outputs: if True, save P, z, theta0 to disk
    
    Returns:
        dict with keys: test_accuracy, d_sub, seed, P, z, theta0
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"[Intrinsic-Dim] device={device}, seed={seed}, d_sub={d_sub}")
    
    tr, te = get_dataloaders(cfg)
    
    model = SmallCNN(num_classes=cfg['model'].get('num_classes', 10)).to(device)
    model.eval()  # Always in eval mode
    
    for p in model.parameters():
        p.requires_grad = False
    
    param_list = list(model.parameters())
    with torch.no_grad():
        theta0 = flatten_params(param_list)
    
    # CRITICAL FIX: Ensure dtype/device consistency
    param_dtype = param_list[0].dtype
    param_device = param_list[0].device
    theta0 = to_device_and_dtype(theta0, param_list[0])
    
    D = theta0.numel()
    
    P = torch.randn(D, d_sub, device=param_device, dtype=param_dtype) / np.sqrt(d_sub)
    z = torch.zeros(d_sub, device=param_device, dtype=param_dtype, requires_grad=True)
    
    # Use AdamW for stability
    opt = torch.optim.AdamW([z], lr=float(cfg['optimizer'].get('lr', 1e-3)),
                             weight_decay=float(cfg['optimizer'].get('weight_decay', 1e-4)))
    loss_fn = torch.nn.CrossEntropyLoss()
    
    param_names = [n for n, _ in model.named_parameters()]
    
    for epoch in range(cfg['train']['epochs']):
        model.eval()
        train_loss = 0.0
        num_batches = 0
        
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            
            # Reconstruct theta in subspace: theta = theta0 + P @ z
            # z has requires_grad=True → autograd path intact
            theta = theta0 + P @ z
            
            # Unflatten back to parameter structure
            new_params = unflatten_like(theta, param_list)
            
            # Create param dict for functional_call (in same order as param_names)
            param_dict = {name: param for name, param in zip(param_names, new_params)}
            
            # CRITICAL FIX: Use stateless (no fallback)
            out = stateless.functional_call(model, param_dict, (xb,))
            
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        if verbose and (epoch + 1) % max(1, cfg['train']['epochs'] // 3) == 0:
            print(f"  Epoch {epoch+1}/{cfg['train']['epochs']} train loss: {train_loss/num_batches:.4f}")
    
    # Evaluate on test set with final z
    tot, corr = 0, 0
    with torch.no_grad():
        theta_final = theta0 + P @ z.detach()
        final_params = unflatten_like(theta_final, param_list)
        param_dict = {name: param for name, param in zip(param_names, final_params)}
        
        for xb, yb in te:
            xb, yb = xb.to(device), yb.to(device)
            preds = stateless.functional_call(model, param_dict, (xb,)).argmax(dim=1)
            tot += yb.size(0)
            corr += (preds == yb).sum().item()
    
    acc = corr / tot
    
    if verbose:
        print(f"  Final test accuracy: {acc:.4f}")
    
    if save_outputs:
        _save_subspace_results(P, z, theta0, param_names, cfg, seed, acc, verbose=verbose)
    
    return {'test_accuracy': acc, 'd_sub': d_sub, 'seed': seed, 'P': P, 'z': z.detach(), 'theta0': theta0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Intrinsic Dimension: Random Subspace Training')
    p.add_argument('--config', type=str, required=True, help='Path to config YAML')
    p.add_argument('--d_sub', type=int, required=True, help='Subspace dimension')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--no-save', action='store_true', help='Do not save outputs')
    args = p.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    
    print("="*70)
    print(f"Intrinsic Dimension: d_sub={args.d_sub}, seed={args.seed}")
    print("="*70)
    
    try:
        result = train_subspace(cfg, args.d_sub, seed=args.seed, verbose=True, save_outputs=not args.no_save)
        print(f"\n✓ Result: acc={result['test_accuracy']:.4f}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
