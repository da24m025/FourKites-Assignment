"""AutoNEB (Automatic Nudged Elastic Band) - mode connection path finding.

Uses ParameterList with proper optimization to find low-loss paths between minima.
Gradients flow correctly through stateless.functional_call.

"""
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import time
import json

# REQUIRED for gradient flow to nodes
try:
    from torch.nn.utils import stateless
    _HAS_STATELESS = True
except ImportError:
    _HAS_STATELESS = False

from src.models.small_cnn import SmallCNN
from src.utils.exp import get_dataloaders, load_checkpoint


def flatten_params(model):
    """Flatten all model parameters."""
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def unflatten_like(flat, params_template):
    """Unflatten to parameter structure."""
    out = []
    idx = 0
    for p in params_template:
        n = p.numel()
        out.append(flat[idx:idx+n].view_as(p))
        idx += n
    return out


def _save_neb_nodes(nodes, ckpt_a, ckpt_b, cfg, metadata, verbose=True):
    """Save NEB path nodes to disk as numpy arrays + metadata JSON.
    
    Args:
        nodes: ParameterList of nodes
        ckpt_a, ckpt_b: checkpoint paths
        cfg: config dict
        metadata: dict with run params (n_nodes, iters_run, spring_k, etc)
        verbose: if True, print save location
    """
    out_dir = Path(cfg.get('train', {}).get('save_dir', 'results')) / "auto_neb"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_a_name = Path(ckpt_a).stem
    ckpt_b_name = Path(ckpt_b).stem
    
    # Save each node as numpy array
    for idx, node in enumerate(nodes):
        flat = node.detach().cpu().numpy()
        node_path = out_dir / f"node_{idx:02d}_{ckpt_a_name}_to_{ckpt_b_name}.npy"
        np.save(node_path, flat)
    
    # Save metadata JSON
    metadata_full = {
        'ckpt_a': str(ckpt_a),
        'ckpt_b': str(ckpt_b),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        **metadata,
    }
    meta_path = out_dir / f"neb_meta_{ckpt_a_name}_to_{ckpt_b_name}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata_full, f, indent=2)
    
    if verbose:
        print(f"Saved {len(nodes)} nodes to {out_dir}/")
        print(f"Metadata: {meta_path}")



def auto_neb(ckpt_a, ckpt_b, cfg, n_nodes=8, iters=100, spring_k=1e-2, lr=1e-3, 
             seed=42, verbose=True, save_nodes=True):
    
    # Check stateless availability (REQUIRED for correct gradients)
    if not _HAS_STATELESS:
        raise RuntimeError(
            "stateless.functional_call not available. AutoNEB requires PyTorch >= 1.9 "
            "with functorch support for correct gradient flow to nodes. "
            "Upgrade PyTorch or install functorch."
        )
    
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Validate n_nodes
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2, got {n_nodes}")
    if n_nodes < 3:
        if verbose:
            print(f"AutoNEB: n_nodes < 3 -> no interior nodes to optimize. Returning linear interpolation.")
        return None  # or return trivial path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"AutoNEB: device={device}, seed={seed}")
    
    tr, te = get_dataloaders(cfg)
    
    # Load both checkpoints
    model_a = SmallCNN(num_classes=cfg['model'].get('num_classes', 10)).to(device)
    ckpt_a_data = load_checkpoint(ckpt_a, map_location=device)
    model_a.load_state_dict(ckpt_a_data['model'])
    model_a.eval()
    
    model_b = SmallCNN(num_classes=cfg['model'].get('num_classes', 10)).to(device)
    ckpt_b_data = load_checkpoint(ckpt_b, map_location=device)
    model_b.load_state_dict(ckpt_b_data['model'])
    model_b.eval()
    
    # Flatten parameters
    flat_a = flatten_params(model_a).to(device)
    flat_b = flatten_params(model_b).to(device)
    
    param_names = [n for n, _ in model_a.named_parameters()]
    
    # Initialize nodes as ParameterList for proper gradient tracking
    nodes = nn.ParameterList([
        nn.Parameter((flat_a * (1 - t) + flat_b * t).detach(), requires_grad=(0 < i < n_nodes - 1))
        for i, t in enumerate(torch.linspace(0, 1, n_nodes))
    ])
    
    # Get interior nodes for optimization
    interior_nodes = [nodes[i] for i in range(1, n_nodes - 1)]
    
    # Use AdamW for better convergence (better than SGD for this problem)
    optimizer = torch.optim.AdamW(interior_nodes, lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    if verbose:
        print(f"AutoNEB: n_nodes={n_nodes}, interior_nodes={len(interior_nodes)}, "
              f"max_iters={iters}, spring_k={spring_k}, lr={lr}")
    
    # ============  Persistent data iterator ============
    # Create ONE iterator before loop; advance it properly
    data_iter = iter(te)
    
    # ============ Early stopping ============
    best_loss = float('inf')
    patience = max(2, iters // 20)  # be patient for short runs
    no_improve = 0
    eval_every = max(1, iters // 10)
    
    start_time = time.time()
    
    for it in range(iters):
        optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        barrier_losses = []
        
        for i in range(1, n_nodes - 1):
            # ============  Advance single iterator ============
            try:
                xb, yb = next(data_iter)
            except StopIteration:
                # Restart iterator at epoch boundary
                data_iter = iter(te)
                xb, yb = next(data_iter)
            
            xb, yb = xb.to(device), yb.to(device)
            
            # Unflatten node i to model parameter structure
            new_params = unflatten_like(nodes[i], list(model_a.parameters()))
            param_dict = {name: p for name, p in zip(param_names, new_params)}
            
            # Forward via stateless (correct gradient flow to nodes)
            out = stateless.functional_call(model_a, param_dict, (xb,))
            L = loss_fn(out, yb)
            barrier_losses.append(L.item())
            
            # Spring penalty: keep nodes evenly spaced
            spring_prev = (nodes[i-1] - nodes[i]).pow(2).sum()
            spring_next = (nodes[i+1] - nodes[i]).pow(2).sum()
            spring_penalty = spring_prev + spring_next
            
            node_loss = L + spring_k * spring_penalty
            total_loss = total_loss + node_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(interior_nodes, max_norm=1.0)
        
        # Optimization step
        optimizer.step()
        
        # Early stopping logic
        if total_loss.item() < best_loss - 1e-7:
            best_loss = float(total_loss.item())
            no_improve = 0
        else:
            no_improve += 1
        
        # Logging
        if verbose and (it % eval_every == 0 or it == iters - 1):
            elapsed = time.time() - start_time
            max_barrier = max(barrier_losses) if barrier_losses else 0.0
            avg_barrier = sum(barrier_losses) / len(barrier_losses) if barrier_losses else 0.0
            print(f"  Iter {it:4d} | loss {total_loss.item():.6f} | "
                  f"barrier max={max_barrier:.4f} avg={avg_barrier:.4f} | "
                  f"elapsed {elapsed:.1f}s")
        
        # Early stopping
        if no_improve >= patience and it > iters // 2:
            if verbose:
                print(f"Early stopping: no improvement for {patience} iterations")
            break
    
    if verbose:
        print(f"AutoNEB complete after {it+1} iterations")
    
    # ============  FIX #4: Save nodes & metadata ============
    if save_nodes:
        _save_neb_nodes(nodes, ckpt_a, ckpt_b, cfg, {
            'n_nodes': n_nodes,
            'iters_run': it + 1,
            'spring_k': spring_k,
            'lr': lr,
            'seed': seed,
            'best_loss': best_loss,
        }, verbose=verbose)
    
    return nodes


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='AutoNEB: Mode connection path finding')
    p.add_argument('--ckpt_a', type=str, required=True, help='Path to checkpoint A (first minimum)')
    p.add_argument('--ckpt_b', type=str, required=True, help='Path to checkpoint B (second minimum)')
    p.add_argument('--config', type=str, required=True, help='Path to config YAML')
    p.add_argument('--n_nodes', type=int, default=8, help='Number of nodes on path (>= 3)')
    p.add_argument('--iters', type=int, default=100, help='Max optimization iterations')
    p.add_argument('--spring_k', type=float, default=1e-2, help='Spring constant')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument('--no-save', action='store_true', help='Do not save nodes to disk')
    args = p.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    
    print("="*70)
    print("AutoNEB: Automatic Nudged Elastic Band Path Finding")
    print("="*70)
    print(f"Checkpoint A: {args.ckpt_a}")
    print(f"Checkpoint B: {args.ckpt_b}")
    print(f"Config: {args.config}")
    print(f"Hyperparams: n_nodes={args.n_nodes}, iters={args.iters}, spring_k={args.spring_k}, lr={args.lr}, seed={args.seed}")
    
    try:
        nodes = auto_neb(
            args.ckpt_a, args.ckpt_b, cfg,
            n_nodes=args.n_nodes,
            iters=args.iters,
            spring_k=args.spring_k,
            lr=args.lr,
            seed=args.seed,
            verbose=True,
            save_nodes=not args.no_save
        )
        
        if nodes is not None:
            print(f"\n✓ AutoNEB complete: {len(nodes)} nodes on path")
        else:
            print("\n✗ AutoNEB failed: returned None")
            
    except Exception as e:
        print(f"\n✗ AutoNEB failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

