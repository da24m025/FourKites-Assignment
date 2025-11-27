"""Hessian probing utilities:
- hvp(loss, model, v): Hessian-vector product
- power_top_eig(): Top eigenvalue estimation via power iteration
- hutchinson_trace(): Trace estimation via Hutchinson estimator

Usage:
  python -m src.probes.hessian --checkpoint results/mvp_smallbatch.ckpt --config experiments/configs/mvp.yml
"""
import argparse
import json
import yaml
import torch
from torch.autograd import grad
import numpy as np
from pathlib import Path
from src.utils.exp import get_dataloaders, load_checkpoint
from src.models.small_cnn import SmallCNN


def get_param_vector(model):
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.view(-1) for p in model.parameters()])


def hvp(loss, model, v):
    """Compute Hessian-vector product H*v using autograd.
    
    Args:
        loss: scalar loss value
        model: neural network model
        v: vector (flat torch tensor)
    
    Returns:
        H*v as a flat vector
    """
    grads = grad(loss, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
    g_v = (flat_grads * v).sum()
    hv = grad(g_v, model.parameters(), retain_graph=True)
    hv_flat = torch.cat([
        h.contiguous().view(-1) if h is not None else torch.zeros_like(p).view(-1)
        for h, p in zip(hv, model.parameters())
    ])
    return hv_flat


def power_top_eig(model, loss_fn, data_loader, device, power_iters=30, num_batches=3):
    """Estimate top eigenvalue of Hessian via power iteration (averaged over batches).
    
    Important: Averages eigenvalue estimates across multiple *different* minibatches for robustness.
    The returned eigenvector corresponds to the batch with the highest eigenvalue estimate.
    
    Args:
        model: trained neural network
        loss_fn: loss function
        data_loader: dataloader for computing loss
        device: device to compute on
        power_iters: number of power iteration steps per batch
        num_batches: number of different batches to average over for robustness
    
    Returns:
        (mean_eigenvalue, eigenvalue_std, eigenvector)
    """
    model.to(device).eval()
    D = sum(p.numel() for p in model.parameters())
    
    # Collect eigenvalue estimates across multiple *different* batches
    eig_estimates = []
    vecs = []
    
    # Create iterator once, advance it properly (fixes repeated-batch bug)
    data_iter = iter(data_loader)
    
    for batch_idx in range(num_batches):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            # Restart iterator if we exhaust the loader
            data_iter = iter(data_loader)
            xb, yb = next(data_iter)
        
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = loss_fn(out, yb)
        
        # Initialize random vector
        vec = torch.randn(D, device=device)
        vec /= vec.norm()
        
        # Power iteration on this batch
        for _ in range(power_iters):
            hv = hvp(loss, model, vec)
            hv = hv.detach()
            norm = hv.norm()
            vec = hv / (norm + 1e-12)
        
        # Rayleigh quotient estimate
        hv = hvp(loss, model, vec).detach()
        eig = (vec * hv).sum().item()
        eig_estimates.append(eig)
        vecs.append(vec.detach().cpu())
    
    # Choose eigenvector corresponding to largest eigenvalue estimate
    best_idx = int(np.argmax(eig_estimates))
    final_vec = vecs[best_idx].to(device)
    
    mean_eig = sum(eig_estimates) / len(eig_estimates)
    std_eig = (sum((e - mean_eig)**2 for e in eig_estimates) / max(1, len(eig_estimates) - 1))**0.5
    
    return mean_eig, std_eig, final_vec


def hutchinson_trace(model, loss_fn, data_loader, device, samples=30, num_batches=3):
    """Estimate Hessian trace via Hutchinson estimator (averaged over batches).
    
    Important: Averages trace estimates across multiple *different* minibatches for robustness.
    Uses Rademacher random vectors (±1 with equal probability) for variance reduction.
    
    Args:
        model: trained neural network
        loss_fn: loss function
        data_loader: dataloader for computing loss (different batches used for averaging)
        device: device to compute on
        samples: number of random vectors per batch
        num_batches: number of different batches to average over
    
    Returns:
        (mean_trace, trace_std)
    """
    model.to(device).eval()
    D = sum(p.numel() for p in model.parameters())
    batch_traces = []
    
    # Create iterator once, advance it properly (fixes repeated-batch bug)
    data_iter = iter(data_loader)
    
    for _ in range(num_batches):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            # Restart iterator if we exhaust the loader
            data_iter = iter(data_loader)
            xb, yb = next(data_iter)
        
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = loss_fn(out, yb)
        
        acc = 0.0
        for _ in range(samples):
            v = torch.randint(0, 2, (D,), device=device).float() * 2 - 1
            hv = hvp(loss, model, v).detach()
            acc += (v * hv).sum().item()
        
        batch_trace = acc / samples
        batch_traces.append(batch_trace)
    
    mean_trace = sum(batch_traces) / len(batch_traces)
    std_trace = (sum((t - mean_trace)**2 for t in batch_traces) / max(1, len(batch_traces) - 1))**0.5
    
    return mean_trace, std_trace


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--power_iters', type=int, default=30)
    p.add_argument('--hutchinson_samples', type=int, default=20)
    p.add_argument('--num_batches', type=int, default=3, help='Batches to average over')
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    p.add_argument('--output_dir', type=str, default='results', 
                  help='Directory to save results JSON')
    args = p.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        from src.utils.exp import set_seed
        set_seed(args.seed)
    
    cfg = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, te = get_dataloaders(cfg)
    
    model = SmallCNN(num_classes=cfg['model'].get('num_classes', 10))
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print(f"Estimating Hessian properties (averaging over {args.num_batches} batches)...")
    eig_mean, eig_std, vec = power_top_eig(model, loss_fn, te, device, 
                                           power_iters=args.power_iters,
                                           num_batches=args.num_batches)
    trace_mean, trace_std = hutchinson_trace(model, loss_fn, te, device, 
                                             samples=args.hutchinson_samples,
                                             num_batches=args.num_batches)
    
    n_params = sum(p.numel() for p in model.parameters())
    avg_eig = trace_mean / n_params
    
    print(f"\nHessian Analysis:")
    print(f"  Top eigenvalue:        {eig_mean:.6f} ± {eig_std:.6f}")
    print(f"  Trace (Hutchinson):    {trace_mean:.6f} ± {trace_std:.6f}")
    print(f"  Avg eig (trace/D):     {avg_eig:.6f}")
    print(f"  Number of parameters:  {n_params}")
    print(f"\n  Note: Condition number (λ_max/λ_min) requires smallest eigenvalue estimation.")
    print(f"        Use Lanczos or other methods to estimate λ_min for full spectrum analysis.")
    
    # Save results to JSON
    results = {
        'top_eigenvalue': {
            'mean': float(eig_mean),
            'std': float(eig_std),
        },
        'trace': {
            'mean': float(trace_mean),
            'std': float(trace_std),
        },
        'avg_eigenvalue': {
            'mean': float(avg_eig),
        },
        'n_parameters': int(n_params),
        'checkpoint': args.checkpoint,
        'config': args.config,
        'power_iters': args.power_iters,
        'hutchinson_samples': args.hutchinson_samples,
        'num_batches': args.num_batches,
        'seed': args.seed,
    }
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.checkpoint).stem
    output_file = Path(args.output_dir) / f'hessian_{ckpt_name}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
