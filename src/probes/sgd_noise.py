"""SGD noise covariance probe: analyze gradient covariance and alignment with Hessian.

CRITICAL FIXES:
- Memory-safe per-example gradient computation (push to CPU per sample)
- Gram matrix trick: compute top eigenvalue of gradient covariance without DxD matrix
- Alignment via inner product: v^T C v without forming full covariance
- Batch averaging for robust alignment estimate

Measures how SGD noise (from mini-batch sampling) aligns with loss landscape curvature.
"""
import argparse
import yaml
import torch
import numpy as np
import json
from pathlib import Path
from src.utils.exp import get_dataloaders, load_checkpoint
from src.models.small_cnn import SmallCNN
from src.probes.hessian import power_top_eig


def per_example_gradients_cpu(model, xb, yb, loss_fn):
    """Compute per-example gradients safely (memory-efficient).
    
    CRITICAL FIX: Push each gradient to CPU immediately to avoid DxB GPU memory.
    Handles None gradients (frozen params) gracefully.
    
    Args:
        model: neural network
        xb: batch of inputs
        yb: batch of labels
        loss_fn: loss function
    
    Returns:
        G_cpu: gradient matrix (D, B) on CPU as torch tensor
    """
    model.zero_grad()
    outs = model(xb)
    losses = torch.nn.functional.cross_entropy(outs, yb, reduction='none')
    gflats = []
    params = list(model.parameters())
    
    for loss_i in losses:
        # compute gradient w.r.t. all params
        grads = torch.autograd.grad(loss_i, params, retain_graph=True, allow_unused=True)
        # handle None gradients (frozen params)
        gflat = torch.cat([
            (g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1))
            for g, p in zip(grads, params)
        ])
        # CRITICAL: push to CPU immediately to avoid OOM on GPU
        gflats.append(gflat.detach().cpu())
    
    # Stack per-example gradients: gflats is list of B tensors each size D
    # torch.stack(gflats, dim=0) -> (B, D)
    G_cpu = torch.stack(gflats, dim=0)  # (B, D)
    return G_cpu.t()  # (D, B)


def estimate_alignment_and_spectrum(model, tr, te, loss_fn, device, 
                                    batch_size=32, num_batches=4, seed=42):
    """Estimate alignment and gradient covariance spectrum (memory-safe).
    
    CRITICAL FIXES:
    - Gram matrix trick: avoid forming DxD covariance
    - Batch averaging: robust alignment estimate across multiple batches
    - Low-rank structure: use BxB matrix instead of DxD
    
    Args:
        model: neural network
        tr: training dataloader
        te: test dataloader (for Hessian)
        loss_fn: loss function
        device: compute device
        batch_size: max number of samples per batch for per-example grads
        num_batches: number of batches to average alignment over
        seed: random seed
    
    Returns:
        dict with alignment, eig_c_top, eig_h, and ratios
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Collect eval batches (persistent iterator, not first batch only)
    eval_batches = []
    it_tr = iter(tr)
    for _ in range(num_batches):
        try:
            xb, yb = next(it_tr)
        except StopIteration:
            it_tr = iter(tr)
            xb, yb = next(it_tr)
        # Limit batch size for memory
        xb, yb = xb[:batch_size].to(device), yb[:batch_size].to(device)
        eval_batches.append((xb, yb))
    
    # Compute Hessian top eigenvalue (once)
    eig_h, eig_h_std, v_h = power_top_eig(model, loss_fn, te, device, power_iters=30)
    v_h_cpu = v_h.detach().cpu().reshape(-1)  # D
    
    # Verify dimension matches
    D_total = sum(p.numel() for p in model.parameters())
    if v_h_cpu.numel() != D_total:
        raise ValueError(f"Eigenvector dim {v_h_cpu.numel()} != total params {D_total}")
    
    # CRITICAL FIX: Estimate alignment across multiple batches
    alignments = []
    eigs_c_list = []
    
    for batch_idx, (xb, yb) in enumerate(eval_batches):
        # Compute per-example gradients (memory-safe: pushed to CPU) - D x B
        G_cpu = per_example_gradients_cpu(model, xb, yb, loss_fn)  # D x B
        B = G_cpu.size(1)  # Second dim is batch size
        
        if B <= 1:
            print(f"[Warning] Batch {batch_idx}: B={B} too small, skipping")
            continue
        
        # Mean-center gradients (D x B)
        mean_g = G_cpu.mean(dim=1, keepdim=True)  # D x 1
        Gc_cpu = G_cpu - mean_g  # D x B
        
        # CRITICAL FIX #1: Compute alignment WITHOUT forming DxD covariance
        # v^T C v = (v^T Gc) (Gc^T v) / (B-1) = ||Gc^T v||^2 / (B-1)
        # Gc_cpu is D x B, v_h_cpu is D
        # Compute Gc^T v = (B x D) @ (D) -> (B)
        Gc_T = Gc_cpu.t()  # B x D
        gTv = Gc_T @ v_h_cpu  # B
        alignment = (gTv.pow(2).sum().item()) / (B - 1)
        alignments.append(alignment)
        
        # CRITICAL FIX #2: Compute top eigenvalue of C via Gram matrix (BxB)
        # Non-zero evals of C = evals of (Gc^T Gc)/(B-1)
        # Gc_cpu is D x B, so Gc_cpu.t() @ Gc_cpu = (B x D) @ (D x B) = (B x B)
        S = (Gc_cpu.t() @ Gc_cpu).numpy() / (B - 1)  # B x B (small)
        evals = np.linalg.eigvalsh(S)  # sorted ascending
        eig_c_top = float(evals[-1])  # largest eigenvalue
        eigs_c_list.append(eig_c_top)
    
    # Average across batches
    alignment_mean = np.mean(alignments) if alignments else 0.0
    alignment_std = np.std(alignments) if len(alignments) > 1 else 0.0
    eig_c_mean = np.mean(eigs_c_list) if eigs_c_list else 0.0
    
    return {
        'alignment': alignment_mean,
        'alignment_std': alignment_std,
        'eig_c_top': eig_c_mean,
        'eig_h': float(eig_h),
        'ratio': eig_c_mean / (float(eig_h) + 1e-10),
        'num_batches_used': len(alignments),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='SGD Noise Covariance Probe (Memory-Safe)')
    p.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    p.add_argument('--config', type=str, required=True, help='Path to config YAML')
    p.add_argument('--batch_size', type=int, default=32, help='Max batch size for per-example grads')
    p.add_argument('--num_batches', type=int, default=4, help='Number of batches to average over')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--no-save', action='store_true', help='Do not save results')
    args = p.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, te = get_dataloaders(cfg)
    
    model = SmallCNN(num_classes=cfg['model'].get('num_classes', 10))
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()  # Important: use eval mode for determinism
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print("="*70)
    print("SGD Noise Covariance Probe (Memory-Safe, Gram Trick)")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {device}, Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}, Num batches: {args.num_batches}")
    
    try:
        result = estimate_alignment_and_spectrum(
            model, tr, te, loss_fn, device,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            seed=args.seed
        )
        
        print("\n" + "="*70)
        print("Results (averaged over {:d} batches):".format(result['num_batches_used']))
        print("="*70)
        print(f"Hessian top eigenvalue:              {result['eig_h']:.6f}")
        print(f"Alignment (v^T C v):                 {result['alignment']:.6f} ± {result['alignment_std']:.6f}")
        print(f"Gradient covariance top eigenvalue:  {result['eig_c_top']:.6f}")
        print(f"Ratio (eig_C / eig_H):               {result['ratio']:.6f}")
        print("="*70)
        
        # Save results
        if not args.no_save:
            out_dir = Path(cfg.get('probes', {}).get('out_dir', 'results/sgd_noise'))
            out_dir.mkdir(parents=True, exist_ok=True)
            
            ckpt_name = Path(args.checkpoint).stem
            meta_path = out_dir / f"sgd_noise_{ckpt_name}_seed{args.seed}.json"
            with open(meta_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved results to: {meta_path}")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
