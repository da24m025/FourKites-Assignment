"""Hessian spectral analysis using Lanczos algorithm.

Provides robust estimation of top-k and smallest eigenvalues using Lanczos iteration,
which is more efficient and stable than repeated power iteration for full spectrum analysis.

For research-grade loss landscape analysis, Lanczos gives:
- Top-k eigenvalues (λ_1, λ_2, ..., λ_k)
- Smallest eigenvalues (enables condition number λ_max/λ_min)
- Spectral density information

Usage:
  from src.probes.hessian_lanczos import lanczos_spectrum
  eigs = lanczos_spectrum(model, loss_fn, data_loader, device, k=5)

IMPORTANT NOTES:
- Computes loss fresh inside each matvec call (no stale autograd graphs)
- Handles None gradients and frozen parameters safely
- SciPy LinearOperator wrapping ensures compatibility
- Hessians can be indefinite; condition number may be negative or ill-defined
"""

import torch
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from typing import Tuple, Optional, List


class HessianOperator:
    """Wrapper to use Hessian as a linear operator for scipy.sparse.linalg.eigsh.
    
    KEY: Computes forward pass and loss INSIDE matvec() to ensure fresh autograd graph.
    This prevents stale gradients and enables correct second-order derivatives.
    """
    
    def __init__(self, model, loss_fn, batch_x, batch_y, device):
        """
        Args:
            model: neural network (will be called with batch_x in matvec)
            loss_fn: loss function
            batch_x: input batch tensor (stored, reused in each matvec)
            batch_y: target batch tensor (stored, reused in each matvec)
            device: compute device ('cpu' or 'cuda')
        """
        self.model = model
        self.loss_fn = loss_fn
        self.batch_x = batch_x.to(device)
        self.batch_y = batch_y.to(device)
        self.device = device
        
        # Store parameter info
        self.params = list(model.parameters())
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        self.dim = sum(self.numels)
        
        # Store dtype (assume all params have same dtype)
        self.param_dtype = self.params[0].dtype if len(self.params) > 0 else torch.float32
    
    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute H*v (Hessian-vector product).
        
        CRITICAL: Computes loss FRESH each call (no no_grad).
        This ensures autograd graph exists for second derivatives.
        """
        # Convert numpy vector to torch with correct dtype & device
        v_torch = torch.from_numpy(v).to(dtype=self.param_dtype, device=self.device)
        
        # Forward pass (fresh, inside matvec, WITHOUT no_grad!)
        out = self.model(self.batch_x)
        loss = self.loss_fn(out, self.batch_y)
        
        # First-order gradients (create_graph=True for second derivatives)
        grads = torch.autograd.grad(loss, self.params, create_graph=True, 
                                   retain_graph=True, allow_unused=True)
        
        # Flatten gradients, replacing None with zeros
        flat_grads = torch.cat([
            (g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1))
            for g, p in zip(grads, self.params)
        ])
        
        # Inner product: g^T v
        g_v = (flat_grads * v_torch).sum()
        
        # Second derivative (Hessian-vector product)
        # Use retain_graph=False on final grad to free memory
        hvp = torch.autograd.grad(g_v, self.params, retain_graph=False, 
                                 allow_unused=True)
        
        # Flatten HVP, replacing None with zeros
        hv_flat = torch.cat([
            (h.contiguous().view(-1) if h is not None else torch.zeros_like(p).view(-1))
            for h, p in zip(hvp, self.params)
        ])
        
        # Return as numpy (scipy expects float64)
        return hv_flat.detach().cpu().numpy().astype(np.float64)
    
    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        """For symmetric Hessian: rmatvec = matvec."""
        return self.matvec(v)


def lanczos_spectrum(model, loss_fn, data_loader, device, 
                    k: int = 5, num_batches: int = 1,
                    return_eigenvectors: bool = False,
                    verbose: bool = True) -> dict:
    """Estimate Hessian spectrum using Lanczos iteration.
    
    Computes top-k and smallest eigenvalues via Lanczos algorithm, providing
    robust spectral information for loss landscape analysis.
    
    KEY DETAILS:
    - Loss is computed FRESH inside each matvec call (prevents stale gradients)
    - Handles None gradients and frozen parameters safely
    - Wraps model Hessian as scipy LinearOperator for eigsh
    - Returns both top-k (largest algebraic) and smallest eigenvalues
    - Hessians can be indefinite; condition number semantics explained below
    
    Args:
        model: trained neural network (will be called in eval mode)
        loss_fn: loss function
        data_loader: dataloader for batch sampling
        device: compute device ('cpu' or 'cuda')
        k: number of eigenvalues to compute (default 5; must be < dim-1)
        num_batches: number of batches to average spectrum over (default 1)
        return_eigenvectors: if True, also return eigenvectors as numpy arrays
        verbose: if True, print diagnostics
    
    Returns:
        dict with keys:
            'top_eigenvalues': list of k largest algebraic eigenvalues
            'top_eigenvalue_stds': std of top eigenvalue across batches
            'smallest_eigenvalues': list of k smallest algebraic eigenvalues
            'lambda_max': largest eigenvalue
            'lambda_min': smallest eigenvalue
            'condition_number': λ_max / λ_min (may be negative if Hessian indefinite)
            'condition_number_magnitude': |λ_max / λ_min| (more robust for indefinite)
            'n_parameters': total number of model parameters
            'k': number of eigenvalues computed
            'num_batches_used': int
            'status': 'success' or 'failed'
            'warning': str (if condition number ill-defined)
    """
    model.to(device).eval()
    D = sum(p.numel() for p in model.parameters())
    
    if verbose:
        print(f"[Lanczos] Estimating spectrum for {D} parameters, k={k}, num_batches={num_batches}")
    
    # Validate k
    if k <= 0:
        raise ValueError(f"k must be > 0, got k={k}")
    if D <= 2:
        raise RuntimeError(f"Model too small for eigsh. Need dim > 2, got D={D}")
    k_use = max(1, min(k, D - 2))
    if k_use != k and verbose:
        print(f"[Lanczos] Adjusted k from {k} to {k_use} (constraint: k < D-1)")
    
    # Collect eigenvalue estimates across batches
    all_top_eigenvalues = []
    all_smallest_eigenvalues = []
    
    data_iter = iter(data_loader)
    
    for batch_idx in range(num_batches):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            xb, yb = next(data_iter)
        
        xb, yb = xb.to(device), yb.to(device)
        
        if verbose:
            print(f"[Lanczos] Processing batch {batch_idx+1}/{num_batches} (batch size: {xb.shape[0]})")
        
        try:
            # Create Hessian operator for this batch
            operator_obj = HessianOperator(model, loss_fn, xb, yb, device)
            
            # Wrap as scipy LinearOperator (CRITICAL: eigsh requires this)
            linop = LinearOperator(
                shape=(operator_obj.dim, operator_obj.dim),
                matvec=operator_obj.matvec,
                rmatvec=operator_obj.rmatvec,
                dtype=np.float64
            )
            
            # Compute top-k eigenvalues (largest algebraic)
            eigsh_result = eigsh(
                linop, k=k_use, which='LA', return_eigenvectors=False
            )
            eigenvalues_top = eigsh_result if isinstance(eigsh_result, np.ndarray) else eigsh_result[0]
            
            # eigsh may return in ascending order; sort descending
            idx_desc = np.argsort(eigenvalues_top)[::-1]
            eigenvalues_top = eigenvalues_top[idx_desc]
            
            all_top_eigenvalues.append(eigenvalues_top)
            
            if verbose:
                print(f"  Top eigenvalues: {eigenvalues_top[:3]}")  # Print top 3
            
        except Exception as e:
            if verbose:
                print(f"[Lanczos] ERROR computing top eigenvalues on batch {batch_idx}: {e}")
            all_top_eigenvalues.append(np.full(k_use, np.nan))
        
        # Compute smallest eigenvalues (largest magnitude gives best stability)
        try:
            # Reuse operator; compute smallest (algebraic)
            linop = LinearOperator(
                shape=(operator_obj.dim, operator_obj.dim),
                matvec=operator_obj.matvec,
                rmatvec=operator_obj.rmatvec,
                dtype=np.float64
            )
            eigsh_result_small = eigsh(
                linop, k=k_use, which='SA', return_eigenvectors=False
            )
            eigenvalues_small = eigsh_result_small if isinstance(eigsh_result_small, np.ndarray) else eigsh_result_small[0]
            eigenvalues_small = np.sort(eigenvalues_small)  # Ascending order
            all_smallest_eigenvalues.append(eigenvalues_small)
            
            if verbose:
                print(f"  Smallest eigenvalues: {eigenvalues_small[:3]}")
            
        except Exception as e:
            if verbose:
                print(f"[Lanczos] WARNING: Could not compute smallest eigenvalues: {e}")
            all_smallest_eigenvalues.append(np.full(k_use, np.nan))
    
    # Average over batches
    if not all_top_eigenvalues or all(np.isnan(e).all() for e in all_top_eigenvalues):
        if verbose:
            print("[Lanczos] ERROR: All batch computations failed!")
        return {
            'status': 'failed',
            'error': 'Lanczos failed on all batches',
            'n_parameters': D,
            'k': k_use,
            'num_batches_used': num_batches,
        }
    
    all_top_eigenvalues = np.array(all_top_eigenvalues)
    mean_top_eigenvalues = np.nanmean(all_top_eigenvalues, axis=0)
    std_top_eigenvalues = np.nanstd(all_top_eigenvalues, axis=0)
    
    all_smallest_eigenvalues = np.array(all_smallest_eigenvalues)
    mean_smallest_eigenvalues = np.nanmean(all_smallest_eigenvalues, axis=0)
    
    lambda_max = mean_top_eigenvalues[0]
    lambda_min = mean_smallest_eigenvalues[0]
    
    # Compute condition number with caveats
    warning = None
    if np.isnan(lambda_max) or np.isnan(lambda_min):
        condition_number = np.nan
        condition_number_magnitude = np.nan
        warning = "NaN in eigenvalue estimates; condition number undefined"
    elif abs(lambda_min) < 1e-10:
        condition_number = np.sign(lambda_max) * np.inf
        condition_number_magnitude = np.inf
        warning = "λ_min near zero; condition number ill-defined or infinite"
    else:
        condition_number = lambda_max / lambda_min
        condition_number_magnitude = abs(lambda_max / lambda_min)
        if lambda_min < 0:
            warning = "Hessian has negative eigenvalues (indefinite); condition number may be negative"
    
    results = {
        'top_eigenvalues': mean_top_eigenvalues.tolist(),
        'top_eigenvalue_stds': std_top_eigenvalues.tolist(),
        'smallest_eigenvalues': mean_smallest_eigenvalues.tolist(),
        'lambda_max': float(lambda_max),
        'lambda_min': float(lambda_min),
        'condition_number': float(condition_number),
        'condition_number_magnitude': float(condition_number_magnitude),
        'n_parameters': int(D),
        'k': int(k_use),
        'num_batches_used': num_batches,
        'status': 'success',
    }
    
    if warning:
        results['warning'] = warning
        if verbose:
            print(f"[Lanczos] WARNING: {warning}")
    
    return results


if __name__ == '__main__':
    # Example usage (standalone test)
    import argparse
    import yaml
    from pathlib import Path
    import json
    from src.utils.exp import get_dataloaders, load_checkpoint, set_seed
    from src.models.small_cnn import SmallCNN
    
    p = argparse.ArgumentParser(description='Estimate Hessian spectrum via Lanczos')
    p.add_argument('--checkpoint', type=str, required=True,
                  help='Path to model checkpoint')
    p.add_argument('--config', type=str, required=True,
                  help='Path to config YAML')
    p.add_argument('--k', type=int, default=5,
                  help='Number of eigenvalues (default 5)')
    p.add_argument('--num_batches', type=int, default=1,
                  help='Batches to average spectrum over (default 1)')
    p.add_argument('--output_dir', type=str, default=None,
                  help='Output directory for results JSON (overrides config)')
    p.add_argument('--seed', type=int, default=None,
                  help='Random seed for reproducibility')
    p.add_argument('--verbose', action='store_true',
                  help='Print detailed diagnostics')
    args = p.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
    
    cfg = yaml.safe_load(open(args.config))
    
    # Determine output directory: use arg if provided, else use config, else default to results
    if args.output_dir is None:
        args.output_dir = cfg.get('probes', {}).get('out_dir', 'results')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, te = get_dataloaders(cfg)
    
    model = SmallCNN(num_classes=cfg['model'].get('num_classes', 10))
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print(f"Computing Lanczos spectrum...")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  k: {args.k}, num_batches: {args.num_batches}")
    print(f"  Device: {device}")
    
    results = lanczos_spectrum(model, loss_fn, te, device, k=args.k, 
                              num_batches=args.num_batches, verbose=args.verbose)
    
    # Display results
    print(f"\n{'='*60}")
    print("Lanczos Spectrum Analysis Results")
    print(f"{'='*60}")
    
    if results['status'] == 'failed':
        print(f"ERROR: {results.get('error', 'Unknown error')}")
    else:
        print(f"Status:                  {results['status']}")
        print(f"Number of parameters:    {results['n_parameters']}")
        print(f"Eigenvalues computed:    {results['k']}")
        print(f"Batches averaged:        {results['num_batches_used']}")
        print(f"\nTop eigenvalues (mean ± std):")
        for i, (lam, std) in enumerate(zip(results['top_eigenvalues'], 
                                          results['top_eigenvalue_stds'])):
            print(f"  λ_{i+1}: {lam:>10.6f} ± {std:.6f}")
        print(f"\nSmallest eigenvalues:")
        for i, lam in enumerate(results['smallest_eigenvalues'][:3]):
            print(f"  λ_min+{i}: {lam:>10.6f}")
        print(f"\nSpectral analysis:")
        print(f"  λ_max (largest):        {results['lambda_max']:>10.6f}")
        print(f"  λ_min (smallest):       {results['lambda_min']:>10.6f}")
        print(f"  Condition number:       {results['condition_number']:>10.6f}")
        print(f"  |λ_max/λ_min|:          {results['condition_number_magnitude']:>10.6f}")
        
        if 'warning' in results:
            print(f"\n  WARNING: {results['warning']}")
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.checkpoint).stem
    output_file = Path(args.output_dir) / f'lanczos_spectrum_{ckpt_name}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
