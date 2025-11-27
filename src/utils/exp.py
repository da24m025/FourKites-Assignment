"""Utilities: dataloaders, seeding, checkpointing."""
import torch
import torchvision.transforms as T
import torchvision
import random
import numpy as np
from pathlib import Path
import json
import subprocess
from datetime import datetime


def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(cfg):
    """Load CIFAR-10 dataloaders with proper normalization and augmentation.
    
    CIFAR-10 statistics (ImageNet-normalized):
    Mean: [0.4914, 0.4822, 0.4465]
    Std:  [0.2470, 0.2435, 0.2616]
    """
    ds = cfg['dataset']
    
    # Normalization transform
    normalize = T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
    
    # Training: with augmentation
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    
    # Testing: no augmentation
    test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    
    train = getattr(torchvision.datasets, ds['name'])(
        root=ds['root'], train=True, download=True, transform=train_transform
    )
    test = getattr(torchvision.datasets, ds['name'])(
        root=ds['root'], train=False, download=True, transform=test_transform
    )
    
    tr = torch.utils.data.DataLoader(
        train, batch_size=ds['train_batch'], shuffle=True, num_workers=2
    )
    te = torch.utils.data.DataLoader(
        test, batch_size=ds['eval_batch'], shuffle=False, num_workers=2
    )
    return tr, te


def save_checkpoint(path: Path, model, optimizer, cfg, epoch=None, metrics=None):
    """Save model checkpoint with rich metadata for reproducibility.
    
    Args:
        path: save path
        model: neural network
        optimizer: optimizer
        cfg: config dict
        epoch: training epoch (optional)
        metrics: dict with train/val metrics (optional)
    """
    # Collect RNG states for reproducibility
    rng_states = {
        'np_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states['cuda_random_state'] = torch.cuda.get_rng_state_all()
    
    # Get git info for version tracking
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        git_dirty = len(subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()) > 0
    except:
        git_hash = None
        git_dirty = None
    
    data = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'cfg': cfg,
        'epoch': epoch,
        'metrics': metrics or {},
        'rng_states': rng_states,
        'git_hash': git_hash,
        'git_dirty': git_dirty,
        'timestamp': datetime.now().isoformat(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    
    # Also save metadata as JSON for easy inspection
    metadata = {k: v for k, v in data.items() if k not in ['model', 'optim', 'rng_states']}
    json_path = Path(path).with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_checkpoint(path: Path, map_location='cpu'):
    """Load checkpoint with metadata.
    
    Returns checkpoint dict with keys: model, optim, cfg, epoch, metrics, rng_states, etc.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_rng_states(checkpoint, device='cpu'):
    """Restore random number generator states from checkpoint for reproducibility."""
    rng_states = checkpoint.get('rng_states', {})
    if 'np_random_state' in rng_states:
        np.random.set_state(rng_states['np_random_state'])
    if 'torch_random_state' in rng_states:
        torch.set_rng_state(rng_states['torch_random_state'])
    if 'cuda_random_state' in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states['cuda_random_state'])
