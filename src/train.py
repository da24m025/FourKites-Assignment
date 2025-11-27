"""Config-driven training harness.

Usage:
  python -m src.train --config experiments/configs/mvp.yml
"""
import argparse
import yaml
import torch
from pathlib import Path
from src.models.small_cnn import SmallCNN
from src.utils.exp import save_checkpoint, set_seed, get_dataloaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    return p.parse_args()


def build_model(cfg):
    if cfg['model']['name'] == 'small_cnn':
        return SmallCNN(num_classes=cfg['model'].get('num_classes', 10))
    else:
        raise NotImplementedError(cfg['model']['name'])


def build_optimizer(cfg, model):
    opt_cfg = cfg['optimizer']
    if opt_cfg['name'] == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=opt_cfg['lr'], 
                             momentum=opt_cfg.get('momentum', 0.9))
    if opt_cfg['name'] == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=opt_cfg['lr'])
    raise NotImplementedError(opt_cfg['name'])


if __name__ == '__main__':
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(cfg.get('seed', 0))
    model = build_model(cfg).to(device)
    opt = build_optimizer(cfg, model)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    tr_loader, te_loader = get_dataloaders(cfg)
    
    save_dir = Path(cfg['train']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(cfg['train']['epochs']):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            preds = out.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            running_loss += loss.item() * yb.size(0)
        
        train_acc = correct / total
        train_loss = running_loss / total
        print(f"Epoch {epoch} train acc {train_acc:.4f} loss {train_loss:.4f}")
    
    # Final save with metadata
    ckpt_path = save_dir / cfg['train']['save_name']
    model.eval()
    val_tot, val_corr = 0, 0
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            val_tot += yb.size(0)
            val_corr += (preds == yb).sum().item()
    val_acc = val_corr / val_tot
    
    metrics = {'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc}
    save_checkpoint(ckpt_path, model, opt, cfg, epoch=cfg['train']['epochs']-1, metrics=metrics)
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Final validation accuracy: {val_acc:.4f}")
