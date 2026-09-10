from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from bapred2.config import load_config
from bapred2.data.dataset import ProcessedComplexDataset
from bapred2.model import model_from_sample


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_recycles(values: list[int], probs: list[float]) -> int:
    if len(values) != len(probs):
        raise ValueError("train_recycles and train_recycle_probs must have the same length")
    return random.choices(values, weights=probs, k=1)[0]


def loss_fn(pred, target, name: str, delta: float):
    if name == "mse":
        return F.mse_loss(pred, target)
    if name == "mae":
        return F.l1_loss(pred, target)
    if name == "huber":
        return F.huber_loss(pred, target, delta=delta)
    raise ValueError(f"Unknown loss: {name}")


def metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    p = float(pearsonr(a, b).statistic) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
    s = float(spearmanr(a, b).statistic) if len(a) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "pearson": p, "spearman": s}


def run_eval(model, loader, device, recycles: int):
    model.eval()
    ys, ps, cycle_deltas = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, aux = model(batch, recycles=recycles, return_aux=True)
            target = batch.y.reshape(-1)
            ys.extend(target.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            if aux["cycle_delta"].numel():
                cycle_deltas.append(aux["cycle_delta"].cpu())
    out = metrics(ys, ps)
    if cycle_deltas:
        out["cycle_delta"] = torch.stack(cycle_deltas).mean(0).tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description="Train BA-Pred2")
    ap.add_argument("--manifest", required=True, help="processed_manifest.csv from bapred2-preprocess")
    ap.add_argument("--config", default="configs/bapred2_base.yaml")
    ap.add_argument("--out", default="runs/bapred2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ProcessedComplexDataset(args.manifest, "train")
    val_ds = ProcessedComplexDataset(args.manifest, "val")
    try:
        test_ds = ProcessedComplexDataset(args.manifest, "test")
    except ValueError:
        test_ds = None

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers) if test_ds else None

    model = model_from_sample(train_ds[0], cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_rmse, bad_epochs, history = math.inf, 0, []
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        losses = []
        bar = tqdm(train_loader, desc=f"epoch {epoch:03d}")
        for batch in bar:
            batch = batch.to(device)
            recycles = sample_recycles(cfg.model.train_recycles, cfg.model.train_recycle_probs)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(batch, recycles=recycles)
                target = batch.y.reshape(-1)
                loss = loss_fn(pred, target, cfg.train.loss, cfg.train.huber_delta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}", r=recycles)

        scheduler.step()
        val = run_eval(model, val_loader, device, cfg.model.eval_recycles)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": scheduler.get_last_lr()[0], "val": val}
        history.append(record)
        print(json.dumps(record, indent=2))
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, out_dir / "last.pt")
        if val["rmse"] < best_rmse:
            best_rmse, bad_epochs = val["rmse"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if test_loader is not None:
        ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        result = run_eval(model, test_loader, device, cfg.model.eval_recycles)
        print("test", json.dumps(result, indent=2))
        (out_dir / "test.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
