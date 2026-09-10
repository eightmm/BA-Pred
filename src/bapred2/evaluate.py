from __future__ import annotations

import argparse
import json

import torch
from torch_geometric.loader import DataLoader

from bapred2.config import load_config
from bapred2.data.dataset import ProcessedComplexDataset
from bapred2.model import model_from_sample
from bapred2.train import run_eval


def main():
    ap = argparse.ArgumentParser(description="Evaluate BA-Pred2 and sweep test-time recycles")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/bapred2_base.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--recycles", default="1,2,3,4,6,8,12")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = ProcessedComplexDataset(args.manifest, args.split)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model_from_sample(ds[0], cfg.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    results = {}
    for r in [int(x) for x in args.recycles.split(",")]:
        results[str(r)] = run_eval(model, loader, device, r)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
