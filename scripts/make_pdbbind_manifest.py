from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def parse_index(path: Path) -> dict[str, float]:
    labels = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            pdbid = parts[0].lower()
            if len(pdbid) != 4 or not pdbid.isalnum():
                continue
            try:
                labels[pdbid] = float(parts[3])
            except ValueError:
                continue
    return labels


def index_complex_files(root: Path):
    proteins, ligands = {}, {}
    for p in root.rglob("*_protein.pdb"):
        proteins.setdefault(p.name[:4].lower(), p)
    for pattern in ("*.sdf", "*.mol2"):
        for p in root.rglob(pattern):
            name = p.name.lower()
            if "ligand" not in name:
                continue
            pdbid = name[:4]
            if len(pdbid) == 4 and (pdbid not in ligands or p.suffix.lower() == ".sdf"):
                ligands[pdbid] = p
    return proteins, ligands


def main():
    ap = argparse.ArgumentParser(description="Build a BA-Pred2 manifest from a PDBbind-style directory")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    labels = parse_index(args.index)
    proteins, ligands = index_complex_files(args.root)
    ids = sorted(set(labels) & set(proteins) & set(ligands))
    random.Random(args.seed).shuffle(ids)
    n_train = int(len(ids) * args.train_frac)
    n_val = int(len(ids) * args.val_frac)
    rows = []
    for i, pdbid in enumerate(ids):
        split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        rows.append({"id": pdbid, "protein_path": str(proteins[pdbid].resolve()), "ligand_path": str(ligands[pdbid].resolve()), "affinity": labels[pdbid], "split": split})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} complexes -> {args.out}")


if __name__ == "__main__":
    main()
