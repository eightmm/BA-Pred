# BA-Pred2

Experimental successor to BA-Pred using **recurrent graph inference** for protein-ligand binding-affinity prediction.

> Full architecture, data, training, equivariance, stability, and ablation plan: **[SPEC.md](SPEC.md)**

## Core idea

Instead of stacking independent GNN layers, BA-Pred2 learns one shared binding-inference operator and repeatedly applies it:

1. refine the dynamic protein-ligand interface state,
2. send bidirectional protein <-> ligand messages,
3. propagate the binding-conditioned state inside each molecule,
4. re-inject the static initial node/interface representation every cycle.

The recurrent core uses LayerNorm, gated updates and LayerScale. The old BA-Pred MHA path is intentionally absent.

## Graphs

Protein and ligand are separate PyG `HeteroData` node types.

- protein intra graph: covalent + spatial edges
- ligand intra graph: covalent + spatial edges
- protein -> ligand contact graph: dynamic interface state
- topology anchor: random-walk PE on the covalent graph
- interface features: distance RBF, donor/acceptor-like atom flags, charge terms and local orientation cosines

Default cutoffs are 8 A for pocket extraction, 5 A for protein-ligand contacts, 5 A for protein spatial edges and 4.5 A for ligand spatial edges.

## Raw manifest

```text
id,protein_path,ligand_path,affinity,split
1abc,/data/1abc_protein.pdb,/data/1abc_ligand.sdf,7.21,train
```

`affinity` should be a pKd/pKi-like `-log10(M)` target.

### Build a PDBbind manifest

```bash
python scripts/make_pdbbind_manifest.py \
  --root /data/PDBbind \
  --index /data/PDBbind/index/INDEX_general_PL_data.2020 \
  --out data/pdbbind.csv
```

The helper uses column 4 of a standard PDBbind index as the target. The default split is random; replace it with the exact split required by your benchmark before reporting results.

## Preprocess

```bash
pip install -e .

bapred2-preprocess \
  --manifest data/pdbbind.csv \
  --out data/processed \
  --config configs/bapred2_base.yaml
```

Output: cached PyG graphs plus `data/processed/processed_manifest.csv`.

## Train

```bash
bapred2-train \
  --manifest data/processed/processed_manifest.csv \
  --config configs/bapred2_base.yaml \
  --out runs/base \
  --device cuda
```

Training samples recurrent depth from `[2, 3, 4, 6, 8]`. The default objective is Huber loss and the best checkpoint is selected by validation RMSE.

## Test-time recycle sweep

```bash
bapred2-eval \
  --manifest data/processed/processed_manifest.csv \
  --checkpoint runs/base/best.pt \
  --split test \
  --recycles 1,2,3,4,6,8,12
```

The evaluator reports RMSE, MAE, Pearson, Spearman and the hidden-state update magnitude at each cycle. The important architectural test is whether performance remains stable or improves when inference recurrence is increased at fixed parameter count.

## First ablations

- independent fixed-depth GNN vs shared recurrent block
- covalent-only vs covalent + spatial intra edges
- static-state reinjection on/off
- endpoint interface context on/off
- ligand-only vs ligand + interface readout
- recycle sweep at fixed parameters

The current implementation uses contacts sharing a protein or ligand atom as a batching-friendly interface-context operator. Exact P-P-L / P-L-L topology-aware triangle updates are intentionally left as the next extension rather than being approximated silently.

## Status

Training and preprocessing code are implemented. Scientific validation still requires preprocessing the target dataset and running the benchmark experiments.
