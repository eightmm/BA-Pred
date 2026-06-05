<div align="center">

# BA-Pred

**Protein-Ligand Binding Affinity Prediction using Graph Neural Networks**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.7+-green.svg)](https://pyg.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CASP16](https://img.shields.io/badge/CASP16-2nd%20Place-gold.svg)](https://predictioncenter.org/casp16/)
[![GitHub stars](https://img.shields.io/github/stars/eightmm/BA-Pred.svg?style=social&label=Star)](https://github.com/eightmm/BA-Pred)

*High-performance protein-ligand binding affinity prediction model - 2nd place in CASP16 ligand affinity challenge*

</div>

## Overview

BA-Pred predicts protein-ligand binding affinity (pKd) directly from a protein
structure and one or more docked ligand poses. It builds three graphs per
complex — protein pocket, ligand, and the protein-ligand interface — and lets
them exchange information at every layer.

**How it works**
1. **Pocket extraction** — protein residues with any atom within 8 Å of the ligand.
2. **Graph construction** — protein/ligand atom graphs (covalent bonds) + a complex
   interaction graph (atom pairs within 5 Å, hydrogen-bond / electrostatic /
   hydrophobic features + distance RBF). Each node carries a 20-step random-walk
   positional encoding (LSPE).
3. **Message passing** — 3 parallel stacks (protein / ligand / complex) of 6
   `GatedGCNLSPE` layers (`emb=256`). After each layer the protein and ligand node
   states are stitched into the complex graph and fed back, coupling the three views.
4. **Readout** — sum-pool ligand nodes → MLP → **pKd**.

> Companion model: **[RMSD-Pred](https://github.com/eightmm/RMSD-Pred)** scores
> binding-pose quality (predicted RMSD). Typical pipeline: dock → filter poses
> with RMSD-Pred → rank affinity with BA-Pred.

## Quick Start

```bash
pip install bapred
```

> **GPU (incl. NVIDIA Blackwell / sm_120):** install a CUDA build of PyTorch.
> From a checkout, `uv sync` uses the bundled CUDA 12.8 index. With pip:
> `pip install torch --index-url https://download.pytorch.org/whl/cu128`

Run a prediction:

```bash
bapred -r protein.pdb -l ligands.sdf -o results.tsv
```

Predictions will be saved in `results.tsv`.

## Usage

Installed CLI:

```bash
bapred -r protein.pdb -l ligands.sdf -o results.tsv
```

Use a different weight:

```bash
bapred -r protein.pdb -l ligands.sdf -o results.tsv --weight /path/to/checkpoint.pth
```

From a source checkout:

```bash
python -m bapred.inference -r example/1KLT.pdb -l example/ligands.sdf -o results.tsv
```

Python API:

```python
from bapred.inference import inference

inference(
    protein_pdb="example/1KLT.pdb",
    ligand_file="example/ligands.sdf",
    output="results.tsv",
    batch_size=128,
    device="cuda",
)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-r, --protein_pdb` | Receptor protein PDB file | required |
| `-l, --ligand_file` | Ligand file (`.sdf` / `.mol2` / `.dlg` / `.pdbqt` / `.txt`) | required |
| `-o, --output` | Output TSV file | `result.tsv` |
| `--weight` | Model weight file | packaged `random/cutoff8_seed0_best.pth` |
| `--batch_size` | Batch size | `128` |
| `--ncpu` | CPU threads / DataLoader workers | `4` |
| `--device` | `cuda` or `cpu` | `cuda` |

## Input/Output Formats

### Input
- **Protein**: PDB format (`.pdb`)
- **Ligands**: one of
  - `.sdf` — SD file (multi-molecule supported)
  - `.mol2` — Tripos MOL2 (multi-molecule supported)
  - `.dlg` / `.pdbqt` — AutoDock poses (parsed via Meeko)
  - `.txt` — a list of any of the above file paths, one per line

### Output
Tab-separated file with columns:
- `Name` — ligand identifier (`_Name` property if present, else `<file>_<index>`)
- `pKd` — predicted binding affinity (pKd scale)
- `Kcal/mol` — binding energy in kcal/mol (`pKd / -0.73349`)

Ligands that fail to parse/build yield `NaN`.

## Weights

Packaged checkpoints live in `src/bapred/weight/random/` (random split, 8 Å
pocket cutoff, seeds 0–2). The default is `cutoff8_seed0_best.pth`. Pass an
ensemble member with `--weight` and average predictions for best accuracy.

## Project Structure

```
BA-Pred/
├── src/
│   └── bapred/            # Main package
│       ├── data/          # Graph construction, atom/bond features, loaders
│       ├── model/         # GatedGCNLSPE, GraphGPS, PredictionPKD (PyG)
│       ├── weight/        # Packaged model weights
│       └── inference.py   # Inference engine + CLI
├── example/               # Sample protein + ligand library
├── tests/                 # Smoke tests (dataset build + forward)
├── pyproject.toml
└── README.md
```

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## Citation

If you use BA-Pred in your research, please cite the paper:

```bibtex
@article{Sim_2026,
  title={BA-Pred and RMSD-Pred: Integrated Graph Neural Network Models for Accurate Protein-Ligand Binding Affinity and Binding Pose Prediction},
  author={Sim, Jaemin and Lee, Juyong},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  month={apr},
  doi={10.1021/acs.jcim.5c02591},
  url={https://doi.org/10.1021/acs.jcim.5c02591}
}
```

---

<div align="center">

**Made with care for the scientific community**

Star us on GitHub if this project helped you!

</div>
