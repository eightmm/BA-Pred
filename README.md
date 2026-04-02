<div align="center">

# BA-Pred

**Protein-Ligand Binding Affinity Prediction using Graph Neural Networks**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange.svg)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-2.4-green.svg)](https://www.dgl.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CASP16](https://img.shields.io/badge/CASP16-2nd%20Place-gold.svg)](https://predictioncenter.org/casp16/)
[![GitHub stars](https://img.shields.io/github/stars/eightmm/BA-Pred.svg?style=social&label=Star)](https://github.com/eightmm/BA-Pred)

*High-performance protein-ligand binding affinity prediction model - 2nd place in CASP16 ligand affinity challenge*

</div>

## Quick Start

```bash
pip install bapred
```

Run a prediction:

```bash
bapred -r protein.pdb -l ligands.sdf -o results.tsv
```

By default, the installed CLI uses the packaged `random_seed0` checkpoint.
Predictions will be saved in `results.tsv`.

## Usage

Installed CLI:

```bash
bapred -r protein.pdb -l ligands.sdf -o results.tsv
```

Choose another packaged random checkpoint:

```bash
bapred -r protein.pdb -l ligands.sdf -o results.tsv --model random_seed1
```

From a source checkout:

```bash
python scripts/run_inference.py -r example/1KLT.pdb -l example/ligands.sdf -o results.tsv
```

Python API:

```python
from bapred.inference import inference

inference(
    protein_pdb="example/1KLT.pdb",
    ligand_file="example/ligands.sdf",
    output="results.tsv",
    batch_size=128,
    model="random_seed0",
    device="cuda"
)
```

## Project Structure

```
BA-Pred/
├── src/
│   └── bapred/            # Main package
│       ├── data/          # Data processing modules
│       ├── model/         # Neural network models
│       ├── weight/        # Packaged model weights and presets
│       ├── weights.py     # Checkpoint preset definitions
│       ├── cli.py         # Installed CLI entry point
│       └── inference.py   # Inference engine
├── example/               # Example files
│   ├── 1KLT.pdb             # Sample protein structure
│   └── ligands.sdf          # Sample ligand library
├── scripts/
│   └── run_inference.py   # Source-tree inference wrapper
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Package configuration
└── README.md            # You are here!
```

## Input/Output Formats

### Input
- **Protein**: PDB format (`.pdb`)
- **Ligands**: SDF (`.sdf`), MOL2 (`.mol2`), or text file with paths (`.txt`)

### Output
- **CSV/TSV file** with columns:
  - `Name`: Ligand identifier
  - `pKd`: Predicted binding affinity (pKd scale)
  - `Kcal/mol`: Binding energy in kcal/mol

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
