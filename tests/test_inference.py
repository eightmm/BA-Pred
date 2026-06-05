"""Smoke tests: dataset graph construction + model forward on the example data.

Fast (a couple of ligands on CPU) -- guards against regressions in the PyG
data pipeline and weight loading, not numerical accuracy.
"""
from pathlib import Path

import torch
import pytest

from bapred.data.data import BAPredDataset, collate_pyg
from bapred.model.model import PredictionPKD

ROOT = Path(__file__).resolve().parent.parent
PROT = str(ROOT / "example" / "1KLT.pdb")
LIG = str(ROOT / "example" / "ligands.sdf")
WEIGHT = str(ROOT / "src" / "bapred" / "weight" / "random" / "cutoff8_seed0_best.pth")


@pytest.fixture(scope="module")
def dataset():
    return BAPredDataset(protein_pdb=PROT, ligand_file=LIG)


def test_dataset_builds_pyg_graphs(dataset):
    assert len(dataset) > 0
    gp, gl, gc, error, idx, name = dataset[0]
    assert error == 0
    assert gp.x.shape[1] == 57          # protein node features
    assert gl.x.shape[1] == 57          # ligand node features
    assert gl.pos_enc.shape[1] == 20    # random-walk PE
    assert gc.edge_attr.shape[1] == 25  # complex edge features (interact + distance)
    assert gp.edge_index.shape[0] == 2


def test_model_forward_finite(dataset):
    bgp, bgl, bgc, error, idx, name = collate_pyg([dataset[0], dataset[1]])
    model = PredictionPKD(57, 256, 13, 25, 20, 6, 0.2)
    state = torch.load(WEIGHT, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(state)  # 1:1 load proves param names match the trained checkpoint
    model.train(False)
    with torch.no_grad():
        out = model(bgp, bgl, bgc).view(-1)
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()
