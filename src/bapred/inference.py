"""Inference helpers for BA-Pred."""

from pathlib import Path
import random
from typing import Union

import numpy as np
import pandas as pd
import torch
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

from bapred.data.data import BAPredDataset
from bapred.model.model import PredictionPKD
from bapred.weights import DEFAULT_MODEL, resolve_packaged_weight


DEFAULT_SEED = 42


def _set_reproducible_seed(seed: int = DEFAULT_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_weight_file(model: str, model_path: str | None) -> Path:
    if model_path is not None:
        candidate = Path(model_path)
        return candidate / "BAPred.pth" if candidate.is_dir() else candidate
    return resolve_packaged_weight(model)


def inference(
    protein_pdb: str,
    ligand_file: str,
    output: str,
    batch_size: int,
    model: str = DEFAULT_MODEL,
    model_path: str | None = None,
    device: Union[str, torch.device] = "cpu",
    use_mha: bool = False,
) -> None:
    _set_reproducible_seed()

    resolved_device = torch.device(device)
    dataset = BAPredDataset(protein_pdb=protein_pdb, ligand_file=ligand_file)
    loader = GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=resolved_device.type != "cpu",
        num_workers=0,
    )

    model = PredictionPKD(57, 256, 13, 25, 20, 6, 0.2, use_mha=use_mha).to(resolved_device)
    weight_path = _resolve_weight_file(model, model_path)
    checkpoint = torch.load(weight_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = {"Name": [], "pKd": [], "Kcal/mol": []}

    with torch.no_grad():
        progress_bar = tqdm(total=len(loader.dataset), unit="ligand")

        for data in loader:
            bgp, bgl, bgc, error, idx, name = data
            bgp = bgp.to(resolved_device)
            bgl = bgl.to(resolved_device)
            bgc = bgc.to(resolved_device)

            pkd = model(bgp, bgl, bgc).view(-1)
            pkd[error == 1] = float("nan")

            results["Name"].extend(str(item) for item in name)
            results["pKd"].extend(pkd.tolist())
            results["Kcal/mol"].extend((pkd / -0.73349).tolist())
            progress_bar.update(len(idx))

        progress_bar.close()

    df = pd.DataFrame(results).round(4)
    df.to_csv(output, sep="\t", na_rep="NaN", index=False)
