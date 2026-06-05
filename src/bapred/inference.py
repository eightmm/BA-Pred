import os
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from bapred.data.data import BAPredDataset, collate_pyg
from bapred.model.model import PredictionPKD


WEIGHT_DIR = Path(__file__).resolve().parent / "weight"
DEFAULT_WEIGHT = str(WEIGHT_DIR / "random" / "cutoff8_seed0_best.pth")


def inference(protein_pdb, ligand_file, output, batch_size, weight=DEFAULT_WEIGHT, device='cpu', num_workers=0):
    dataset = BAPredDataset(protein_pdb=protein_pdb, ligand_file=ligand_file)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_pyg,
        num_workers=num_workers, persistent_workers=(num_workers > 0)
    )

    weights = [weight] if isinstance(weight, str) else list(weight)
    models = []
    for w in weights:
        m = PredictionPKD(57, 256, 13, 25, 20, 6, 0.2).to(device)
        m.load_state_dict(torch.load(w, map_location=device, weights_only=False)['model_state_dict'])
        m.train(False)  # eval mode
        models.append(m)

    results = {"Name": [], "pKd": [], "Kcal/mol": []}

    with torch.no_grad():
        progress_bar = tqdm(total=len(loader.dataset), unit='ligand')

        for data in loader:
            bgp, bgl, bgc, error, idx, name = data
            bgp, bgl, bgc = bgp.to(device), bgl.to(device), bgc.to(device)

            # ensemble: mean prediction across weights (single weight -> unchanged)
            pkd = torch.stack([m(bgp, bgl, bgc).view(-1) for m in models], dim=0).mean(dim=0)
            pkd[error == 1] = float('nan')

            results["Name"].extend(str(item) for item in name)
            results["pKd"].extend(pkd.tolist())
            results["Kcal/mol"].extend((pkd / -0.73349).tolist())
            progress_bar.update(len(idx))

        progress_bar.close()

    df = pd.DataFrame(results).round(4)
    df.to_csv(output, sep='\t', na_rep='NaN', index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='BA-Pred: Predict protein-ligand binding affinity using Graph Neural Networks'
    )
    parser.add_argument('-r', '--protein_pdb', required=True, help='Receptor protein PDB file')
    parser.add_argument('-l', '--ligand_file', required=True, help='Ligand file (.sdf, .mol2, or .txt list)')
    parser.add_argument('-o', '--output', default='./result.tsv', help='Output results file (default: result.tsv)')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for inference (default: 128)')
    parser.add_argument('--ncpu', default=4, type=int, help='Number of CPU workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Compute device: cpu or cuda (default: cuda)')
    parser.add_argument('--weight', type=str, nargs='+', default=[DEFAULT_WEIGHT],
                        help='Model weight file(s); multiple are ensembled by mean (default: packaged cutoff8_seed0)')

    args = parser.parse_args()

    assert os.path.isfile(args.protein_pdb), f"Protein PDB not found: {args.protein_pdb}"
    assert os.path.isfile(args.ligand_file), f"Ligand file not found: {args.ligand_file}"
    for w in args.weight:
        assert os.path.isfile(w), f"Weight file not found: {w}"

    os.environ["OMP_NUM_THREADS"] = str(args.ncpu)
    os.environ["MKL_NUM_THREADS"] = str(args.ncpu)
    torch.set_num_threads(args.ncpu)

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    inference(
        protein_pdb=args.protein_pdb,
        ligand_file=args.ligand_file,
        output=args.output,
        batch_size=args.batch_size,
        weight=args.weight,
        device=device,
        num_workers=args.ncpu,
    )


if __name__ == "__main__":
    main()
