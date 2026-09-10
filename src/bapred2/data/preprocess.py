from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import HeteroData
from tqdm import tqdm

from bapred2.config import GraphConfig, load_config
from .features import add_spatial_edges, atom_property_masks, bonded_edges, coords_tensor, local_direction, node_feature_tensor, random_walk_pe, rbf_distance


def _load_ligand(path: str | Path) -> Chem.Mol:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".sd"}:
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
    else:
        raise ValueError(f"Unsupported ligand format: {path}")
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not parse 3D ligand: {path}")
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        pass
    return mol


def _load_protein(path: str | Path) -> Chem.Mol:
    mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False, proximityBonding=True)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not parse protein PDB: {path}")
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        pass
    return mol


def _heavy_indices(mol: Chem.Mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]


def _residue_key(atom: Chem.Atom):
    info = atom.GetPDBResidueInfo()
    if info is None:
        return None
    return (info.GetChainId(), info.GetResidueNumber(), info.GetInsertionCode(), info.GetResidueName())


def _select_pocket(protein: Chem.Mol, ligand: Chem.Mol, cutoff: float) -> list[int]:
    p_heavy = _heavy_indices(protein)
    l_heavy = _heavy_indices(ligand)
    pcoord = coords_tensor(protein, p_heavy)
    lcoord = coords_tensor(ligand, l_heavy)
    near_local = torch.where(torch.cdist(pcoord, lcoord).min(dim=1).values < cutoff)[0].tolist()
    near_old = {p_heavy[i] for i in near_local}
    residue_keys = {_residue_key(protein.GetAtomWithIdx(i)) for i in near_old}
    residue_keys.discard(None)
    if not residue_keys:
        return sorted(near_old)
    return [i for i in p_heavy if _residue_key(protein.GetAtomWithIdx(i)) in residue_keys]


def _formal_charge(atom: Chem.Atom) -> float:
    try:
        return float(atom.GetFormalCharge())
    except Exception:
        return 0.0


class ComplexPreprocessor:
    def __init__(self, cfg: GraphConfig):
        self.cfg = cfg

    def _intra_graph(self, mol: Chem.Mol, selected: list[int], spatial_cutoff: float):
        x = node_feature_tensor(mol, selected)
        pos = coords_tensor(mol, selected)
        cov_index, cov_attr = bonded_edges(mol, selected, pos, self.cfg.distance_rbf_dim)
        pe = random_walk_pe(cov_index, len(selected), self.cfg.rwpe_dim)
        edge_index, edge_attr = add_spatial_edges(pos, cov_index, cov_attr, spatial_cutoff, self.cfg.max_spatial_neighbors, self.cfg.distance_rbf_dim)
        direction = local_direction(pos, cov_index)
        return x, pos, pe, edge_index, edge_attr, direction

    def build(self, protein_path: str | Path, ligand_path: str | Path, affinity: float, sample_id: str) -> HeteroData:
        protein = _load_protein(protein_path)
        ligand = _load_ligand(ligand_path)
        p_sel = _select_pocket(protein, ligand, self.cfg.pocket_cutoff)
        l_sel = _heavy_indices(ligand)
        if not p_sel or not l_sel:
            raise ValueError("Empty protein pocket or ligand")

        px, ppos, ppe, pei, pea, pdir = self._intra_graph(protein, p_sel, self.cfg.protein_spatial_cutoff)
        lx, lpos, lpe, lei, lea, ldir = self._intra_graph(ligand, l_sel, self.cfg.ligand_spatial_cutoff)
        dist = torch.cdist(ppos, lpos)
        p_idx, l_idx = torch.where(dist < self.cfg.interface_cutoff)
        if p_idx.numel() == 0:
            raise ValueError("No protein-ligand interface edges inside cutoff")
        d = dist[p_idx, l_idx]

        p_mask = atom_property_masks(protein, p_sel)
        l_mask = atom_property_masks(ligand, l_sel)
        cross_vec = lpos[l_idx] - ppos[p_idx]
        cross_unit = cross_vec / cross_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos_p = (pdir[p_idx] * cross_unit).sum(-1, keepdim=True)
        cos_l = (ldir[l_idx] * (-cross_unit)).sum(-1, keepdim=True)
        p_charge = torch.tensor([_formal_charge(protein.GetAtomWithIdx(i)) for i in p_sel], dtype=torch.float32)
        l_charge = torch.tensor([_formal_charge(ligand.GetAtomWithIdx(i)) for i in l_sel], dtype=torch.float32)
        charge_prod = (p_charge[p_idx] * l_charge[l_idx]).reshape(-1, 1) / 9.0
        charge_abs = (p_charge[p_idx].abs() + l_charge[l_idx].abs()).reshape(-1, 1) / 6.0
        d_scaled = (d / self.cfg.interface_cutoff).reshape(-1, 1)
        interface_attr = torch.cat([rbf_distance(d, self.cfg.distance_rbf_dim, self.cfg.interface_cutoff), p_mask[p_idx], l_mask[l_idx], cos_p, cos_l, charge_prod, charge_abs, d_scaled], dim=-1)

        data = HeteroData()
        data["protein"].x, data["protein"].pos, data["protein"].pos_enc = px, ppos, ppe
        data["ligand"].x, data["ligand"].pos, data["ligand"].pos_enc = lx, lpos, lpe
        data[("protein", "intra", "protein")].edge_index = pei
        data[("protein", "intra", "protein")].edge_attr = pea
        data[("ligand", "intra", "ligand")].edge_index = lei
        data[("ligand", "intra", "ligand")].edge_attr = lea
        data[("protein", "contact", "ligand")].edge_index = torch.stack([p_idx, l_idx], dim=0)
        data[("protein", "contact", "ligand")].edge_attr = interface_attr
        data.y = torch.tensor([float(affinity)], dtype=torch.float32)
        data.sample_id = str(sample_id)
        return data


def preprocess_manifest(manifest: str | Path, out_dir: str | Path, cfg: GraphConfig) -> Path:
    manifest = Path(manifest).resolve()
    out_dir = Path(out_dir).resolve()
    graph_dir = out_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest)
    required = {"id", "protein_path", "ligand_path", "affinity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if "split" not in df.columns:
        df["split"] = "train"

    builder = ComplexPreprocessor(cfg)
    rows = []
    for row in tqdm(df.to_dict("records"), desc="preprocess"):
        protein_path = Path(row["protein_path"])
        ligand_path = Path(row["ligand_path"])
        if not protein_path.is_absolute():
            protein_path = (manifest.parent / protein_path).resolve()
        if not ligand_path.is_absolute():
            ligand_path = (manifest.parent / ligand_path).resolve()
        key = hashlib.sha1(f"{row['id']}|{protein_path}|{ligand_path}".encode()).hexdigest()[:12]
        graph_path = graph_dir / f"{row['id']}_{key}.pt"
        try:
            graph = builder.build(protein_path, ligand_path, float(row["affinity"]), str(row["id"]))
            torch.save(graph, graph_path)
            rows.append({"id": row["id"], "graph_path": str(graph_path), "affinity": float(row["affinity"]), "split": row["split"]})
        except Exception as exc:
            print(f"[skip] {row['id']}: {exc}")

    processed_manifest = out_dir / "processed_manifest.csv"
    pd.DataFrame(rows).to_csv(processed_manifest, index=False)
    return processed_manifest


def main():
    parser = argparse.ArgumentParser(description="Preprocess protein-ligand complexes into BA-Pred2 HeteroData graphs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(preprocess_manifest(args.manifest, args.out, cfg.graph))


if __name__ == "__main__":
    main()
