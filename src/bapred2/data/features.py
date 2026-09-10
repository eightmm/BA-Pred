from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from rdkit import Chem

ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se", "METAL", "OTHER"]
METALS = {
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI",
    "CU", "ZN", "Y", "ZR", "NB", "MO", "RU", "RH", "PD", "AG", "CD", "HF", "TA", "W", "RE", "OS", "IR",
    "PT", "AU", "HG", "AL", "GA", "IN", "SN", "PB", "BI", "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB",
    "DY", "HO", "ER", "TM", "YB", "LU",
}
HYBRIDS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices: Iterable) -> list[float]:
    choices = list(choices)
    return [1.0 if value == c else 0.0 for c in choices]


def safe_call(fn, default=0):
    try:
        return fn()
    except Exception:
        return default


def atom_features(atom: Chem.Atom) -> list[float]:
    symbol = atom.GetSymbol()
    symbol_key = "METAL" if symbol.upper() in METALS else (symbol if symbol in ELEMENTS else "OTHER")
    degree = min(int(safe_call(atom.GetDegree, 0)), 6)
    total_h = min(int(safe_call(atom.GetTotalNumHs, 0)), 4)
    formal_charge = max(-3, min(3, int(safe_call(atom.GetFormalCharge, 0)))) / 3.0
    mass = float(safe_call(atom.GetMass, 0.0)) / 200.0
    atomic_num = float(atom.GetAtomicNum()) / 100.0
    feat = []
    feat += one_hot(symbol_key, ELEMENTS)
    feat += one_hot(degree, range(7))
    feat += one_hot(total_h, range(5))
    feat += one_hot(safe_call(atom.GetHybridization, Chem.rdchem.HybridizationType.UNSPECIFIED), HYBRIDS)
    feat += [
        float(safe_call(atom.GetIsAromatic, False)),
        float(safe_call(atom.IsInRing, False)),
        formal_charge,
        mass,
        atomic_num,
    ]
    return feat


def node_feature_tensor(mol: Chem.Mol, atom_indices: list[int] | None = None) -> torch.Tensor:
    indices = atom_indices if atom_indices is not None else list(range(mol.GetNumAtoms()))
    return torch.tensor([atom_features(mol.GetAtomWithIdx(i)) for i in indices], dtype=torch.float32)


def coords_tensor(mol: Chem.Mol, atom_indices: list[int] | None = None) -> torch.Tensor:
    conf = mol.GetConformer()
    indices = atom_indices if atom_indices is not None else list(range(mol.GetNumAtoms()))
    return torch.tensor([list(conf.GetAtomPosition(i)) for i in indices], dtype=torch.float32)


def rbf_distance(distance: torch.Tensor, dim: int = 16, cutoff: float = 8.0) -> torch.Tensor:
    distance = distance.reshape(-1, 1)
    centers = torch.linspace(0.0, cutoff, dim, device=distance.device, dtype=distance.dtype).reshape(1, -1)
    width = cutoff / max(dim - 1, 1)
    gamma = 1.0 / max(width * width, 1e-6)
    return torch.exp(-gamma * (distance - centers) ** 2)


def random_walk_pe(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    if num_nodes == 0:
        return torch.zeros((0, k), dtype=torch.float32)
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, k), dtype=torch.float32)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(-1, keepdim=True).clamp_min(1.0)
    rw = adj / deg
    out = rw
    pe = [torch.diagonal(out)]
    for _ in range(k - 1):
        out = out @ rw
        pe.append(torch.diagonal(out))
    return torch.stack(pe, dim=-1)


def atom_property_masks(mol: Chem.Mol, atom_indices: list[int] | None = None) -> torch.Tensor:
    """Five stable atom-local chemistry flags: HBA, HBD, cationic, anionic, hydrophobic."""
    indices = atom_indices if atom_indices is not None else list(range(mol.GetNumAtoms()))
    rows = []
    for idx in indices:
        atom = mol.GetAtomWithIdx(idx)
        z = atom.GetAtomicNum()
        charge = int(safe_call(atom.GetFormalCharge, 0))
        total_h = int(safe_call(atom.GetTotalNumHs, 0))
        aromatic = bool(safe_call(atom.GetIsAromatic, False))
        hba = float(z in {7, 8, 9, 15, 16, 17, 35, 53} and charge <= 0)
        hbd = float(z in {7, 8, 16} and total_h > 0 and charge >= 0)
        cationic = float(charge > 0)
        anionic = float(charge < 0)
        hydrophobic = float(z in {6, 9, 16, 17, 35, 53} and charge == 0 and (aromatic or z != 16))
        rows.append([hba, hbd, cationic, anionic, hydrophobic])
    return torch.tensor(rows, dtype=torch.float32)


def local_direction(coords: torch.Tensor, covalent_edge_index: torch.Tensor) -> torch.Tensor:
    """Rotation-equivariant local outward vector used only through invariant cosines."""
    n = coords.size(0)
    nbrs: dict[int, list[int]] = defaultdict(list)
    for s, d in covalent_edge_index.t().tolist():
        nbrs[d].append(s)
    out = torch.zeros_like(coords)
    for i in range(n):
        if nbrs[i]:
            mean_nbr = coords[torch.tensor(nbrs[i], dtype=torch.long)].mean(0)
            out[i] = coords[i] - mean_nbr
    norm = out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return out / norm


def bonded_edges(mol: Chem.Mol, selected: list[int], coords: torch.Tensor, rbf_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    old_to_new = {old: new for new, old in enumerate(selected)}
    src, dst, feats = [], [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in old_to_new or b not in old_to_new:
            continue
        bf = one_hot(bond.GetBondType(), BOND_TYPES) + [
            float(bond.GetIsConjugated()), float(bond.IsInRing()), float(bond.GetIsAromatic())
        ]
        for u_old, v_old in ((a, b), (b, a)):
            u, v = old_to_new[u_old], old_to_new[v_old]
            dist = torch.norm(coords[u] - coords[v]).reshape(1)
            feat = torch.tensor(bf + [1.0, 0.0], dtype=torch.float32)
            feat = torch.cat([feat, rbf_distance(dist, rbf_dim, 8.0).squeeze(0)])
            src.append(u)
            dst.append(v)
            feats.append(feat)
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.stack(feats) if feats else torch.empty((0, 4 + 3 + 2 + rbf_dim), dtype=torch.float32)
    return edge_index, edge_attr


def add_spatial_edges(
    coords: torch.Tensor,
    covalent_edge_index: torch.Tensor,
    covalent_edge_attr: torch.Tensor,
    cutoff: float,
    max_neighbors: int,
    rbf_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = coords.size(0)
    if n <= 1:
        return covalent_edge_index, covalent_edge_attr
    bonded = set(map(tuple, covalent_edge_index.t().tolist()))
    dist = torch.cdist(coords, coords)
    spatial_src, spatial_dst, spatial_feat = [], [], []
    for dst in range(n):
        candidates = torch.where((dist[:, dst] < cutoff) & (dist[:, dst] > 1e-6))[0]
        if candidates.numel() > max_neighbors:
            order = torch.argsort(dist[candidates, dst])[:max_neighbors]
            candidates = candidates[order]
        for src in candidates.tolist():
            if (src, dst) in bonded:
                continue
            base = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
            feat = torch.cat([base, rbf_distance(dist[src, dst].reshape(1), rbf_dim, 8.0).squeeze(0)])
            spatial_src.append(src)
            spatial_dst.append(dst)
            spatial_feat.append(feat)
    if not spatial_src:
        return covalent_edge_index, covalent_edge_attr
    sp_index = torch.tensor([spatial_src, spatial_dst], dtype=torch.long)
    sp_attr = torch.stack(spatial_feat)
    return torch.cat([covalent_edge_index, sp_index], dim=1), torch.cat([covalent_edge_attr, sp_attr], dim=0)
