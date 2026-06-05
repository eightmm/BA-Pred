import os
from collections import defaultdict

import torch
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_torch_csr_tensor, to_edge_index, get_self_loop_attr
from meeko import PDBQTMolecule, RDKitMolCreate

from bapred.data.atom_feature import (
    get_mol_coordinate,
    get_atom_feature,
    get_bond_feature,
    get_interact_feature,
    get_distance_feature,
)

RDLogger.DisableLog('rdApp.*')


def _process_dlg_pdbqt(file_path, is_dlg, only_cluster_leads=True):
    """Helper function to process .dlg and .pdbqt files."""
    name = os.path.basename(file_path).split('.')[0]
    pdbqt_mol = PDBQTMolecule.from_file(
        file_path, name=name, is_dlg=is_dlg, skip_typing=True
    )
    rdkit_mols = RDKitMolCreate.from_pdbqt_mol(
        pdbqt_mol, only_cluster_leads=only_cluster_leads, keep_flexres=False
    )
    sdf_string, _ = RDKitMolCreate.write_sd_string(pdbqt_mol, only_cluster_leads=only_cluster_leads)

    adg_score = []
    for line in sdf_string.split('\n'):
        if '{' in line:
            words = line.split(',')
            free_energy = words[1].split(':')[1].strip()
            adg_score.append(float(free_energy))

    mols, err_tags, names = [], [], []
    for i, conf in enumerate(rdkit_mols[0].GetConformers()):
        mol = Chem.Mol(rdkit_mols[0])
        if mol is None:
            mols.append(None)
            err_tags.append(1)
        else:
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            mol = Chem.RemoveHs(mol)
            mols.append(mol)
            err_tags.append(0)
        names.append(f"{name}_{i}")
    return mols, err_tags, names, adg_score


def _process_sdf(file_path):
    """Helper function to process .sdf files."""
    supplier = Chem.SDMolSupplier(file_path, sanitize=False)
    return _process_supplier(supplier, file_path)


def _process_mol2(file_path):
    """Helper function to process .mol2 files"""
    with open(file_path, 'r') as f:
        mol2_data = f.read()
    mol2_blocks = mol2_data.split('@<TRIPOS>MOLECULE')
    supplier = (
        Chem.MolFromMol2Block('@<TRIPOS>MOLECULE' + block, sanitize=False)
        for block in mol2_blocks[1:]
    )
    return _process_supplier(supplier, file_path)


def _process_supplier(supplier, file_path):
    """Common logic for processing SDF and Mol2 suppliers."""
    ligands, err_tag, ligand_names = [], [], []
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for idx, mol in enumerate(supplier):
        if mol is not None:
            mol = Chem.RemoveHs(mol)
            ligands.append(mol)
            err_tag.append(0)
            mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') and mol.GetProp('_Name').strip() else base_name
            ligand_names.append(f"{mol_name}_{idx}")
        else:
            ligands.append(None)
            err_tag.append(1)
            ligand_names.append(f"{base_name}_err_{idx}")

    return ligands, err_tag, ligand_names, [float('nan')] * len(ligands)


def process_ligand_file(file_path, only_cluster_leads=True):
    """Processes a single ligand file (.dlg, .pdbqt, .sdf, .mol2)."""
    extension = os.path.splitext(file_path)[-1].lower()

    if extension == '.dlg':
        return _process_dlg_pdbqt(file_path, is_dlg=True, only_cluster_leads=only_cluster_leads)
    elif extension == '.pdbqt':
        return _process_dlg_pdbqt(file_path, is_dlg=False, only_cluster_leads=only_cluster_leads)
    elif extension == '.sdf':
        return _process_sdf(file_path)
    elif extension == '.mol2':
        return _process_mol2(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def load_ligands(file_path, only_cluster_leads=True):
    """Loads ligands from a file or a list of files."""
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == '.txt':
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        lig_mols, err_tags, lig_names, adg_scores = [], [], [], []
        for line in lines:
            assert os.path.isfile(line), f"File not found: {line}"
            file_ligands, file_err_tag, file_ligand_names, file_adg = process_ligand_file(line, only_cluster_leads=only_cluster_leads)
            lig_mols.extend(file_ligands)
            err_tags.extend(file_err_tag)
            lig_names.extend(file_ligand_names)
            adg_scores.extend(file_adg)
        return lig_mols, err_tags, lig_names, adg_scores

    elif file_extension in ['.sdf', '.mol2', '.dlg', '.pdbqt']:
        return process_ligand_file(file_path, only_cluster_leads=only_cluster_leads)
    else:
        raise ValueError("Unsupported file type. Use '.txt', '.sdf', '.mol2', '.dlg', or '.pdbqt'.")


def random_walk_pe(edge_index, num_nodes, k):
    """Self-return landing probabilities for 1..k steps (matches dgl.random_walk_pe).

    RW = D^{-1} A (row-normalized by src degree); PE[:, t] = diagonal(RW^{t+1}).
    Sparse CSR implementation: molecular bond graphs are sparse/banded so RW^t
    stays sparse (nnz << N^2) -- cheaper than the dense N^3 power iteration.
    """
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, k).float()

    row = edge_index[0]
    # 1/deg(src), clamp(min=1) keeps isolated rows at 0 (no out-edges -> zero row)
    deg = scatter(torch.ones(row.size(0)), row, dim_size=num_nodes, reduce='sum').clamp(min=1)
    value = (1.0 / deg)[row]

    adj = to_torch_csr_tensor(edge_index, value, size=(num_nodes, num_nodes))

    def diag(out):
        return get_self_loop_attr(*to_edge_index(out), num_nodes=num_nodes)

    out = adj
    pe = [diag(out)]
    for _ in range(k - 1):
        out = out @ adj
        pe.append(diag(out))
    return torch.stack(pe, dim=-1).float()


class BAPredDataset(Dataset):
    """Protein-ligand binding affinity dataset producing torch_geometric Data graphs."""

    def __init__(self, protein_pdb, ligand_file, train=True, only_cluster_leads=True):
        super().__init__()
        self.lig_mols, self.err_tags, self.lig_names, _ = load_ligands(
            ligand_file, only_cluster_leads=only_cluster_leads
        )
        self.prot_atom_line, self.prot_atom_coord = self.get_protein_info(protein_pdb)

    def __getitem__(self, idx):
        name = self.lig_names[idx]
        if self.err_tags[idx] == 0:
            try:
                lmol = self.lig_mols[idx]
                pmol = self.get_pocket_with_ligand_in_protein(self.prot_atom_line, self.prot_atom_coord, lmol)
                gl = self.mol_to_graph(lmol)
                gp = self.mol_to_graph(pmol)
                gc = self.complex_to_graph(pmol, lmol)
                return gp, gl, gc, 0, idx, name
            except Exception:
                pass  # parsed but graph build failed -> emit dummy, flag as error

        gp = self.prot_dummy_graph(num_nodes=1000)
        gl = self.lig_dummy_graph(num_nodes=2)
        gc = self.comp_dummy_graph(num_nodes=1002)
        return gp, gl, gc, 1, idx, name

    def __len__(self):
        return len(self.lig_mols)

    def lig_dummy_graph(self, num_nodes):
        edge_index = torch.randint(0, num_nodes, (2, 10))
        g = Data(
            x=torch.zeros((num_nodes, 57)).float(),
            edge_index=edge_index,
            edge_attr=torch.zeros((10, 13)).float(),
            pos_enc=torch.zeros((num_nodes, 20)).float(),
            pos=torch.randn((num_nodes, 3)).float(),
        )
        g.num_nodes = num_nodes
        return g

    def prot_dummy_graph(self, num_nodes):
        edge_index = torch.randint(0, num_nodes, (2, 10))
        g = Data(
            x=torch.zeros((num_nodes, 57)).float(),
            edge_index=edge_index,
            edge_attr=torch.zeros((10, 13)).float(),
            pos_enc=torch.zeros((num_nodes, 20)).float(),
            pos=torch.randint(0, 100, (num_nodes, 3)).float(),
        )
        g.num_nodes = num_nodes
        return g

    def comp_dummy_graph(self, num_nodes):
        edge_index = torch.randint(0, num_nodes, (2, 10))
        g = Data(
            edge_index=edge_index,
            edge_attr=torch.zeros((10, 25)).float(),
            distance=torch.zeros((10, 1)).float(),
            pos=torch.randint(0, 100, (num_nodes, 3)).float(),
        )
        g.num_nodes = num_nodes
        return g

    def get_protein_info(self, prot_pdb):
        prot_atom_line = []
        prot_atom_coord = []
        for line in open(prot_pdb).readlines():
            if line[0:4] in ['ATOM', 'HETA'] and 'H' not in line[12:14] and 'HOH' not in line[17:20]:
                prot_atom_line.append(line)
                prot_atom_coord.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

        return prot_atom_line, prot_atom_coord

    def get_pocket_with_ligand_in_protein(self, prot_atom_line, prot_atom_coord, lig_mol):
        lig_atom_coord = torch.tensor(lig_mol.GetConformers()[0].GetPositions()).float()
        prot_atom_coord = torch.tensor(prot_atom_coord).float()

        pl_distance = torch.cdist(prot_atom_coord, lig_atom_coord)
        select_index = set(torch.where(pl_distance < 8)[0].tolist())

        select_residue = defaultdict(set)
        for idx, line in enumerate(prot_atom_line):
            if idx in select_index:
                select_residue[line[21]].add(int(line[22:26]))
        total_lines = """"""
        for idx, line in enumerate(prot_atom_line):
            if int(line[22:26]) in select_residue[line[21]]:
                total_lines += line

        mol = Chem.MolFromPDBBlock(total_lines, sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                pass

        return mol

    def mol_to_graph(self, mol):
        n = mol.GetNumAtoms()
        coord = get_mol_coordinate(mol)
        h = get_atom_feature(mol)
        edge_index, e = get_bond_feature(mol)

        pos_enc = random_walk_pe(edge_index, n, 20)

        g = Data(
            x=h,
            edge_index=edge_index,
            edge_attr=e,
            pos_enc=pos_enc,
            pos=coord,
        )
        g.num_nodes = n
        return g

    def complex_to_graph(self, pmol, lmol):
        pcoord = get_mol_coordinate(pmol)
        lcoord = get_mol_coordinate(lmol)
        ccoord = torch.cat([pcoord, lcoord])

        npa = pmol.GetNumAtoms()
        nla = lmol.GetNumAtoms()

        distance = torch.cdist(pcoord, lcoord)
        u, v = torch.where(distance < 5)  # u - src protein node, v - dst ligand node

        distance = distance[u, v].unsqueeze(-1)

        interact_feature = get_interact_feature(pmol, lmol, u, v)
        distance_feature = get_distance_feature(distance).squeeze(-1)

        e = torch.cat([interact_feature, distance_feature], dim=1)
        e = torch.cat([e, e])

        distance = torch.cat([distance, distance])

        u, v = torch.cat([u, v + npa]), torch.cat([v + npa, u])

        g = Data(
            edge_index=torch.stack([u, v], dim=0),
            edge_attr=e,
            distance=distance,
            pos=ccoord,
        )
        g.num_nodes = npa + nla
        return g


def collate_pyg(samples):
    """Batch a list of (gp, gl, gc, error, idx, name) into PyG Batch objects."""
    gps, gls, gcs, errors, idxs, names = zip(*samples)
    bgp = Batch.from_data_list(list(gps))
    bgl = Batch.from_data_list(list(gls))
    bgc = Batch.from_data_list(list(gcs))
    errors = torch.tensor(errors)
    idxs = torch.tensor(idxs)
    return bgp, bgl, bgc, errors, idxs, list(names)
