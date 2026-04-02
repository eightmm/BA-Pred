import os
import torch, dgl
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

from dgl.data import DGLDataset

from meeko import PDBQTMolecule, RDKitMolCreate


from bapred.data.atom_feature import *



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
    supplier = Chem.SDMolSupplier(file_path)
    return _process_supplier(supplier, file_path)

def _process_mol2(file_path):
    """Helper function to process .mol2 files"""
    with open(file_path, 'r') as f:
        mol2_data = f.read()
    mol2_blocks = mol2_data.split('@<TRIPOS>MOLECULE')
    supplier = (
        Chem.MolFromMol2Block('@<TRIPOS>MOLECULE' + block)
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
            ligand_names.append(f"{base_name}_{idx}")
        else:
            ligands.append(None)
            err_tag.append(1)
            ligand_names.append(f"{base_name}_{idx}")

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
        lig_mols, err_tags, lig_names = [], [], []
        for line in lines:
            assert os.path.isfile(line), f"File not found: {line}"
            file_ligands, file_err_tag, file_ligand_names, _ = process_ligand_file(line, only_cluster_leads=only_cluster_leads)
            lig_mols.extend(file_ligands)
            err_tags.extend(file_err_tag)
            lig_names.extend(file_ligand_names)
        return lig_mols, err_tags, lig_names

    elif file_extension in ['.sdf', '.mol2', '.dlg', '.pdbqt']:
        return process_ligand_file(file_path, only_cluster_leads=only_cluster_leads)
    else:
        raise ValueError("Unsupported file type. Use '.txt', '.sdf', '.mol2', '.dlg', or '.pdbqt'.")


# def process_ligand_file(file_path):
#     extension = os.path.splitext(file_path)[-1].lower()

#     if extension == '.sdf':
#         supplier = enumerate(Chem.SDMolSupplier(file_path))
#     elif extension == '.mol2':
#         with open(file_path, 'r') as f:
#             mol2_data = f.read()
#         mol2_blocks = mol2_data.split('@<TRIPOS>MOLECULE')
#         supplier = enumerate(Chem.MolFromMol2Block('@<TRIPOS>MOLECULE' + block) for block in mol2_blocks[1:])
#     else:
#         raise ValueError(f"Unsupported file type: {extension}")

#     ligands = []
#     err_tag = []
#     ligand_names = []
#     base_name = os.path.splitext(os.path.basename(file_path))[0]

#     for idx, mol in supplier:
#         if mol is not None:
#             ligands.append(mol)
#             err_tag.append(0)
#             ligand_name = mol.GetProp('_Name')
#             if ligand_name == '':
#                 ligand_name = f"{base_name}_{idx}"
#             ligand_names.append(ligand_name)
#         else:
#             ligands.append(None)
#             err_tag.append(1)
#             ligand_names.append(f"{base_name}_{idx}")

#     return ligands, err_tag, ligand_names

# def load_ligands(file_path):
#     lig_mols = []
#     err_tags = []
#     lig_names = []

#     def process_single_file(line):
#         assert os.path.isfile(line), f"File not found: {line}"
#         return process_ligand_file(line)

#     file_extension = os.path.splitext(file_path)[-1].lower()

#     if file_extension == '.txt':
#         with open(file_path, 'r') as f:
#             lines = [line.strip() for line in f if line.strip()]

#         for line in lines:
#             file_ligands, file_err_tag, file_ligand_names = process_single_file(line)
#             lig_mols.extend(file_ligands)
#             err_tags.extend(file_err_tag)
#             lig_names.extend(file_ligand_names)

#     elif file_extension in ['.sdf', '.mol2']:
#         lig_mols, err_tags, lig_names = process_single_file(file_path)

#     else:
#         raise ValueError("Unsupported file type. Use '.txt', '.sdf', or '.mol2'.")

#     return lig_mols, err_tags, lig_names


class BAPredDataset(DGLDataset):
    def __init__(self, protein_pdb, ligand_file, train=True, only_cluster_leads=True):
        super(BAPredDataset, self).__init__(name='Protein Ligand Binding Affinity prediction')

        self.lig_mols, self.err_tags, self.lig_names, _ = load_ligands(ligand_file, only_cluster_leads=only_cluster_leads)

        self.prot_atom_line, self.prot_atom_coord = self.get_protein_info( protein_pdb )

    def __getitem__(self, idx):
        name = self.lig_names[idx]
        if self.err_tags[idx] == 0:
            lmol = self.lig_mols[idx]
            pmol = self.get_pocket_with_ligand_in_protein( self.prot_atom_line, self.prot_atom_coord, lmol )
            gl = self.mol_to_graph( lmol )
            gp = self.mol_to_graph( pmol )
            gc = self.complex_to_graph( pmol, lmol )
            error = 0
        else:
            gp = self.prot_dummy_graph( num_nodes=1000)
            gl = self.lig_dummy_graph( num_nodes=2 )
            gc = self.comp_dummy_graph( num_nodes=1002 )
            error = 1

        return gp, gl, gc, error, idx, name

    def __len__(self):
        return len(self.lig_mols)

    def lig_dummy_graph(self, num_nodes):
        src = torch.randint(0, num_nodes, (10,))
        dst = torch.randint(0, num_nodes, (10,))
        gl = dgl.graph( (src, dst), num_nodes=num_nodes)
        gl.ndata['feats'] = torch.zeros((num_nodes, 57)).float()
        gl.ndata['pos_enc'] = torch.zeros((num_nodes, 20)).float()
        gl.ndata['coord'] = torch.randn((num_nodes, 3)).float()
        gl.edata['feats'] = torch.zeros((10, 13)).float()
        return gl

    def prot_dummy_graph(self, num_nodes):
        src = torch.randint(0, num_nodes, (10,))
        dst = torch.randint(0, num_nodes, (10,))
        gp = dgl.graph( (src, dst), num_nodes=num_nodes)
        gp.ndata['feats'] = torch.zeros((num_nodes, 57)).float()
        gp.ndata['pos_enc'] = torch.zeros((num_nodes, 20)).float()
        gp.ndata['coord'] = torch.randint(0, 100, (num_nodes, 3)).float()
        gp.edata['feats'] = torch.zeros((10, 13)).float()
        return gp

    def comp_dummy_graph( self, num_nodes):
        src = torch.randint(0, num_nodes, (10,))
        dst = torch.randint(0, num_nodes, (10,))
        gc = dgl.graph( (src, dst), num_nodes=num_nodes)
        gc.ndata['coord'] = torch.randint(0, 100, (num_nodes, 3)).float()
        gc.edata['feats'] = torch.zeros((10, 25)).float()
        gc.edata['distance'] = torch.zeros((10, 1)).float()
        return gc


    def get_protein_info( self, prot_pdb ):
        prot_atom_line = []
        prot_atom_coord = []
        for line in open(prot_pdb).readlines():
            if line[0:4] in ['ATOM', 'HETA'] and 'H' not in line[12:14] and 'HOH' not in line[17:20]:
                prot_atom_line.append( line )
                prot_atom_coord.append( [ float(line[30:38]), float(line[38:46]), float(line[46:54]) ])

        return prot_atom_line, prot_atom_coord

    def get_pocket_with_ligand_in_protein(self, prot_atom_line, prot_atom_coord, lig_mol ):
        lig_atom_coord = torch.tensor( lig_mol.GetConformers()[0].GetPositions() ).float()
        prot_atom_coord = torch.tensor( prot_atom_coord ).float()

        pl_distance = torch.cdist( prot_atom_coord, lig_atom_coord )
        select_index = torch.where( pl_distance < 8 )[0]
        select_atom = [ line for idx, line in enumerate( prot_atom_line ) if idx in select_index ]

        select_residue = defaultdict(set)
        for idx, line in enumerate(prot_atom_line):
            if idx in select_index:
                select_residue[line[21]].add( int(line[22:26]) )
        total_lines = """"""
        for idx, line in enumerate(prot_atom_line):
            if int( line[22:26] ) in select_residue[ line[21] ]:
                total_lines += line
        
        mol = Chem.MolFromPDBBlock( total_lines )

        return mol

    def mol_to_graph( self, mol ):
        n     = mol.GetNumAtoms()
        coord = get_mol_coordinate(mol)
        h     = get_atom_feature(mol)
        adj   = get_bond_feature(mol).to_sparse(sparse_dim=2)

        u = adj.indices()[0]
        v = adj.indices()[1]
        e = adj.values()

        g = dgl.DGLGraph()
        g.add_nodes(n)
        g.add_edges(u, v)

        g.ndata['feats'] = h
        g.ndata['coord'] = coord
        g.edata['feats'] = e

        g.ndata['pos_enc'] = dgl.random_walk_pe(g, 20)

        return g

    def complex_to_graph( self, pmol, lmol):
        pcoord = get_mol_coordinate(pmol)
        lcoord = get_mol_coordinate(lmol)
        ccoord = torch.cat( [pcoord, lcoord] )

        npa = pmol.GetNumAtoms()
        nla = lmol.GetNumAtoms()

        distance = torch.cdist(pcoord, lcoord)
        u, v = torch.where( distance < 5 ) ### u - src protein node, v - dst ligand node

        distance = distance[ u, v ].unsqueeze(-1)

        interact_feature = get_interact_feature( pmol, lmol, u, v  )
        distance_feature = get_distance_feature(distance).squeeze(-1)

        e = torch.cat( [interact_feature, distance_feature], dim=1)
        e = torch.cat( [e, e] )

        distance = torch.cat( [ distance, distance] )

        u, v = torch.cat( [u, v+npa] ), torch.cat( [v+npa, u] )

        g = dgl.DGLGraph()
        g.add_nodes( npa + nla )
        g.add_edges( u, v )

        g.ndata['coord'] = ccoord
        g.edata['feats'] = e
        g.edata['distance'] = distance

        return g
