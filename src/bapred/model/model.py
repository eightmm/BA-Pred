import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool

from .GraphGPS import GraphGPS

"""PredictionPKD (PyG implementation).

Inputs gp, gl, gc are torch_geometric Batch objects with fields:
    .x         node features        (was DGL ndata['feats'])
    .edge_attr edge features         (was DGL edata['feats'])
    .edge_index [2, E] connectivity  (was DGL graph topology)
    .pos_enc   random-walk PE        (was DGL ndata['pos_enc'])
    .batch     node->graph assignment
gc has no .x / .pos_enc (built from stitched protein+ligand features).
"""


class PredictionPKD(nn.Module):
    def __init__(
        self,
        in_size,
        emb_size,
        intra_edge_size,
        inter_edge_size,
        pose_size,
        num_layers,
        dropout_ratio=0.15,
    ):
        super(PredictionPKD, self).__init__()
        self.protein_node_encoder = nn.Linear(in_size, emb_size)
        self.protein_edge_encoder = nn.Linear(intra_edge_size, emb_size)
        self.protein_pose_encoder = nn.Linear(pose_size, emb_size)

        self.ligand_node_encoder = nn.Linear(in_size, emb_size)
        self.ligand_edge_encoder = nn.Linear(intra_edge_size, emb_size)
        self.ligand_pose_encoder = nn.Linear(pose_size, emb_size)

        self.complex_edge_encoder = nn.Linear(inter_edge_size, emb_size)

        self.protein_norm = nn.LayerNorm(emb_size)
        self.ligand_norm = nn.LayerNorm(emb_size)

        blocks = [
            nn.ModuleList(
                [
                    GraphGPS(
                        emb_size,
                        4
                    )
                    for _ in range(num_layers)
                ]
            )
            for _ in range(3)
        ]

        self.protein_block = blocks[0]
        self.ligand_block = blocks[1]
        self.complex_block = blocks[2]

        self.mlp_binding_affinity = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(emb_size, 1),
        )

    def forward(self, gp, gl, gc):
        hp = self.protein_node_encoder(gp.x)
        ep = self.protein_edge_encoder(gp.edge_attr)
        pp = self.protein_pose_encoder(gp.pos_enc)

        hl = self.ligand_node_encoder(gl.x)
        el = self.ligand_edge_encoder(gl.edge_attr)
        pl = self.ligand_pose_encoder(gl.pos_enc)

        ec = self.complex_edge_encoder(gc.edge_attr)

        hp = self.protein_norm(hp)
        hl = self.ligand_norm(hl)

        gp_sizes = torch.bincount(gp.batch, minlength=gp.num_graphs).tolist()
        gl_sizes = torch.bincount(gl.batch, minlength=gl.num_graphs).tolist()

        # Precompute static gather indices (constant across layers) for the
        # protein+ligand -> complex stitching. Complex node order per sample is
        # [protein nodes of s, ligand nodes of s], matching complex_to_graph /
        # Batch.from_data_list. Replaces the per-layer python slice loops.
        Np = hp.size(0)
        device = hp.device
        to_complex = []   # index into cat([hp, hl]) -> complex node order
        hp_from_c = []    # index into hc -> recover hp order
        hl_from_c = []    # index into hc -> recover hl order
        gp_off = 0
        gl_off = 0
        cpos = 0
        for gp_size, gl_size in zip(gp_sizes, gl_sizes):
            to_complex.extend(range(gp_off, gp_off + gp_size))
            to_complex.extend(range(Np + gl_off, Np + gl_off + gl_size))
            hp_from_c.extend(range(cpos, cpos + gp_size))
            cpos += gp_size
            hl_from_c.extend(range(cpos, cpos + gl_size))
            cpos += gl_size
            gp_off += gp_size
            gl_off += gl_size
        to_complex = torch.tensor(to_complex, dtype=torch.long, device=device)
        hp_from_c = torch.tensor(hp_from_c, dtype=torch.long, device=device)
        hl_from_c = torch.tensor(hl_from_c, dtype=torch.long, device=device)

        for (protein_layer, ligand_layer, complex_layer) in zip(self.protein_block, self.ligand_block, self.complex_block):
            hp, pp, ep = protein_layer(gp.edge_index, hp, pp, ep)  # edge_index, h, p, e
            hl, pl, el = ligand_layer(gl.edge_index, hl, pl, el)

            hc = torch.cat([hp, hl], 0)[to_complex]
            pc = torch.cat([pp, pl], 0)[to_complex]

            hc, pc, ec = complex_layer(gc.edge_index, hc, pc, ec)

            hp = hc[hp_from_c]
            hl = hc[hl_from_c]

        h = global_add_pool(hl, gl.batch)

        binding_affinity = self.mlp_binding_affinity(h)

        return binding_affinity
