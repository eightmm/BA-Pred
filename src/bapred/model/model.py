import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.pytorch.glob import SumPooling

from .GraphGPS import GraphGPS

class PredictionPKD(nn.Module):
    def __init__(self, in_size, emb_size, intra_edge_size, inter_edge_size, pose_size, num_layers, dropout_ratio=0.15):
        super(PredictionPKD, self).__init__()
        self.protein_node_encoder = nn.Linear( in_size, emb_size )
        self.protein_edge_encoder = nn.Linear( intra_edge_size,  emb_size )
        self.protein_pose_encoder = nn.Linear( pose_size,  emb_size )

        self.ligand_node_encoder = nn.Linear( in_size, emb_size )
        self.ligand_edge_encoder = nn.Linear( intra_edge_size,  emb_size )
        self.ligand_pose_encoder = nn.Linear( pose_size,  emb_size )

        self.complex_edge_encoder = nn.Linear( inter_edge_size, emb_size )

        self.protein_norm = nn.LayerNorm( emb_size )
        self.ligand_norm  = nn.LayerNorm( emb_size )

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
        self.ligand_block  = blocks[1]
        self.complex_block = blocks[2]

        self.mlp_binding_affinity = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(emb_size, 1),
        )

        self.sum_pooling = SumPooling()

    def forward(self, gp, gl, gc):
        hp = self.protein_node_encoder( gp.ndata['feats'] )
        ep = self.protein_edge_encoder( gp.edata['feats'] )
        pp = self.protein_pose_encoder( gp.ndata['pos_enc'] )

        hl = self.ligand_node_encoder( gl.ndata['feats'] )
        el = self.ligand_edge_encoder( gl.edata['feats'] )
        pl = self.ligand_pose_encoder( gl.ndata['pos_enc'] )

        ec = self.complex_edge_encoder( gc.edata['feats'] )

        hp = self.protein_norm( hp )
        hl = self.ligand_norm( hl )

        hp_raw = hp
        hl_raw = hl

        gp_batch_sizes = gp.batch_num_nodes()
        gl_batch_sizes = gl.batch_num_nodes()

        gp_start_indices = [0] + torch.cumsum(gp_batch_sizes[:-1], dim=0).tolist()
        gl_start_indices = [0] + torch.cumsum(gl_batch_sizes[:-1], dim=0).tolist()

        for (protein_layer, ligand_layer, complex_layer) in zip(self.protein_block, self.ligand_block, self.complex_block):
            hp, pp, ep = protein_layer( gp, hp, pp, ep ) #  g, h, p, e,
            hl, pl, el = ligand_layer( gl, hl, pl, el )

            hc = []
            pc = []
            for gp_start, gp_size, gl_start, gl_size in zip(gp_start_indices, gp_batch_sizes, gl_start_indices, gl_batch_sizes):
                gp_slice = hp[gp_start:gp_start + gp_size]
                gl_slice = hl[gl_start:gl_start + gl_size]
                pp_slice = pp[gp_start:gp_start + gp_size]
                pl_slice = pl[gl_start:gl_start + gl_size]
                hc.append( torch.cat( [gp_slice, gl_slice] ) )
                pc.append( torch.cat( [pp_slice, pl_slice] ) )

            hc = torch.cat( hc )
            pc = torch.cat( pc )

            hc, pc, ec = complex_layer( gc, hc, pc, ec )

            hp_separated = []
            hl_separated = []
            start = 0
            for gp_size, gl_size in zip(gp_batch_sizes, gl_batch_sizes):
                hp_separated.append(hc[start: start + gp_size])
                start += gp_size
                hl_separated.append(hc[start: start + gl_size])
                start += gl_size

            hp = torch.cat(hp_separated)
            hl = torch.cat(hl_separated)

        h = self.sum_pooling(gl, hl)

        binding_affinity = self.mlp_binding_affinity( h )

        return binding_affinity
