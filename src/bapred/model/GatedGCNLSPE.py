import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import scatter

"""
    GatedGCNLSPE: GatedGCN with LSPE (PyG implementation)

    Edge convention: edge_index[0] = src (u), edge_index[1] = dst (v).
    Messages aggregate at dst (v), matching the original DGL update_all semantics.
"""


class GatedGCNLSPELayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, batch_norm, use_lapeig_loss=False, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_lapeig_loss = use_lapeig_loss

        if input_dim != output_dim:
            self.residual = False

        self.A1 = nn.Linear(input_dim * 2, output_dim, bias=True)
        self.A2 = nn.Linear(input_dim * 2, output_dim, bias=True)
        self.B1 = nn.Linear(input_dim, output_dim, bias=True)
        self.B2 = nn.Linear(input_dim, output_dim, bias=True)
        self.B3 = nn.Linear(input_dim, output_dim, bias=True)
        self.C1 = nn.Linear(input_dim, output_dim, bias=True)
        self.C2 = nn.Linear(input_dim, output_dim, bias=True)

        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(self, edge_index, h, p, e):
        # residual connection
        h_in = h
        p_in = p
        e_in = e

        src = edge_index[0]  # u
        dst = edge_index[1]  # v
        N = h.size(0)

        # node-wise projections
        A1_h = self.A1(torch.cat((h, p), -1))
        B1_h = self.B1(h)
        B2_h = self.B2(h)
        C1_p = self.C1(p)

        # edge-wise projection
        B3_e = self.B3(e)

        #--------------------------------------------------------------------------------------#
        # Calculation of h
        # fn.u_add_v('B1_h', 'B2_h'): src B1 + dst B2
        B1_B2_h = B1_h[src] + B2_h[dst]
        hat_eta = B1_B2_h + B3_e
        sigma_hat_eta = torch.sigmoid(hat_eta)

        # sum_j' sigma_hat_eta_ij' (aggregate at dst v)
        sum_sigma_hat_eta = scatter(sigma_hat_eta, dst, dim=0, dim_size=N, reduce='sum')
        # sigma_hat_eta_ij / sum_j' sigma_hat_eta_ij'
        eta_ij = sigma_hat_eta / (sum_sigma_hat_eta[dst] + 1e-6)

        # v_ij = A2(cat(h_j, p_j)) with j = src
        v_ij = self.A2(torch.cat((h[src], p[src]), -1))
        eta_mul_v = eta_ij * v_ij
        sum_eta_v = scatter(eta_mul_v, dst, dim=0, dim_size=N, reduce='sum')
        h = A1_h + sum_eta_v

        # Calculation of p
        C2_pj = self.C2(p[src])
        eta_mul_p = eta_ij * C2_pj
        sum_eta_p = scatter(eta_mul_p, dst, dim=0, dim_size=N, reduce='sum')
        p = C1_p + sum_eta_p

        #--------------------------------------------------------------------------------------#

        # passing towards output
        e = hat_eta

        # batch normalization
        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)
            # No BN for p

        # non-linear activation
        h = F.relu(h)
        e = F.relu(e)
        p = torch.tanh(p)

        # residual connection
        if self.residual:
            h = h_in + h
            p = p_in + p
            e = e_in + e

        # dropout
        h = F.dropout(h, self.dropout, training=self.training)
        p = F.dropout(p, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, p, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
