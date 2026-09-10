from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class IntraPropagation(nn.Module):
    """Input-anchored gated propagation over a static molecular graph."""

    def __init__(self, dim: int, dropout: float, layerscale_init: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)
        self.score = MLP(dim * 3, dim, 1, dropout)
        self.value = MLP(dim * 2, dim, dim, dropout)
        self.update = MLP(dim * 3, dim * 2, dim, dropout)
        self.gate = nn.Linear(dim * 3, dim)
        self.scale = nn.Parameter(torch.full((dim,), layerscale_init))

    def forward(self, h, h0, edge_index, edge_emb):
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        x = self.norm(h)
        e = self.edge_norm(edge_emb)
        pair = torch.cat([x[src], x[dst], e], dim=-1)
        score = torch.sigmoid(self.score(pair))
        denom = scatter(score, dst, dim=0, dim_size=h.size(0), reduce="sum").clamp_min(1e-6)
        alpha = score / denom[dst]
        msg = alpha * self.value(torch.cat([x[src], e], dim=-1))
        agg = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="sum")
        u = torch.cat([x, agg, h0], dim=-1)
        delta = self.update(u)
        gate = torch.sigmoid(self.gate(u))
        return h + self.scale * gate * delta


class RecurrentBindingBlock(nn.Module):
    """One shared cycle: interface refinement -> cross message -> intra propagation."""

    def __init__(self, dim: int, dropout: float, layerscale_init: float, use_endpoint_context: bool = True):
        super().__init__()
        self.use_endpoint_context = use_endpoint_context
        self.p_norm = nn.LayerNorm(dim)
        self.l_norm = nn.LayerNorm(dim)
        self.q_norm = nn.LayerNorm(dim)
        q_in_dim = dim * (6 if use_endpoint_context else 4)
        self.q_candidate = MLP(q_in_dim, dim * 2, dim, dropout)
        self.q_gate = nn.Linear(q_in_dim, dim)
        self.q_scale = nn.Parameter(torch.full((dim,), layerscale_init))
        self.pl_score = MLP(dim * 3, dim, 1, dropout)
        self.lp_score = MLP(dim * 3, dim, 1, dropout)
        self.pl_value = MLP(dim * 2, dim, dim, dropout)
        self.lp_value = MLP(dim * 2, dim, dim, dropout)
        self.p_cross_update = MLP(dim * 3, dim * 2, dim, dropout)
        self.l_cross_update = MLP(dim * 3, dim * 2, dim, dropout)
        self.p_cross_gate = nn.Linear(dim * 3, dim)
        self.l_cross_gate = nn.Linear(dim * 3, dim)
        self.p_cross_scale = nn.Parameter(torch.full((dim,), layerscale_init))
        self.l_cross_scale = nn.Parameter(torch.full((dim,), layerscale_init))
        self.p_intra = IntraPropagation(dim, dropout, layerscale_init)
        self.l_intra = IntraPropagation(dim, dropout, layerscale_init)

    @staticmethod
    def _normalized_sigmoid(score, dst, dim_size):
        s = torch.sigmoid(score)
        denom = scatter(s, dst, dim=0, dim_size=dim_size, reduce="sum").clamp_min(1e-6)
        return s / denom[dst]

    def forward(self, hp, hl, hp0, hl0, q, q0, cross_edge_index, p_edge_index, p_edge_emb, l_edge_index, l_edge_emb):
        pidx, lidx = cross_edge_index
        pn, ln, qn = self.p_norm(hp), self.l_norm(hl), self.q_norm(q)
        pieces = [pn[pidx], ln[lidx], qn, q0]
        if self.use_endpoint_context:
            pctx = scatter(qn, pidx, dim=0, dim_size=hp.size(0), reduce="mean")
            lctx = scatter(qn, lidx, dim=0, dim_size=hl.size(0), reduce="mean")
            pieces += [pctx[pidx], lctx[lidx]]
        qu = torch.cat(pieces, dim=-1)
        qcand = self.q_candidate(qu)
        qgate = torch.sigmoid(self.q_gate(qu))
        q = q + self.q_scale * qgate * (qcand - q)
        qn = self.q_norm(q)
        pair = torch.cat([pn[pidx], ln[lidx], qn], dim=-1)
        a_pl = self._normalized_sigmoid(self.pl_score(pair), lidx, hl.size(0))
        a_lp = self._normalized_sigmoid(self.lp_score(pair), pidx, hp.size(0))
        m_pl = a_pl * self.pl_value(torch.cat([pn[pidx], qn], dim=-1))
        m_lp = a_lp * self.lp_value(torch.cat([ln[lidx], qn], dim=-1))
        agg_l = scatter(m_pl, lidx, dim=0, dim_size=hl.size(0), reduce="sum")
        agg_p = scatter(m_lp, pidx, dim=0, dim_size=hp.size(0), reduce="sum")
        pu = torch.cat([pn, agg_p, hp0], dim=-1)
        lu = torch.cat([ln, agg_l, hl0], dim=-1)
        hp = hp + self.p_cross_scale * torch.sigmoid(self.p_cross_gate(pu)) * self.p_cross_update(pu)
        hl = hl + self.l_cross_scale * torch.sigmoid(self.l_cross_gate(lu)) * self.l_cross_update(lu)
        hp = self.p_intra(hp, hp0, p_edge_index, p_edge_emb)
        hl = self.l_intra(hl, hl0, l_edge_index, l_edge_emb)
        return hp, hl, q
