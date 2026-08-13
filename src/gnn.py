import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

dir_path = r"C:\Users\Faruk\Code\rbc-brain-age-ood\src"

class GATEdgeLayer(nn.Module):
    """Multi-head GAT with FC edge-weight term in attention.

    e_ij = LeakyReLU( a_src·z_i + a_dst·z_j + edge_w * fc_ij )
    alpha_ij = softmax_j(e_ij)
    h_i' = ELU( sum_j alpha_ij * z_j )   [concatenated across heads]

    a_src·z_i + a_dst·z_j is the standard decomposition of a^T[z_i||z_j]
    (avoids materialising the [N,N,H,2D] concat tensor — same result).
    edge_w is a learned per-head scalar on the FC value.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.3):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads
        self.W      = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src  = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst  = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.edge_w = nn.Parameter(torch.ones(num_heads))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.leaky = nn.LeakyReLU(0.2)
        self.act   = nn.ELU()

    def forward(self, g, h):
        # g: [B, N, N], h: [B, N, feat_dim]
        B, N, _ = h.shape
        z = self.W(h).view(B, N, self.num_heads, self.head_dim)  # [B, N, H, D]
        e_src = (z * self.a_src).sum(-1)                          # [B, N, H]
        e_dst = (z * self.a_dst).sum(-1)                          # [B, N, H]
        e = self.leaky(
            e_src.unsqueeze(2) + e_dst.unsqueeze(1)               # [B, N, N, H]
            + g.unsqueeze(-1) * self.edge_w                        # [B, N, N, H]
        )
        e = e.masked_fill((g == 0).unsqueeze(-1), -1e9)
        alpha = self.drop(F.softmax(e, dim=2))                    # [B, N, N, H]
        out = torch.einsum('bijh,bjhd->bihd', alpha, z)           # [B, N, H, D]
        return self.act(out.reshape(B, N, -1))


def build_single_res_fmri(mode, act, drop_p, ROInum, hidden_dim, num_heads=1, degree_normalize=False, num_gnn_layers=1):
    if mode == "gcn":
        return VanilleGCN(ROInum, hidden_dim, act, drop_p, degree_normalize, num_gnn_layers)
    if mode == "gat":
        return VanillaGAT(ROInum, hidden_dim, drop_p, num_heads, num_gnn_layers)
    raise ValueError(f"Unknown gcn_mode for single-resolution fMRI: {mode!r}")

class VanillaGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_p, num_heads, num_gat_layers=1):
        super().__init__()
        self.output_dim = hidden_dim * num_heads   # total features per node, concatenated heads

        self.net_gat_layers = nn.ModuleList([GATEdgeLayer(in_dim if i == 0 else self.output_dim, self.output_dim, num_heads, drop_p) for i in range(num_gat_layers)])
        
    @staticmethod
    def forward_gat_layers(g, h, net_gat_layers):
        for gat_layer in net_gat_layers:
            h = gat_layer(g, h)
        return h

    def forward(self, g_matrix, h):
        # g_matrix: [B, N, N], h: [B, N, feat_dim]
        h = self.forward_gat_layers(g_matrix, h, self.net_gat_layers)
        return h.mean(dim=-2)  # [B, output_dim]

class VanilleGCN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p=0.3, degree_normalize=False, num_layers=1):
        super(VanilleGCN, self).__init__()
        self.proj = nn.ModuleList([nn.Linear(in_dim if i == 0 else out_dim, out_dim) for i in range(num_layers)])
        self.num_layers = num_layers
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.degree_normalize = degree_normalize
        self.output_dim = out_dim

    def forward(self, g, h):
        # g: [B, N, N], h: [B, N, feat_dim]
        if self.degree_normalize:
            deg = g.sum(dim=-1).clamp(min=1e-8).pow(-0.5)  # [B, N]
            g = deg.unsqueeze(-1) * g * deg.unsqueeze(-2)   # [B, N, N]
        for i in range(self.num_layers):
            h = self.drop(h)
            h = torch.bmm(g, h)   # [B, N, feat_dim]
            h = self.proj[i](h)   # [B, N, out_dim]
            h = self.act(h)
        return h.mean(dim=-2)     # [B, out_dim]
    

