import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import gnn


class fMRINet(nn.Module):
    def __init__(self, ROInum, activation, hidden_dim=1, degree_normalize=False, gcn_mode="gcn", num_heads=1, num_gnn_layers=1, feat_dim=None):
        super(fMRINet, self).__init__()
        self.hidden_dim = hidden_dim
        self.ROInum = ROInum
        # feat_dim defaults to ROInum (FC-profile rows as features). Pass an explicit value for
        # multimodal where node features come from sMRI (in_dim = num sMRI features).
        self.feat_dim = feat_dim if feat_dim is not None else ROInum
        self.GNN = gnn.build_single_res_fmri(gcn_mode, nn.ReLU(), 0.3, self.feat_dim, hidden_dim, num_heads, degree_normalize, num_gnn_layers)
        self.paranum = self.GNN.output_dim

        self.bn1 = torch.nn.BatchNorm1d(self.paranum)
        self.fl1 = nn.Linear(self.paranum, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.fl2 = nn.Sequential(nn.Linear(64,1), nn.Sigmoid()) if activation == "sigmoid" else nn.Linear(64,1)


    def forward(self, g_matrix, node_features=None):
        fea = node_features if node_features is not None else g_matrix

        out = self.GNN(g_matrix, fea)  # [B, output_dim]

        out = self.bn1(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.fl2(out)

        return out
