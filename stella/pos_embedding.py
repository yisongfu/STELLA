import torch
import numpy as np
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler


class PositionEmbedding(nn.Module):
    r"""
    Attaches spatial and temporal embedding to the data embedding

    - spatial embedding: projects the location of stations to the latent space with dimension d_model.
    - temporal embedding: embeds the temporal information of data to an embedding with dimension d_model.

    The output of this module is the sum of above three embeddings.
    """

    def __init__(self, d_model, num_nodes, root_path, if_rel=False):
        super(PositionEmbedding, self).__init__()

        self.randomly_init = if_rel
        node_pos = np.load(os.path.join(root_path, "pos_data.npy"), allow_pickle=True)
        scaler = StandardScaler()
        scaler.fit(node_pos)
        node_pos = scaler.transform(node_pos)
        self.node_pos = torch.from_numpy(node_pos).float()
        self.time_emb = TemporalEmbedding(d_model)
        self.node_emb = SpatialEmbedding(d_model, num_nodes, if_rel)

    def forward(self, x, x_mark):
        # x in [B, C, N, L]
        B, C, N, _ = x.shape

        # x_mark in [B, f], time_emb in  [B, C, N, D]
        # dimension changes as [B, f] -> [B, D] -> [C, N, B, D] -> [B, C, N, D]
        time_emb = self.time_emb(x_mark).repeat(C, N, 1, 1).permute(2, 0, 1, 3)
        # node_pos in [N, 3], node_emb in [B, C, N, D]
        # dimension changes as [N, 3] -> [N, D] -> [B, C, N, D]
        pos = self.node_pos.to(x.device)
        node_emb = self.node_emb(pos).repeat(B, C, 1, 1)
        emb = x + time_emb + node_emb
        return emb


class SpatialEmbedding(nn.Module):

    def __init__(self, node_dim, num_nodes, if_rel=False):
        super(SpatialEmbedding, self).__init__()
        self.if_rel = if_rel
        self.num_nodes = num_nodes

        if self.if_rel is False:
            self.embedding = nn.Sequential(
                nn.Linear(3, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim) # nn.Linear(node_dim / 2, node_dim) for alternative
            )

        else:
            self.embedding = nn.Embedding(num_nodes, node_dim)

    def forward(self, x):
        if self.if_rel is False:
            x = self.embedding(x)
        else:
            node_id = torch.Tensor(np.arange(self.num_nodes)).to(x.device)
            x = self.embedding(node_id)
        return x


class TemporalEmbedding(nn.Module):

    def __init__(self, time_dim):
        super(TemporalEmbedding, self).__init__()

        self.hour_size = 24
        self.day_size = 32
        self.month_size = 13

        self.hour_in_day_embed = nn.Embedding(self.hour_size, time_dim)
        self.day_in_month_embed = nn.Embedding(self.day_size, time_dim)
        self.month_in_year_embed = nn.Embedding(self.month_size, time_dim)

    def forward(self, x):
        _, f = x.shape
        hour_x = self.hour_in_day_embed((x[..., 2] * self.hour_size).long()) if f > 2 else 0.
        day_x = self.day_in_month_embed((x[..., 1] * self.day_size).long()) if f > 1 else 0.
        month_x = self.month_in_year_embed((x[..., 0] * self.month_size).long())

        return hour_x + day_x + month_x
