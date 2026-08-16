import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index_struct, edge_weight=None):
        """
        x:                 (N, F) Node features
        edge_index_struct: (2, E) Adjacency for message passing
        """
        h = self.conv1(x, edge_index_struct, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index_struct, edge_weight)

        return h


class EdgeScoringHead(nn.Module):
    """
    Scores node pairs encodings
    """

    def __init__(self, node_emb_dim, pair_hidden_dim=None):
        super().__init__()
        d = node_emb_dim
        pair_in_dim = 4 * d
        if pair_hidden_dim is None:
            pair_hidden_dim = 2 * d
        self.pair_trunk = nn.Sequential(
            nn.Linear(pair_in_dim, pair_hidden_dim),
            nn.ReLU(),
        )
        self.edge_out = nn.Linear(pair_hidden_dim, 1)

    def forward(self, h, edge_index_lab):
        """
        h:              (N, d) node embeddings
        edge_index_lab: (2, M) candidate node pairs

        Returns
        -------
        edge_logits: (M, 1)
        """
        src, dst = edge_index_lab

        h_src = h[src]  # (M, d)
        h_dst = h[dst]  # (M, d)

        pair_feat = torch.cat(
            [
                h_src * h_dst,
                torch.abs(h_src - h_dst),
                torch.minimum(h_src, h_dst),
                torch.maximum(h_src, h_dst),
            ],
            dim=-1,
        )  # (M, 4d)

        z = self.pair_trunk(pair_feat)
        edge_logits = self.edge_out(z)

        return edge_logits


class LinkPredictionGNN(nn.Module):
    """
    Encode the graph and score candidate node pairs.

    Usage:
        logits = model(x, edge_index_struct, edge_index_lab)
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = GNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

        self.edge_head = EdgeScoringHead(out_dim)

    def forward(self, x, edge_index_struct, edge_index_lab, edge_weight=None):
        h = self.encoder(x, edge_index_struct, edge_weight)
        return self.edge_head(h, edge_index_lab)