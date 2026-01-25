import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index_struct):
        """
        x:                (N, F) node features
        edge_index_struct:(2, E) COO adjacency for message passing
        """
        h = self.conv1(x, edge_index_struct)
        h = F.relu(h)
        h = self.conv2(h, edge_index_struct)
        return h  # (N, out_dim)


class EdgeMLP(nn.Module):
    def __init__(self, node_emb_dim):
        super().__init__()
        # pair_feat has 4 * d features (see forward), so in_features must be 4 * node_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(4 * node_emb_dim, 2 * node_emb_dim),
            nn.ReLU(),
            nn.Linear(2 * node_emb_dim, 1)   # 1 logit per pair
        )

    def forward(self, h, edge_index_lab):
        """
        h:              (N, d) node embeddings
        edge_index_lab: (2, M) node pairs to score
        """
        src, dst = edge_index_lab
        h_src = h[src]  # (M, d)
        h_dst = h[dst]  # (M, d)

        pair_feat = torch.cat([
            h_src * h_dst,
            torch.abs(h_src - h_dst),
            #torch.cat([h_src, h_dst], dim=-1)
            h_src, h_dst
        ], dim=-1)  # (M, 4d)

        logits = self.mlp(pair_feat)  # (M, 1)
        return logits

class LinkPredictionGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = GNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            #num_layers=num_layers,
            #dropout=dropout,
        )

        self.edge_head = EdgeMLP(out_dim)

    def forward(self, x, edge_index_struct, edge_index_lab):
        h = self.encoder(x, edge_index_struct)
        logits = self.edge_head(h, edge_index_lab)
        return logits