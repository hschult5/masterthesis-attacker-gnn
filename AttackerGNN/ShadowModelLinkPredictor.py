import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

'''
class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        dropout=0.3
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # input layer
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # output projection layer
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.norms.append(nn.LayerNorm(out_dim))

    def forward(self, x, edge_index_struct):
        """
        x: (N, F)
        edge_index_struct: (2, E)
        """

        layer_outputs = []
        h = x

        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index_struct)
            h_new = self.norms[i](h_new)

            if i < self.num_layers - 1:
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # residual connection (when dimensions allow)
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new

            layer_outputs.append(h)

        # Jumping Knowledge: concatenate all layer outputs
        h_jk = torch.cat(layer_outputs, dim=-1)  # (N, sum dims)

        return h_jk

class EdgeMLP(nn.Module):
    def __init__(self, node_emb_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4 * node_emb_dim, 2 * node_emb_dim),
            nn.ReLU(),
            nn.Linear(2 * node_emb_dim, 1)
        )

    def forward(self, h, edge_index_lab):
        src, dst = edge_index_lab
        h_src = h[src]
        h_dst = h[dst]

        pair_feat = torch.cat([
            h_src * h_dst,
            torch.abs(h_src - h_dst),
            torch.cat([h_src, h_dst], dim=-1)
        ], dim=-1)

        logits = self.mlp(pair_feat)
        return logits


'''
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
            torch.cat([h_src, h_dst], dim=-1)
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