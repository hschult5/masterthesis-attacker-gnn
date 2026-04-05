import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index_struct):
        """
        x:                (N, F) node features
        edge_index_struct:(2, E) COO adjacency for message passing
        """
        h = self.conv1(x, edge_index_struct)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index_struct)
        return h  # (N, out_dim)


class EdgeMultiTaskHead(nn.Module):
    """
    Main output:
        - edge logit for the candidate pair (drop-in replacement)

    Auxiliary outputs:
        - endpoint flip logit for src
        - endpoint flip logit for dst

    By default, forward(...) returns only edge logits, so existing code keeps working.
    """
    def __init__(self, node_emb_dim, pair_hidden_dim=None, endpoint_hidden_dim=None):
        super().__init__()

        d = node_emb_dim
        pair_in_dim = 4 * d

        if pair_hidden_dim is None:
            pair_hidden_dim = 2 * d
        if endpoint_hidden_dim is None:
            endpoint_hidden_dim = d

        # Shared pair feature trunk
        self.pair_trunk = nn.Sequential(
            nn.Linear(pair_in_dim, pair_hidden_dim),
            nn.ReLU(),
        )

        # Main edge score head (this is what PRBCD can keep using)
        self.edge_out = nn.Linear(pair_hidden_dim, 1)

        # Shared endpoint head:
        # input = [shared_pair_context, endpoint_embedding]
        self.endpoint_head = nn.Sequential(
            nn.Linear(pair_hidden_dim + d, endpoint_hidden_dim),
            nn.ReLU(),
            nn.Linear(endpoint_hidden_dim, 1),
        )

    def forward(self, h, edge_index_lab, return_aux=False):
        """
        h:              (N, d) node embeddings
        edge_index_lab: (2, M) node pairs to score
        return_aux:     if True, also returns endpoint logits

        Returns
        -------
        if return_aux == False:
            edge_logits: (M, 1)

        if return_aux == True:
            {
                "edge_logits": (M, 1),
                "src_flip_logits": (M, 1),
                "dst_flip_logits": (M, 1),
                "pair_prob_from_endpoints": (M, 1)
            }
        """
        src, dst = edge_index_lab
        h_src = h[src]  # (M, d)
        h_dst = h[dst]  # (M, d)

        pair_feat = torch.cat([
            h_src * h_dst,
            torch.abs(h_src - h_dst),
            h_src,
            h_dst
        ], dim=-1)  # (M, 4d)

        z = self.pair_trunk(pair_feat)          # (M, pair_hidden_dim)
        edge_logits = self.edge_out(z)          # (M, 1)

        if not return_aux:
            return edge_logits

        src_feat = torch.cat([z, h_src], dim=-1)
        dst_feat = torch.cat([z, h_dst], dim=-1)

        src_flip_logits = self.endpoint_head(src_feat)  # (M, 1)
        dst_flip_logits = self.endpoint_head(dst_feat)  # (M, 1)

        # Optional derived pair probability:
        # P(pair harmful) = 1 - (1 - P(src_flip)) * (1 - P(dst_flip))
        src_prob = torch.sigmoid(src_flip_logits)
        dst_prob = torch.sigmoid(dst_flip_logits)
        pair_prob_from_endpoints = 1.0 - (1.0 - src_prob) * (1.0 - dst_prob)

        return {
            "edge_logits": edge_logits,
            "src_flip_logits": src_flip_logits,
            "dst_flip_logits": dst_flip_logits,
            "pair_prob_from_endpoints": pair_prob_from_endpoints,
        }


class LinkPredictionGNN(nn.Module):
    """
    Drop-in replacement:
        logits = model(x, edge_index_struct, edge_index_lab)

    Optional multitask mode:
        out = model(x, edge_index_struct, edge_index_lab, return_aux=True)
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

        self.edge_head = EdgeMultiTaskHead(out_dim)

    def forward(self, x, edge_index_struct, edge_index_lab, return_aux=False):
        h = self.encoder(x, edge_index_struct)

        if not return_aux:
            logits = self.edge_head(h, edge_index_lab, return_aux=False)
            return logits

        out = self.edge_head(h, edge_index_lab, return_aux=True)
        out["node_embeddings"] = h
        return out

'''import torch
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
    ):
        super().__init__()

        self.encoder = GNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )

        self.edge_head = EdgeMLP(out_dim)

    def forward(self, x, edge_index_struct, edge_index_lab):
        h = self.encoder(x, edge_index_struct)
        logits = self.edge_head(h, edge_index_lab)
        return logits'''