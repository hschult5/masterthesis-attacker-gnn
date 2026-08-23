
import random
from typing import Any
import copy

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import average_precision_score
from torch import Tensor
from torch_sparse import SparseTensor

from AttackerGNN.ShadowModelLinkPredictor import LinkPredictionGNN
from rgnn_at_scale.attacks import PRBCD
from sklearn.model_selection import train_test_split


def _edge_set(edge_index, *, undirected: bool = True) -> set[tuple[int, int]]:
    edges = edge_index.detach().cpu().numpy().T.tolist()
    out: set[tuple[int, int]] = set()
    for u, v in edges:
        if u == v:
            continue
        edge = tuple(sorted((int(u), int(v)))) if undirected else (int(u), int(v))
        out.add(edge)
    return out

def _dense_adj(edge_index, n_nodes: int, *, device=None):
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=device)
    if edge_index.numel():
        adj[edge_index[0].to(device), edge_index[1].to(device)] = 1.0
    adj.fill_diagonal_(0.0)
    return torch.maximum(adj, adj.t())

# Computes Accuracy on model
def accuracy(model, x, labels, idx, edge_index=None):
    model.eval()
    with torch.no_grad():
        logits = model(data=x, adj=edge_index)
        predictions = logits[idx].argmax(dim=1)
        return (predictions == labels[idx]).float().mean().item()


def train_selector(
    attr: torch.Tensor,
    edge_index_struct: torch.Tensor,
    edge_index_lab: torch.Tensor,
    y_label: torch.Tensor,
    label_mode: str,
    device: str = "cpu",
    num_epochs: int = 200,
    hidden_dim: int = 64,
    out_dim: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.1, # Only applies to endpoint binary labels
    log_every: int = 20,
    log_grad_norm: bool = False,
    # ---- early stopping / best checkpoint ----
    min_epochs_before_early_stop: int = 40,
    early_stop: bool = True,
    patience: int = 15,
    early_stop_min_delta: float = 1e-4,
    # ---- train / validation split ----
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    """Train the selector GNN on hard or soft edge targets in [0, 1].

    The training function deliberately does only what is needed for fitting:
    train/validation splitting, BCE training, validation-loss early stopping,
    and best-checkpoint restoration. Exhaustive selector statistics belong in
    the later test-set evaluation.
    """

    # Prepare inputs
    attr = attr.to(device)
    edge_index_struct = edge_index_struct.long().to(device)
    edge_index_lab = edge_index_lab.long().to(device)
    y_label = y_label.float().view(-1).to(device)

    if edge_index_struct.ndim != 2 or edge_index_struct.size(0) != 2:
        raise ValueError("edge_index_struct must have shape (2, E).")
    if edge_index_lab.ndim != 2 or edge_index_lab.size(0) != 2:
        raise ValueError("edge_index_lab must have shape (2, M).")

    M = edge_index_lab.size(1)

    # Define training state variables
    train_losses: list[float] = [] # Train loss on unsmoothed labels
    val_losses: list[float] = [] # Early stopping metric for subset_accuracy_drop labeling mode
    val_aps: list[float] = [] # Early stopping metric for endpoint labeling mode

    if label_mode == "endpoint":
        best_metric_score = float("-inf")
        early_stop_metric = "val_ap"
    else:
        best_metric_score = float("inf")
        early_stop_metric = "val_loss"

    best_epoch = None
    best_state = None
    patience_left = patience


    # Train/Validation/Test split
    train_ratio, val_ratio, test_ratio = split_ratios

    indices = np.arange(M)

    if label_mode == "endpoint":
        stratify_labels = (y_label > 0.5).numpy().astype(int)

    else:
        # Sort continuous labels and divide into 5 equally sized rank bins
        order = np.argsort(y_label.numpy())
        stratify_labels = np.empty(M, dtype=int)

        for bin_id, bin_indices in enumerate(np.array_split(order, 5)):
            stratify_labels[bin_indices] = bin_id


    # First split for trainíng set
    temp_ratio = val_ratio + test_ratio

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        stratify=stratify_labels,
        random_state=0,
    )

    # Then split the remaining into validation and test sets
    relative_test_ratio = test_ratio / temp_ratio

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        stratify=stratify_labels[temp_idx],
        random_state=0,
    )

    # Convert to tensors
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

    train_y_labels = y_label[train_idx]
    edge_index_lab_train = edge_index_lab[:, train_idx]

    val_y_labels = y_label[val_idx]
    edge_index_lab_val = edge_index_lab[:, val_idx]

    # Add label-smoothing only for endpoint binary labels.
    if label_mode == "endpoint":
        train_y_labels_smooth = (train_y_labels * (1.0 - label_smoothing) + 0.5 * label_smoothing)
    else:
        train_y_labels_smooth = train_y_labels

    # Initialize label weight only for unbalanced endpoint labels.
    if label_mode == "endpoint":
        label_one_count = float(train_y_labels.sum().item())
        label_zero_count = float((1.0 - train_y_labels).sum().item())
        label_weight = label_zero_count / label_one_count
    else:
        label_weight = 1.0

    # Define Model
    model = LinkPredictionGNN(
        in_dim=attr.size(1),
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    ).to(device)

    # Define Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Define loss
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(label_weight, dtype=torch.float32, device=device)
    )

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits_train = model(attr, edge_index_struct, edge_index_lab_train).view(-1)

        optimization_loss = loss_fn(logits_train, train_y_labels_smooth)
        optimization_loss.backward()

        # Compute gradient norm for investigating gradient explosion
        grad_norm = None
        if log_grad_norm:
            total_norm_sq = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    norm = parameter.grad.detach().norm(2).item()
                    total_norm_sq += norm * norm
            grad_norm = total_norm_sq ** 0.5

        optimizer.step()
        model.eval()

        # Compute validation loss and the training loss on unsmoothed labels for comparison
        with torch.no_grad():
            logits_train_eval = model(attr, edge_index_struct, edge_index_lab_train).view(-1)
            train_loss = loss_fn(logits_train_eval, train_y_labels)

            logits_val = model(attr, edge_index_struct, edge_index_lab_val).view(-1)
            val_loss = loss_fn(logits_val, val_y_labels)

            # For endpoint label mode compute Average Precision
            if label_mode == "endpoint":
                val_probs = torch.sigmoid(logits_val)
                val_ap = average_precision_score(val_y_labels.detach().numpy(), val_probs.detach().numpy())
            else:
                val_ap = None

        train_loss_value = float(train_loss.item())
        val_loss_value = float(val_loss.item())

        train_losses.append(train_loss_value)
        val_losses.append(val_loss_value)

        if val_ap is not None:
            val_aps.append(val_ap)

        # Record early stopping metric and patience if a better score is not found
        if label_mode == "endpoint":
            current_score = val_ap
            is_best = current_score > best_metric_score + early_stop_min_delta
        else:
            current_score = val_loss_value
            is_best = current_score < best_metric_score - early_stop_min_delta

        # Record best model
        if is_best:
            best_metric_score = current_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience

        # Reduce patience if metric_score moves away from target
        elif early_stop and epoch >= min_epochs_before_early_stop:
            patience_left -= 1

        # Logging message
        if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
            message = (
                f"[LP-GNN] Epoch {epoch:03d}/{num_epochs} | "
                f"train_loss={train_loss_value:.4f} | "
                f"val_loss={val_loss_value:.4f} | "
            )
            if early_stop: message += f" | early_stopping_metric={early_stop_metric}"
            if early_stop: message += f" | current_best={best_metric_score:.4f}"
            if early_stop: message += f" | patience_left={patience_left}"
            if grad_norm is not None: message += f" | grad_norm={grad_norm:.3e}"
            if is_best: message += " | new_best"
            print(message)

        stopped_at = epoch

        if early_stop and epoch >= min_epochs_before_early_stop and patience_left <= 0:
            print(
                f"Selector Training stopping at epoch {epoch}. "
                f"Best early_stopping_metric={best_metric_score:.4f} at epoch {best_epoch}."
            )
            break

    # Restore the best model
    model.load_state_dict(best_state)

    # Record training history
    model.training_history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_aps": val_aps if label_mode == "endpoint" else None,
        "stopped_at": stopped_at,
        "best_epoch": best_epoch,
        "train_idx": train_idx.detach(),
        "val_idx": val_idx.detach(),
        "test_idx": test_idx.detach(),
        "positive_class_weight": label_weight,
        "label_mode": label_mode,
    }

    return model


@torch.no_grad()
def mine_candidate_edge_scores(
    model: torch.nn.Module,
    attr: Tensor,
    edge_index: Tensor,
    labels: Tensor,
    eval_idx: Tensor,
    candidates: list[tuple[int, int]],
    n_nodes: int,
    *,
    mode: str,
    subset_fraction: float = 0.1,
    n_subsets: int = 100,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Overarching label mining function for candidate labels.

    Label modes
    -----
    subset accuracy drop:
        Scores randomly chosen subsets of the candidate set, by flipping them
        on the clean adjacency and recording the drop in accuracy.
        Labels are constructed by summing all drops that an edge caused
    endpoint:
        Labels indicate if one of the endpoints of a flipped edge changes prediction
        from correct to incorrect
    """
    if device is None:
        device = attr.device

    model = model.to(device)
    model.eval()

    attr = attr.to(device)

    edge_index = edge_index.to(
        device=device,
        dtype=torch.long,
    )

    labels = labels.to(
        device=device,
        dtype=torch.long,
    )

    adj_orig = _dense_adj(
        edge_index=edge_index,
        n_nodes=n_nodes,
        device=device,
    ).float()

    cand_src = torch.tensor(
        [int(u) for u, _ in candidates],
        device=device,
        dtype=torch.long,
    )

    cand_dst = torch.tensor(
        [int(v) for _, v in candidates],
        device=device,
        dtype=torch.long,
    )

    # Compute Clean Stats
    clean_logits = model(attr, adj_orig)
    clean_preds = clean_logits.argmax(dim=-1)
    clean_correct = clean_preds == labels
    clean_accuracy = float((clean_preds[eval_idx] == labels[eval_idx]).float().mean().item())

    if mode == "subset_accuracy_drop":
        exists = adj_orig[cand_src, cand_dst].clone()
        return _mine_subset_accuracy_drop(
            model=model,
            attr=attr,
            labels=labels,
            eval_idx=eval_idx,
            adj_orig=adj_orig,
            cand_src=cand_src,
            cand_dst=cand_dst,
            exists=exists,
            clean_accuracy=clean_accuracy,
            subset_fraction=subset_fraction,
            n_subsets=n_subsets,
            seed=seed,
        )
    if mode == "endpoint":
        return _mine_endpoint_flips(
            model=model,
            attr=attr,
            labels=labels,
            adj_orig=adj_orig,
            candidates=candidates,
            clean_correct=clean_correct,
            clean_accuracy=clean_accuracy,
        )

@torch.no_grad()
def _mine_subset_accuracy_drop(
    *,
    model: torch.nn.Module,
    attr: Tensor,
    labels: Tensor,
    eval_idx: Tensor,
    adj_orig: Tensor,
    cand_src: Tensor,
    cand_dst: Tensor,
    exists: Tensor,
    clean_accuracy: float,
    subset_fraction: float,
    n_subsets: int,
    seed: int,
) -> dict[str, Any]:

    device = adj_orig.device
    n_cands = int(cand_src.numel())
    subset_size = subset_fraction * n_cands
    rng = np.random.default_rng(seed)
    labels_raw = np.zeros(n_cands, dtype=np.float64)
    inclusion_count = np.zeros(n_cands, dtype=np.int64)
    drop_per_subset = np.zeros(n_subsets, dtype=np.float64)

    adj_work = adj_orig.clone()
    y_eval = labels[eval_idx]

    for subset_idx in range(n_subsets):
        # Chose edge randomly from all candidate edges
        chosen = rng.choice(n_cands, size=int(subset_size), replace=False)
        chosen_t = torch.as_tensor(chosen,device=device,dtype=torch.long)
        src = cand_src[chosen_t]
        dst = cand_dst[chosen_t]

        # Flip edge
        original_values = exists[chosen_t]
        flipped_values = 1.0 - original_values
        adj_work[src, dst] = flipped_values
        adj_work[dst, src] = flipped_values

        # Compute accuracy drop when subset edges are flipped
        pert_accuracy = accuracy(model, attr, labels, eval_idx ,adj_work)
        drop = max(0.0, clean_accuracy - pert_accuracy)
        drop_per_subset[subset_idx] = drop

        # Add the recorded accuracy drop to chosen edges labels
        labels_raw[chosen] += drop
        inclusion_count[chosen] += 1

        # Restore the original adjacency
        adj_work[src, dst] = original_values
        adj_work[dst, src] = original_values

    # Min-Max norm the labels
    max_raw = float(labels_raw.max())
    min_raw = float(labels_raw.min())
    labels_norm = ((labels_raw - min_raw) / (max_raw - min_raw)).astype(np.float32)

    mean_drop_when_selected = np.divide(
        labels_raw,
        inclusion_count,
        out=np.zeros_like(labels_raw),
        where=inclusion_count > 0,
    )

    n_positive = int((labels_norm > 0).sum())

    print(f"Clean evaluation accuracy: {clean_accuracy:.4f}")
    print(
        f"Subset size: {subset_size} "
        f"({subset_fraction:.0%} of {n_cands} candidates)"
    )
    print(f"Subsets run: {n_subsets}")
    print(
        "Accuracy drop per subset: "
        f"mean={drop_per_subset.mean():.4f}  "
        f"max={drop_per_subset.max():.4f}  "
        f"zero_rate={(drop_per_subset == 0).mean():.2%}"
    )
    print(
        f"Edges with score > 0: {n_positive}/{n_cands} "
        f"({100.0 * n_positive / n_cands:.1f}%)"
    )

    return {
        "mode": "subset_accuracy_drop",
        "labels_raw": labels_raw,
        "labels_norm": labels_norm,
        "drop_per_subset": drop_per_subset,
        "clean_accuracy": clean_accuracy,
        "subset_size": subset_size,
        "inclusion_count": inclusion_count,
        "mean_drop_when_selected": mean_drop_when_selected,
        "exists": exists.detach().cpu().numpy().astype(np.float32),
    }


@torch.no_grad()
def _mine_endpoint_flips(
    *,
    model,
    attr,
    labels,
    adj_orig,
    candidates,
    clean_correct,
    clean_accuracy,
):
    device = adj_orig.device

    model.eval()

    labels = labels.to(device)
    clean_correct = clean_correct.to(device)

    n_candidates = len(candidates)

    if n_candidates == 0:
        raise ValueError("Candidate list is empty.")

    adj_work = adj_orig.clone()

    labels_out = np.zeros(
        n_candidates,
        dtype=np.float32,
    )

    exists = np.zeros(
        n_candidates,
        dtype=np.float32,
    )

    for i, (u, v) in enumerate(candidates):
        u = int(u)
        v = int(v)

        # Determine whether to add or remove the edge
        edge_exists = bool(adj_orig[u, v] > 0.5 or adj_orig[v, u] > 0.5)
        exists[i] = float(edge_exists)
        flipped_value = (0.0 if edge_exists else 1.0)

        # Flip the edge and get predictions
        adj_work[u, v] = flipped_value
        adj_work[v, u] = flipped_value
        pert_preds = model(attr,adj_work).argmax(dim=-1)

        # Determines if edge flipping lead to prediction change from correct -> incorrect
        u_hit = (bool(clean_correct[u]) and pert_preds[u] != labels[u])
        v_hit = (bool(clean_correct[v]) and pert_preds[v] != labels[v])

        # Construct the endpoint labels
        labels_out[i] = float(u_hit or v_hit)

        # Restore clean adjacency matrix
        adj_work[u, v] = adj_orig[u, v]
        adj_work[v, u] = adj_orig[v, u]

        n_positive = int(labels_out.sum())

    print(f"Clean accuracy: {clean_accuracy:.4f}")
    print(f"Candidates evaluated: {n_candidates}")
    print(f"Endpoint hits: "f"{n_positive}/{n_candidates} "f"({n_positive / n_candidates:.2%})")

    return {
        "mode": "endpoint",
        "labels_raw": labels_out,
        "labels_norm": labels_out,
        "endpoint_labels": labels_out,
        "exists": exists,
        "clean_accuracy": clean_accuracy,
    }

def sample_random_candidates(num_nodes: int, count: int, *, seed: int,):
    """Sample candidate edges"""
    rng = random.Random(int(seed))
    sampled = set()
    while len(sampled) < count:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        sampled.add((min(u, v), max(u, v)))
    return sampled


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _compute_clean_stats(context):

    n_nodes = context["n_nodes"]
    edge_index = context["edge_index"]
    attr = context["attr"]
    adj = context["adj_matrix"]
    labels = context["labels"]
    model = context["model"]
    device = attr.device

    # Build nx Graph for degree and pagerank
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(
        (int(u),int(v))
        for u, v in zip(
            edge_index[0],
            edge_index[1],
        )
        if u != v
    )

    # Compute degree and pagerank from reconstructed nx Graph
    degree = np.asarray([graph.degree(node) for node in range(n_nodes)],dtype=float)
    pagerank_dict = nx.pagerank(graph,alpha=0.85)
    pagerank = np.asarray([pagerank_dict[node] for node in range(n_nodes)],dtype=float)

    model.eval()

    with torch.no_grad():
        # Compute confidence and margin
        logits = model(attr, adj)
        probabilities = torch.softmax(logits, dim=-1)
        prediction = logits.argmax(dim=-1)
        confidence = probabilities.max(dim=-1).values

        top2 = torch.topk(probabilities,k=2,dim=-1).values
        margin = (top2[:, 0] - top2[:, 1])

    return {
        "degree":degree,
        "pagerank":pagerank,
        "prediction":prediction.detach().cpu().numpy(),
        "confidence":confidence.detach().cpu().numpy(),
        "margin": margin.detach().cpu().numpy(),
        "labels": labels.detach().cpu().numpy(),
    }

# Flips one edge from the candidate set identified by the index
def _flip_candidate_sparse(adj, candidates, exists, candidate_index):
    u, v = candidates[candidate_index]
    delta = -1.0 if exists[candidate_index] else 1.0
    change = SparseTensor(
        row=torch.tensor([u, v], device=adj.device()),
        col=torch.tensor([v, u], device=adj.device()),
        value=torch.tensor([delta, delta], device=adj.device()),
        sparse_sizes=adj.sparse_sizes(),
    )
    return (adj + change).coalesce()


def _candidates_to_linear_ids(candidates, n_nodes):
    pairs = torch.tensor(candidates, dtype=torch.long).T
    linear_ids = PRBCD.pairs_to_linear_uppertri(pairs, n_nodes).long()
    return linear_ids
