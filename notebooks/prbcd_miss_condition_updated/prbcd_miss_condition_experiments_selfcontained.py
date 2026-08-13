# Auto-generated from prbcd_miss_condition_experiments_selfcontained.ipynb

# %% [cell 1]
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
import gc
import inspect
import json
import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

# Local, in-memory overlay. No repository files are modified.
from activate_local_prbcd_miss import (
    PACKAGE_DIR,
    PROJECT_ROOT,
    activate,
    status as local_overlay_status,
)

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRBCD, experiment_global_attack_direct = activate()

from experiments import experiment_train
from sparse_smoothing.utils import load_and_standardize

print("Project root:", PROJECT_ROOT)
print("Package folder:", PACKAGE_DIR)
print("Local overlay:", local_overlay_status())

# %% [cell 3]
# ============================================================
# Dataset and victim-model configuration
# ============================================================

DATASET = "pubmed"
SEEDS = [0]

if not SEEDS:
    raise ValueError("SEEDS must contain at least one integer seed.")
SEEDS = [int(seed) for seed in SEEDS]
SEED = SEEDS[0]
N_SEEDS = len(SEEDS)

MODEL_NAME = "GCN"
MODEL_LABEL = "GCN"
MODEL_STORAGE_TYPE = "demo_custom_split"

DROPOUT_VICTIM = 0.5
LR_VICTIM = 1e-2
WEIGHT_DECAY_VICTIM = 1e-3
PATIENCE_VICTIM = 300
MAX_EPOCHS_VICTIM = 3000

# experiment_train uses the repository cache. Existing compatible
# victim models may be reused by the underlying storage layer.
VICTIM_DEVICE = "cpu"
VICTIM_DATA_DEVICE = "cpu"

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
DATASET_PATH = DATA_DIR / f"{DATASET}.npz"

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset file not found: {DATASET_PATH}. "
        "Place the dataset in the repository data folder."
    )

CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Configured dataset={DATASET}, seeds={SEEDS}, model={MODEL_LABEL}")

# %% [cell 4]
def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch before every independent run."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def activate_seed_context(context: dict) -> None:
    """Expose one trained victim context through legacy notebook globals."""
    keys = [
        "seed", "train_statistics", "clean_acc", "model", "graph",
        "idx_train", "idx_val", "idx_test", "device", "n_nodes",
        "n_edges_directed", "n_undirected", "attr_matrix",
        "adj_matrix", "labels_raw", "edge_index", "edge_weight",
        "attr", "labels",
    ]
    for key in keys:
        if key in context:
            globals()["SEED" if key == "seed" else key] = context[key]


def aggregate_over_seeds(
    frame: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return mean/std/SEM and number of distinct victim seeds."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    if "seed" not in frame.columns:
        raise KeyError("Every raw table must contain a seed column.")
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing grouping columns: {missing}")
    if metric_cols is None:
        metric_cols = [
            column for column in frame.select_dtypes(include=[np.number]).columns
            if column not in set(group_cols) | {"seed"}
        ]
    metric_cols = [column for column in metric_cols if column in frame.columns]
    if not group_cols:
        row = {"n_seeds": int(frame["seed"].nunique())}
        for metric in metric_cols:
            values = pd.to_numeric(frame[metric], errors="coerce")
            count = int(values.notna().sum())
            std = float(values.std(ddof=1)) if count > 1 else 0.0
            row[f"{metric}_mean"] = float(values.mean()) if count else np.nan
            row[f"{metric}_std"] = std
            row[f"{metric}_count"] = count
            row[f"{metric}_sem"] = std / np.sqrt(max(1, count))
        return pd.DataFrame([row])
    grouped = frame.groupby(group_cols, dropna=False)
    stats = grouped[metric_cols].agg(["mean", "std", "count"])
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    stats = stats.reset_index()
    for metric in metric_cols:
        stats[f"{metric}_std"] = stats[f"{metric}_std"].fillna(0.0)
        stats[f"{metric}_sem"] = stats[f"{metric}_std"] / np.sqrt(
            stats[f"{metric}_count"].clip(lower=1)
        )
    seed_counts = grouped["seed"].nunique().rename("n_seeds").reset_index()
    return stats.merge(seed_counts, on=group_cols, how="left")


def add_mean_std_band(ax, x, mean, std, *, label=None, **plot_kwargs):
    x = np.asarray(x)
    mean = np.asarray(mean, dtype=float)
    std = np.nan_to_num(np.asarray(std, dtype=float), nan=0.0)
    line = ax.plot(x, mean, label=label, **plot_kwargs)[0]
    ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=line.get_color())
    return line

set_global_seed(SEED)

# %% [cell 5]
# ============================================================
# Train/load victim models and build all RQ1 seed contexts
# ============================================================

seed_contexts = []
train_curve_rows = []
train_summary_rows = []

for seed in SEEDS:
    set_global_seed(seed)
    print(f"Training or loading victim model for seed={seed}")

    stats = experiment_train.run(
        data_dir=str(DATA_DIR),
        dataset=DATASET,
        model_params=dict(
            label=MODEL_LABEL,
            model=MODEL_NAME,
            do_cache_adj_prep=True,
            n_filters=64,
            dropout=DROPOUT_VICTIM,
            svd_params=None,
            jaccard_params=None,
            gdc_params={"alpha": 0.15, "k": 64},
        ),
        train_params=dict(
            lr=LR_VICTIM,
            weight_decay=WEIGHT_DECAY_VICTIM,
            patience=PATIENCE_VICTIM,
            max_epochs=MAX_EPOCHS_VICTIM,
        ),
        binary_attr=False,
        make_undirected=True,
        seed=int(seed),
        artifact_dir=str(CACHE_DIR),
        model_storage_type=MODEL_STORAGE_TYPE,
        ppr_cache_params=dict(),
        device=VICTIM_DEVICE,
        data_device=VICTIM_DATA_DEVICE,
        display_steps=100,
        debug_level="info",
        custom_split_ratios=None,
    )

    model_seed = stats["model"]
    graph_seed = stats["graph"]
    idx_train_seed = stats["idx_train"]
    idx_val_seed = stats["idx_val"]
    idx_test_seed = stats["idx_test"]
    model_seed.eval()

    attr_matrix_seed, adj_matrix_seed, labels_raw_seed = graph_seed[:3]
    model_device = next(model_seed.parameters()).device
    row, col, value = adj_matrix_seed.coo()
    edge_index_seed = torch.stack([row, col], dim=0).long().to(model_device)
    edge_weight_seed = (
        torch.ones(edge_index_seed.size(1), dtype=torch.float32, device=model_device)
        if value is None else value.float().to(model_device)
    )
    attr_seed = attr_matrix_seed.float().to(model_device)
    labels_seed = labels_raw_seed.long().to(model_device)
    n_nodes_seed = int(adj_matrix_seed.sizes()[0])
    n_edges_directed_seed = int(adj_matrix_seed.nnz())
    n_undirected_seed = n_edges_directed_seed // 2
    clean_acc_seed = float(stats["accuracy"])

    context = {
        "seed": int(seed),
        "train_statistics": stats,
        "clean_acc": clean_acc_seed,
        "model": model_seed,
        "graph": graph_seed,
        "idx_train": idx_train_seed,
        "idx_val": idx_val_seed,
        "idx_test": idx_test_seed,
        "device": model_device,
        "n_nodes": n_nodes_seed,
        "n_edges_directed": n_edges_directed_seed,
        "n_undirected": n_undirected_seed,
        "attr_matrix": attr_matrix_seed,
        "adj_matrix": adj_matrix_seed,
        "labels_raw": labels_raw_seed,
        "edge_index": edge_index_seed,
        "edge_weight": edge_weight_seed,
        "attr": attr_seed,
        "labels": labels_seed,
    }
    seed_contexts.append(context)

    for split, values in (("train", stats.get("trace_train", [])),
                          ("validation", stats.get("trace_val", []))):
        for epoch, loss in enumerate(values, start=1):
            train_curve_rows.append({
                "seed": int(seed), "split": split,
                "epoch": int(epoch), "loss": float(loss),
            })

    train_summary_rows.append({
        "seed": int(seed),
        "clean_accuracy": clean_acc_seed,
        "n_train_epochs": len(stats.get("trace_train", [])),
        "n_val_epochs": len(stats.get("trace_val", [])),
    })

SEED_CONTEXTS = seed_contexts
SEED_CONTEXT_BY_SEED = {context["seed"]: context for context in seed_contexts}
activate_seed_context(SEED_CONTEXTS[0])

if len(SEED_CONTEXTS) != len(SEEDS):
    raise RuntimeError("Not every configured seed produced a victim context.")

shape_signatures = {
    (context["n_nodes"], context["n_undirected"])
    for context in SEED_CONTEXTS
}
if len(shape_signatures) != 1:
    raise RuntimeError(
        "The graph structure differs across victim seeds; paired edge IDs "
        "would not be comparable."
    )

train_curve_df = pd.DataFrame(train_curve_rows)
train_summary_raw_df = pd.DataFrame(train_summary_rows)
train_summary_df = aggregate_over_seeds(
    train_summary_raw_df,
    group_cols=[],
    metric_cols=["clean_accuracy", "n_train_epochs", "n_val_epochs"],
)

display(train_summary_df)
print("Generated SEED_CONTEXTS for:", sorted(SEED_CONTEXT_BY_SEED))
print("Victim cache:", CACHE_DIR)
print("Graph nodes:", SEED_CONTEXTS[0]["n_nodes"])
print("Undirected edges:", SEED_CONTEXTS[0]["n_undirected"])

# %% [cell 6]
# Optional victim-training diagnostic plot.
if not train_curve_df.empty:
    curve_summary = (
        train_curve_df.groupby(["split", "epoch"], as_index=False)
        .agg(loss_mean=("loss", "mean"), loss_std=("loss", "std"),
             n_seeds=("seed", "nunique"))
    )
    curve_summary["loss_std"] = curve_summary["loss_std"].fillna(0.0)
    curve_summary = curve_summary[curve_summary["n_seeds"] == N_SEEDS]
    if not curve_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for split, group in curve_summary.groupby("split", sort=False):
            group = group.sort_values("epoch")
            add_mean_std_band(
                ax, group["epoch"], group["loss_mean"], group["loss_std"],
                label=f"{split} mean ± SD",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Victim training over {N_SEEDS} seed(s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# %% [cell 8]
MISS_EPSILON = 0.01
# Increase this only when every configured block remains larger than the resulting attack budget.

SMALL_BLOCK_SIZES = [1_000, 5_000]
LARGE_BLOCK_SIZE = 50_000

EPOCHS = 300
FINE_TUNE_EPOCHS = 250
N_RESAMPLING_EPOCHS = EPOCHS - FINE_TUNE_EPOCHS
WITH_EARLY_STOPPING = False

REPEATS_PER_SEED = 1

# Checkpoints are evaluated after the update/projection of the corresponding
# epoch and before resampling.
CHECKPOINT_EPOCHS = sorted({
    epoch
    for epoch in [0, 9, 24, N_RESAMPLING_EPOCHS - 1, 99, EPOCHS - 1]
    if 0 <= epoch < EPOCHS
})

# Candidate construction for the counterfactual probe replay.
N_TOP_SIGNAL_LARGE_ONLY = 75
N_RANDOM_LARGE_ONLY = 75
N_RANDOM_UNSEEN_CONTROLS = 150
MAX_LARGE_FINAL_MISSED = 100
MAX_TOTAL_PROBES = 400

# Every candidate gradient must come from its own B-sized block in which
# exactly one minimum-weight block edge is replaced by that candidate.
EXPECTED_PROBE_GRADIENT_MODE = "isolated_single_swap"

# A positive delta means that the one-edge-replaced block has a larger
# post-step PRBCD attack loss than the unchanged block after the same step.
POSITIVE_DELTA_TOL = 1e-5
ROBUST_POSITIVE_PROBABILITY = 0.60
ROBUST_MEDIAN_DELTA = 1e-5

# Causal injection experiment.
N_INJECTION_EDGES = 10

# Retention details can be large. The notebook samples at most this many
# candidates from each resampling event for plotting.
STORE_RESAMPLE_EDGE_DETAILS = True
RETENTION_ROWS_PER_EVENT = 5_000

# Expensive phases can be switched independently.
RUN_DISCOVERY_PHASE = True
RUN_PROBE_PHASE = True
RUN_INJECTION_PHASE = True
RUN_RETENTION_ABLATION = False

RETENTION_POLICIES = [
    "native",
    "random_keep",
    "reverse_keep",
    "full_resample",
]
RETENTION_ABLATION_BLOCK_SIZES = [
    min(SMALL_BLOCK_SIZES),
    LARGE_BLOCK_SIZE,
]

ARTIFACT_DIR = str(PROJECT_ROOT / "cache")
PERT_ADJ_STORAGE_TYPE = "evasion_global_adj"
PERT_ATTR_STORAGE_TYPE = "evasion_global_attr"

BASE_OUT_DIR = (
    PACKAGE_DIR
    / "outputs"
    / "prbcd_miss_condition_experiments"
)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = BASE_OUT_DIR / f"{DATASET}__{RUN_ID}"
RAW_DIR = OUT_DIR / "raw_diagnostics"
PLOT_DIR = OUT_DIR / "plots"

for directory in [OUT_DIR, RAW_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

(BASE_OUT_DIR / "latest_run.txt").write_text(
    str(OUT_DIR.resolve()),
    encoding="utf-8",
)

# %% [cell 9]
# Verify that the patched PRBCD class is loaded.
required_args = {
    "miss_diagnostics_enabled",
    "miss_sampling_seed",
    "miss_checkpoint_epochs",
    "miss_probe_linear_ids",
    "miss_injection_linear_ids",
    "miss_retention_policy",
}
missing_args = required_args - set(inspect.signature(PRBCD.__init__).parameters)
if missing_args:
    raise RuntimeError(
        "The local PRBCD diagnostics overlay is not active. Missing "
        f"constructor arguments: {sorted(missing_args)}. Run the import/"
        "activation cell again."
    )

if not SEED_CONTEXTS:
    raise RuntimeError("Victim-data generation produced no SEED_CONTEXTS.")

N_NODES = int(SEED_CONTEXTS[0]["n_nodes"])
N_UNDIRECTED = int(SEED_CONTEXTS[0]["n_undirected"])
N_POSSIBLE_EDGES = N_NODES * (N_NODES - 1) // 2
ATTACK_BUDGET = max(1, round(MISS_EPSILON * N_UNDIRECTED))

ALL_BLOCK_SIZES = sorted(set(SMALL_BLOCK_SIZES + [LARGE_BLOCK_SIZE]))
if any(block_size <= ATTACK_BUDGET for block_size in ALL_BLOCK_SIZES):
    invalid = [b for b in ALL_BLOCK_SIZES if b <= ATTACK_BUDGET]
    raise ValueError(
        f"All block sizes must exceed attack budget={ATTACK_BUDGET}. "
        f"Invalid block sizes: {invalid}. Lower MISS_EPSILON or increase the blocks."
    )

graph_sparse = load_and_standardize(
    str(PROJECT_ROOT / "data" / f"{DATASET}.npz")
)

print("Output:", OUT_DIR.resolve())
print("Possible edge flips:", f"{N_POSSIBLE_EDGES:,}")
print("Attack budget:", ATTACK_BUDGET)
print("Checkpoints:", CHECKPOINT_EPOCHS)

# %% [cell 11]
def _scalar(value, default=np.nan):
    if value is None:
        return default
    try:
        return float(torch.as_tensor(value).detach().cpu().item())
    except Exception:
        return default


def _final_accuracy(result):
    rows = result.get("results", []) or []
    if not rows or not isinstance(rows[0], dict):
        return np.nan
    return _scalar(rows[0].get("accuracy"))


def _diagnostics(result):
    stats = result.get("attack_statistics", {}) or {}
    diagnostics = stats.get("miss_diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("Result contains no miss_diagnostics dictionary.")
    return diagnostics


def _tensor_ids(value):
    if value is None:
        return torch.empty(0, dtype=torch.long)
    return torch.as_tensor(value).detach().cpu().long().flatten()


def _sampling_seed(victim_seed, repeat):
    # Intentionally independent of block size so paired conditions begin from
    # the same random stream.
    return 730_000 + int(victim_seed) * 10_000 + int(repeat)


def _run_prbcd(
    *,
    victim_seed,
    block_size,
    sampling_seed,
    probe_ids=None,
    checkpoint_epochs=None,
    injection_ids=None,
    injection_epoch=-1,
    retention_policy="native",
    store_resample_edge_details=False,
):
    probe_ids = [int(x) for x in (probe_ids or [])]
    injection_ids = [int(x) for x in (injection_ids or [])]
    checkpoint_epochs = [int(x) for x in (checkpoint_epochs or [])]

    attack_params = {
        "block_size": int(block_size),
        "epochs": int(EPOCHS),
        "fine_tune_epochs": int(FINE_TUNE_EPOCHS),
        "with_early_stopping": bool(WITH_EARLY_STOPPING),
        "keep_heuristic": "WeightOnly",
        "do_synchronize": True,
        "loss_type": "tanhMargin",

        # New diagnostics. Standard PRBCD is still selected by use_cert="none".
        "miss_diagnostics_enabled": True,
        "miss_sampling_seed": int(sampling_seed),
        "miss_checkpoint_epochs": checkpoint_epochs,
        "miss_store_checkpoint_blocks": True,
        "miss_store_resample_edge_details": bool(
            store_resample_edge_details
        ),
        "miss_track_edge_signals": True,
        "miss_retention_policy": str(retention_policy),
        "miss_probe_linear_ids": probe_ids,
        "miss_probe_max_candidates": max(MAX_TOTAL_PROBES, len(probe_ids)),
        "miss_injection_linear_ids": injection_ids,
        "miss_injection_epoch": int(injection_epoch),
    }

    set_global_seed(int(sampling_seed))
    started = timer()
    result = experiment_global_attack_direct.run(
        graph=graph_sparse,
        data_dir=str(PROJECT_ROOT / "data"),
        dataset=DATASET,
        attack="PRBCD",
        attack_params=attack_params,
        selector_params={},
        epsilons=[MISS_EPSILON],
        binary_attr=False,
        make_undirected=True,
        seed=int(victim_seed),
        artifact_dir=ARTIFACT_DIR,
        pert_adj_storage_type=PERT_ADJ_STORAGE_TYPE,
        pert_attr_storage_type=PERT_ATTR_STORAGE_TYPE,
        model_label=MODEL_LABEL,
        model_storage_type=MODEL_STORAGE_TYPE,
        device="cpu",
        data_device="cpu",
        debug_level="info",
        semi=True,
        use_cert="none",
    )
    return result, timer() - started


def _save_diagnostics(diagnostics, name):
    path = RAW_DIR / f"{name}.pt"
    torch.save(diagnostics, path)
    return path


def _load_diagnostics(path):
    return torch.load(path, map_location="cpu")


def _edge_summary_df(diagnostics):
    summary = diagnostics.get("edge_summary", {}) or {}
    ids = _tensor_ids(summary.get("linear_ids"))
    frame = pd.DataFrame({"linear_id": ids.numpy()})
    for name in [
        "first_seen_epoch",
        "last_seen_epoch",
        "exposure_count",
        "max_weight",
        "positive_gradient_sum",
        "is_final",
    ]:
        value = summary.get(name)
        if value is not None:
            tensor = torch.as_tensor(value).detach().cpu().flatten()
            frame[name] = tensor.numpy()
    return frame


def _sample_uniform_unseen(count, forbidden, rng):
    """Uniform rejection sampling without constructing the full complement."""
    forbidden = set(int(x) for x in forbidden)
    selected = set()
    max_attempts = max(100_000, 100 * count)
    attempts = 0
    while len(selected) < count and attempts < max_attempts:
        need = count - len(selected)
        draws = rng.integers(
            0,
            N_POSSIBLE_EDGES,
            size=max(2 * need, 1_000),
            dtype=np.int64,
        )
        attempts += len(draws)
        for edge_id in np.unique(draws):
            edge_id = int(edge_id)
            if edge_id not in forbidden and edge_id not in selected:
                selected.add(edge_id)
                if len(selected) == count:
                    break
    if len(selected) < count:
        raise RuntimeError(
            f"Could sample only {len(selected)} of {count} unseen controls."
        )
    return sorted(selected)


def _resample_detail_sample(diagnostics, run_metadata, max_rows_per_event):
    rows = []
    rng = np.random.default_rng(
        int(run_metadata["sampling_seed"]) + 909
    )
    for event in diagnostics.get("resample_events", []) or []:
        before_ids = event.get("before_ids")
        before_weights = event.get("before_weights")
        kept_ids = event.get("kept_ids")
        if before_ids is None or before_weights is None or kept_ids is None:
            continue
        before_ids = _tensor_ids(before_ids)
        before_weights = torch.as_tensor(before_weights).cpu().float().flatten()
        before_gradients = event.get("before_gradients")
        if before_gradients is not None:
            before_gradients = torch.as_tensor(
                before_gradients
            ).cpu().float().flatten()
        kept_ids = set(_tensor_ids(kept_ids).tolist())
        n = before_ids.numel()
        choose = np.arange(n)
        if n > max_rows_per_event:
            choose = rng.choice(n, size=max_rows_per_event, replace=False)
        for idx in choose:
            edge_id = int(before_ids[int(idx)].item())
            rows.append({
                **run_metadata,
                "epoch": event.get("epoch"),
                "retention_policy": event.get("policy"),
                "linear_id": edge_id,
                "weight": float(before_weights[int(idx)].item()),
                "gradient": (
                    float(before_gradients[int(idx)].item())
                    if before_gradients is not None
                    else np.nan
                ),
                "retained": edge_id in kept_ids,
                "n_eps": event.get("n_eps"),
                "n_discard": event.get("n_discard"),
                "raw_draw_count": event.get("raw_draw_count"),
            })
    return rows


def _save_figure(filename):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.show()

# %% [cell 13]
discovery_rows = []
discovery_paths = []
retention_sample_rows = []

if RUN_DISCOVERY_PHASE:
    total = len(SEEDS) * REPEATS_PER_SEED * len(ALL_BLOCK_SIZES)
    completed = 0
    for victim_seed in SEEDS:
        for repeat in range(REPEATS_PER_SEED):
            sampling_seed = _sampling_seed(victim_seed, repeat)
            for block_size in ALL_BLOCK_SIZES:
                completed += 1
                print(
                    f"Discovery {completed}/{total}: seed={victim_seed}, "
                    f"repeat={repeat}, B={block_size:,}"
                )
                result, runtime = _run_prbcd(
                    victim_seed=victim_seed,
                    block_size=block_size,
                    sampling_seed=sampling_seed,
                    checkpoint_epochs=CHECKPOINT_EPOCHS,
                    store_resample_edge_details=(
                        STORE_RESAMPLE_EDGE_DETAILS
                    ),
                )
                diagnostics = _diagnostics(result)
                name = (
                    f"discovery_seed{victim_seed}_rep{repeat}_B{block_size}"
                )
                path = _save_diagnostics(diagnostics, name)
                metadata = diagnostics["metadata"]
                row = {
                    "seed": int(victim_seed),
                    "repeat": int(repeat),
                    "sampling_seed": int(sampling_seed),
                    "block_size": int(block_size),
                    "final_accuracy": _final_accuracy(result),
                    "runtime_seconds": float(runtime),
                    "n_ever_seen": int(metadata.get("n_ever_seen", 0)),
                    "coverage_fraction": float(
                        metadata.get("coverage_fraction", np.nan)
                    ),
                    "n_final": int(metadata.get("n_final", 0)),
                    "total_raw_resample_draws": int(
                        metadata.get("total_raw_resample_draws", 0)
                    ),
                    "diagnostics_path": str(path),
                }
                discovery_rows.append(row)
                discovery_paths.append(row)
                if STORE_RESAMPLE_EDGE_DETAILS:
                    retention_sample_rows.extend(
                        _resample_detail_sample(
                            diagnostics,
                            {
                                "seed": int(victim_seed),
                                "repeat": int(repeat),
                                "sampling_seed": int(sampling_seed),
                                "block_size": int(block_size),
                            },
                            RETENTION_ROWS_PER_EVENT,
                        )
                    )
                del result, diagnostics
                gc.collect()

    discovery_df = pd.DataFrame(discovery_rows)
    discovery_df.to_csv(OUT_DIR / "discovery_runs.csv", index=False)
    retention_sample_df = pd.DataFrame(retention_sample_rows)
    retention_sample_df.to_csv(
        OUT_DIR / "retention_candidate_sample.csv",
        index=False,
    )
else:
    discovery_df = pd.read_csv(OUT_DIR / "discovery_runs.csv")
    retention_path = OUT_DIR / "retention_candidate_sample.csv"
    retention_sample_df = (
        pd.read_csv(retention_path)
        if retention_path.exists()
        else pd.DataFrame()
    )

display(discovery_df.sort_values(["seed", "repeat", "block_size"]))

# %% [cell 15]
seed_discovery_df = (
    discovery_df.groupby(["seed", "block_size"], as_index=False)
    .agg(
        final_accuracy=("final_accuracy", "mean"),
        coverage_fraction=("coverage_fraction", "mean"),
        n_ever_seen=("n_ever_seen", "mean"),
        runtime_seconds=("runtime_seconds", "mean"),
    )
)
summary_discovery_df = (
    seed_discovery_df.groupby("block_size", as_index=False)
    .agg(
        final_accuracy_mean=("final_accuracy", "mean"),
        final_accuracy_std=("final_accuracy", "std"),
        coverage_mean=("coverage_fraction", "mean"),
        coverage_std=("coverage_fraction", "std"),
        n_ever_seen_mean=("n_ever_seen", "mean"),
        n_seeds=("seed", "nunique"),
    )
)
summary_discovery_df["accuracy_ci95"] = (
    1.96
    * summary_discovery_df["final_accuracy_std"]
    / np.sqrt(summary_discovery_df["n_seeds"])
)
summary_discovery_df["coverage_ci95"] = (
    1.96
    * summary_discovery_df["coverage_std"]
    / np.sqrt(summary_discovery_df["n_seeds"])
)
summary_discovery_df.to_csv(
    OUT_DIR / "discovery_summary.csv",
    index=False,
)

display(summary_discovery_df)

plt.figure(figsize=(8, 5))
plt.errorbar(
    summary_discovery_df["block_size"],
    summary_discovery_df["final_accuracy_mean"],
    yerr=summary_discovery_df["accuracy_ci95"].fillna(0),
    marker="o",
    capsize=4,
)
plt.xscale("log")
plt.xlabel("PRBCD block size")
plt.ylabel("Final attacked accuracy")
plt.title("Attack performance by block size")
plt.grid(alpha=0.3)
_save_figure("01_final_accuracy_by_block_size.png")

plt.figure(figsize=(8, 5))
plt.errorbar(
    summary_discovery_df["block_size"],
    summary_discovery_df["coverage_mean"],
    yerr=summary_discovery_df["coverage_ci95"].fillna(0),
    marker="o",
    capsize=4,
)
plt.xscale("log")
plt.xlabel("PRBCD block size")
plt.ylabel("Fraction of possible edges ever sampled")
plt.title("Realized candidate coverage by block size")
plt.grid(alpha=0.3)
_save_figure("02_coverage_by_block_size.png")

# %% [cell 16]
if not retention_sample_df.empty:
    retention_plot_df = retention_sample_df.dropna(subset=["weight"]).copy()
    retention_plot_df["weight_quantile"] = (
        retention_plot_df.groupby(
            ["seed", "repeat", "block_size", "epoch"]
        )["weight"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                q=10,
                labels=False,
                duplicates="drop",
            )
        )
    )
    retention_curve_df = (
        retention_plot_df.groupby(
            ["block_size", "weight_quantile"],
            as_index=False,
        )
        .agg(retention_probability=("retained", "mean"))
    )
    plt.figure(figsize=(8, 5))
    for block_size in ALL_BLOCK_SIZES:
        part = retention_curve_df[
            retention_curve_df["block_size"] == block_size
        ].sort_values("weight_quantile")
        plt.plot(
            part["weight_quantile"],
            part["retention_probability"],
            marker="o",
            label=f"B={block_size:,}",
        )
    plt.xlabel("Within-event weight decile (0 = lowest)")
    plt.ylabel("Probability of retention")
    plt.title("WeightOnly retention selectivity")
    plt.legend()
    plt.grid(alpha=0.3)
    _save_figure("03_retention_probability_by_weight_decile.png")

# %% [cell 18]
candidate_rows = []
probe_rows = []
probe_run_rows = []

if RUN_PROBE_PHASE:
    for victim_seed in SEEDS:
        for repeat in range(REPEATS_PER_SEED):
            sampling_seed = _sampling_seed(victim_seed, repeat)
            large_row = discovery_df[
                (discovery_df["seed"] == victim_seed)
                & (discovery_df["repeat"] == repeat)
                & (discovery_df["block_size"] == LARGE_BLOCK_SIZE)
            ].iloc[0]
            large_diag = _load_diagnostics(large_row["diagnostics_path"])
            large_seen = set(_tensor_ids(large_diag["ever_seen_ids"]).tolist())
            large_final = set(_tensor_ids(large_diag["final_linear_ids"]).tolist())
            large_summary = _edge_summary_df(large_diag)

            for small_block_size in SMALL_BLOCK_SIZES:
                small_row = discovery_df[
                    (discovery_df["seed"] == victim_seed)
                    & (discovery_df["repeat"] == repeat)
                    & (discovery_df["block_size"] == small_block_size)
                ].iloc[0]
                small_diag = _load_diagnostics(small_row["diagnostics_path"])
                small_seen = set(
                    _tensor_ids(small_diag["ever_seen_ids"]).tolist()
                )

                large_only = large_seen - small_seen
                large_final_missed = sorted(
                    (large_final - small_seen)
                )[:MAX_LARGE_FINAL_MISSED]

                signal_candidates = large_summary[
                    large_summary["linear_id"].isin(large_only)
                ].copy()
                for column in ["max_weight", "positive_gradient_sum"]:
                    if column not in signal_candidates:
                        signal_candidates[column] = 0.0
                signal_candidates = signal_candidates.sort_values(
                    ["max_weight", "positive_gradient_sum"],
                    ascending=False,
                )
                large_top_signal = signal_candidates[
                    "linear_id"
                ].head(N_TOP_SIGNAL_LARGE_ONLY).astype(int).tolist()

                rng = np.random.default_rng(
                    sampling_seed + small_block_size + 41
                )
                large_only_available = np.array(
                    sorted(large_only),
                    dtype=np.int64,
                )
                n_random_large = min(
                    N_RANDOM_LARGE_ONLY,
                    len(large_only_available),
                )
                large_only_random = (
                    rng.choice(
                        large_only_available,
                        size=n_random_large,
                        replace=False,
                    ).astype(int).tolist()
                    if n_random_large > 0
                    else []
                )

                random_unseen = _sample_uniform_unseen(
                    N_RANDOM_UNSEEN_CONTROLS,
                    forbidden=small_seen | large_seen,
                    rng=rng,
                )

                all_probe_ids = []
                for group_ids in [
                    large_final_missed,
                    large_top_signal,
                    large_only_random,
                    random_unseen,
                ]:
                    for edge_id in group_ids:
                        if edge_id not in all_probe_ids:
                            all_probe_ids.append(int(edge_id))
                all_probe_ids = all_probe_ids[:MAX_TOTAL_PROBES]

                flags = {
                    edge_id: {
                        "is_large_final_missed": edge_id in set(large_final_missed),
                        "is_large_top_signal": edge_id in set(large_top_signal),
                        "is_large_only_random": edge_id in set(large_only_random),
                        "is_random_unseen_control": edge_id in set(random_unseen),
                    }
                    for edge_id in all_probe_ids
                }
                for edge_id in all_probe_ids:
                    edge_flags = flags[edge_id]
                    if edge_flags["is_random_unseen_control"]:
                        primary_group = "random_unseen_control"
                    elif edge_flags["is_large_final_missed"]:
                        primary_group = "large_final_missed"
                    elif edge_flags["is_large_top_signal"]:
                        primary_group = "large_top_signal"
                    else:
                        primary_group = "large_only_random"
                    candidate_rows.append({
                        "seed": int(victim_seed),
                        "repeat": int(repeat),
                        "sampling_seed": int(sampling_seed),
                        "small_block_size": int(small_block_size),
                        "large_block_size": int(LARGE_BLOCK_SIZE),
                        "linear_id": int(edge_id),
                        "primary_group": primary_group,
                        **edge_flags,
                    })

                print(
                    f"Probe replay: seed={victim_seed}, B={small_block_size:,}, "
                    f"candidates={len(all_probe_ids)}"
                )
                replay_result, runtime = _run_prbcd(
                    victim_seed=victim_seed,
                    block_size=small_block_size,
                    sampling_seed=sampling_seed,
                    probe_ids=all_probe_ids,
                    checkpoint_epochs=CHECKPOINT_EPOCHS,
                    store_resample_edge_details=False,
                )
                replay_diag = _diagnostics(replay_result)
                replay_seen = set(
                    _tensor_ids(replay_diag["ever_seen_ids"]).tolist()
                )
                replay_match = replay_seen == small_seen
                if not replay_match:
                    warnings.warn(
                        "Probe replay did not exactly reproduce the baseline "
                        f"seen set for seed={victim_seed}, B={small_block_size}."
                    )

                candidate_meta = {
                    row["linear_id"]: row
                    for row in candidate_rows
                    if row["seed"] == victim_seed
                    and row["repeat"] == repeat
                    and row["small_block_size"] == small_block_size
                }
                for probe in replay_diag.get("probe_results", []) or []:
                    edge_id = int(probe["linear_id"])
                    metadata = candidate_meta.get(edge_id, {})
                    probe_rows.append({
                        "seed": int(victim_seed),
                        "repeat": int(repeat),
                        "sampling_seed": int(sampling_seed),
                        "small_block_size": int(small_block_size),
                        "large_block_size": int(LARGE_BLOCK_SIZE),
                        "replay_seen_set_matches": bool(replay_match),
                        **metadata,
                        **probe,
                    })
                replay_path = _save_diagnostics(
                    replay_diag,
                    f"probe_seed{victim_seed}_rep{repeat}_B{small_block_size}",
                )
                probe_run_rows.append({
                    "seed": int(victim_seed),
                    "repeat": int(repeat),
                    "small_block_size": int(small_block_size),
                    "n_probe_ids": len(all_probe_ids),
                    "replay_seen_set_matches": bool(replay_match),
                    "final_accuracy": _final_accuracy(replay_result),
                    "runtime_seconds": float(runtime),
                    "diagnostics_path": str(replay_path),
                    "n_large_only": len(large_only),
                    "n_large_final_missed": len(large_final - small_seen),
                })
                del replay_result, replay_diag, small_diag
                gc.collect()
            del large_diag

    candidate_df = pd.DataFrame(candidate_rows)
    probe_df = pd.DataFrame(probe_rows)
    probe_runs_df = pd.DataFrame(probe_run_rows)
    candidate_df.to_csv(OUT_DIR / "probe_candidates.csv", index=False)
    probe_df.to_csv(OUT_DIR / "probe_results.csv", index=False)
    probe_runs_df.to_csv(OUT_DIR / "probe_runs.csv", index=False)
else:
    candidate_df = pd.read_csv(OUT_DIR / "probe_candidates.csv")
    probe_df = pd.read_csv(OUT_DIR / "probe_results.csv")
    probe_runs_df = pd.read_csv(OUT_DIR / "probe_runs.csv")

# Refuse silently incompatible results from the former joint-probe gradient
# implementation. Old CSV files must be regenerated with RUN_PROBE_PHASE=True.
if not probe_df.empty:
    required_probe_columns = {
        "probe_gradient_mode",
        "probe_candidates_in_context",
        "probe_context_size",
        "gradient_at_eps",
        "one_step_delta_loss",
    }
    missing_probe_columns = required_probe_columns - set(probe_df.columns)
    if missing_probe_columns:
        raise RuntimeError(
            "Probe results do not contain isolated single-swap metadata. "
            "Set RUN_PROBE_PHASE=True and regenerate them. Missing columns: "
            f"{sorted(missing_probe_columns)}"
        )

    observed_probe_modes = set(
        probe_df["probe_gradient_mode"].dropna().astype(str)
    )
    if observed_probe_modes != {EXPECTED_PROBE_GRADIENT_MODE}:
        raise RuntimeError(
            "Unexpected probe gradient mode(s): "
            f"{sorted(observed_probe_modes)}. Expected only "
            f"{EXPECTED_PROBE_GRADIENT_MODE!r}."
        )

    if not (probe_df["probe_candidates_in_context"] == 1).all():
        raise RuntimeError(
            "At least one probe gradient was evaluated with another probe "
            "present in the same temporary block."
        )

    if not (probe_df["probe_context_size"] > 0).all():
        raise RuntimeError("A probe was evaluated on an empty context.")

display(probe_runs_df)
display(probe_df.head())

# %% [cell 20]
probe_df["one_step_positive"] = (
    probe_df["one_step_delta_loss"] > POSITIVE_DELTA_TOL
)

candidate_effect_df = (
    probe_df.groupby(
        [
            "seed",
            "repeat",
            "small_block_size",
            "linear_id",
            "primary_group",
        ],
        as_index=False,
    )
    .agg(
        mean_one_step_delta_loss=("one_step_delta_loss", "mean"),
        median_one_step_delta_loss=("one_step_delta_loss", "median"),
        max_one_step_delta_loss=("one_step_delta_loss", "max"),
        probability_one_step_positive=("one_step_positive", "mean"),
        mean_gradient_at_eps=("gradient_at_eps", "mean"),
        max_gradient_at_eps=("gradient_at_eps", "max"),
        mean_candidate_post_step_weight=(
            "candidate_post_step_weight",
            "mean",
        ),
        n_checkpoints=("epoch", "nunique"),
    )
)
candidate_effect_df["robustly_harmful"] = (
    (
        candidate_effect_df["probability_one_step_positive"]
        >= ROBUST_POSITIVE_PROBABILITY
    )
    & (
        candidate_effect_df["median_one_step_delta_loss"]
        > ROBUST_MEDIAN_DELTA
    )
)
candidate_effect_df.to_csv(
    OUT_DIR / "candidate_contextual_effect_summary.csv",
    index=False,
)
display(candidate_effect_df.head(20))

# %% [cell 22]
group_order = [
    "large_final_missed",
    "large_top_signal",
    "large_only_random",
    "random_unseen_control",
]
available_groups = [
    group for group in group_order
    if group in set(candidate_effect_df["primary_group"])
]

box_data = [
    candidate_effect_df.loc[
        candidate_effect_df["primary_group"] == group,
        "mean_one_step_delta_loss",
    ].dropna().values
    for group in available_groups
]
plt.figure(figsize=(10, 5))
plt.boxplot(box_data, labels=available_groups, showfliers=False)
plt.axhline(POSITIVE_DELTA_TOL, linestyle="--")
plt.ylabel("Mean one-step PRBCD replacement loss effect")
plt.title("Counterfactual usefulness of edges missed by small blocks")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
_save_figure("04_counterfactual_effect_by_candidate_group.png")

positive_summary = (
    candidate_effect_df.groupby(
        ["small_block_size", "primary_group"],
        as_index=False,
    )
    .agg(
        probability_positive=("probability_one_step_positive", "mean"),
        robust_fraction=("robustly_harmful", "mean"),
    )
)
plt.figure(figsize=(9, 5))
for block_size in SMALL_BLOCK_SIZES:
    part = positive_summary[
        positive_summary["small_block_size"] == block_size
    ].set_index("primary_group").reindex(available_groups)
    plt.plot(
        available_groups,
        part["probability_positive"],
        marker="o",
        label=f"B={block_size:,}",
    )
plt.ylabel("Mean P(positive one-step effect across checkpoints)")
plt.title("How often missed candidates improve the next PRBCD state")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.grid(alpha=0.3)
_save_figure("05_probability_positive_by_group.png")

plt.figure(figsize=(8, 6))
for group in available_groups:
    part = probe_df[probe_df["primary_group"] == group]
    plt.scatter(
        part["gradient_at_eps"],
        part["one_step_delta_loss"],
        s=16,
        alpha=0.35,
        label=group,
    )
plt.axhline(POSITIVE_DELTA_TOL, linestyle="--")
plt.axvline(0, linestyle="--")
plt.xlabel("Candidate gradient at epsilon (isolated single-swap block)")
plt.ylabel("One-step PRBCD replacement loss effect")
plt.title("Isolated PRBCD signal versus one-step counterfactual effect")
plt.legend()
plt.grid(alpha=0.3)
_save_figure("06_gradient_vs_one_step_effect.png")

checkpoint_group_df = (
    probe_df.groupby(
        ["small_block_size", "primary_group", "epoch"],
        as_index=False,
    )
    .agg(mean_delta=("one_step_delta_loss", "mean"))
)
for block_size in SMALL_BLOCK_SIZES:
    plt.figure(figsize=(9, 5))
    block_part = checkpoint_group_df[
        checkpoint_group_df["small_block_size"] == block_size
    ]
    for group in available_groups:
        part = block_part[
            block_part["primary_group"] == group
        ].sort_values("epoch")
        if not part.empty:
            plt.plot(
                part["epoch"],
                part["mean_delta"],
                marker="o",
                label=group,
            )
    plt.axhline(POSITIVE_DELTA_TOL, linestyle="--")
    plt.xlabel("Small-run checkpoint epoch")
    plt.ylabel("Mean one-step PRBCD replacement loss effect")
    plt.title(f"Missed-edge usefulness over time — B={block_size:,}")
    plt.legend()
    plt.grid(alpha=0.3)
    _save_figure(f"07_effect_over_time_B{block_size}.png")

# %% [cell 24]
injection_rows = []
injection_edge_rows = []

if RUN_INJECTION_PHASE:
    for victim_seed in SEEDS:
        for repeat in range(REPEATS_PER_SEED):
            sampling_seed = _sampling_seed(victim_seed, repeat)
            for small_block_size in SMALL_BLOCK_SIZES:
                run_effects = candidate_effect_df[
                    (candidate_effect_df["seed"] == victim_seed)
                    & (candidate_effect_df["repeat"] == repeat)
                    & (
                        candidate_effect_df["small_block_size"]
                        == small_block_size
                    )
                ].copy()
                missed_effects = run_effects[
                    (run_effects["primary_group"] != "random_unseen_control")
                    & (
                        run_effects["mean_one_step_delta_loss"]
                        > POSITIVE_DELTA_TOL
                    )
                ].sort_values(
                    [
                        "robustly_harmful",
                        "mean_one_step_delta_loss",
                        "max_one_step_delta_loss",
                    ],
                    ascending=False,
                )
                harmful_ids = missed_effects[
                    "linear_id"
                ].head(N_INJECTION_EDGES).astype(int).tolist()
                if not harmful_ids:
                    warnings.warn(
                        f"No missed candidates for seed={victim_seed}, "
                        f"B={small_block_size}; skipping injection."
                    )
                    continue

                control_pool = run_effects[
                    run_effects["primary_group"]
                    == "random_unseen_control"
                ].sort_values("linear_id")
                control_ids = control_pool["linear_id"].head(
                    len(harmful_ids)
                ).astype(int).tolist()
                if len(control_ids) < len(harmful_ids):
                    raise RuntimeError(
                        "Not enough random-unseen controls for matched injection."
                    )

                baseline_row = discovery_df[
                    (discovery_df["seed"] == victim_seed)
                    & (discovery_df["repeat"] == repeat)
                    & (discovery_df["block_size"] == small_block_size)
                ].iloc[0]
                baseline_accuracy = float(baseline_row["final_accuracy"])

                # Run the same selected missed/control sets at every configured
                # checkpoint so checkpoint timing is an explicit intervention.
                for injection_epoch in CHECKPOINT_EPOCHS:
                    injection_rows.append({
                        "seed": int(victim_seed),
                        "repeat": int(repeat),
                        "sampling_seed": int(sampling_seed),
                        "small_block_size": int(small_block_size),
                        "condition": "baseline",
                        "injection_epoch": int(injection_epoch),
                        "n_injected": 0,
                        "baseline_accuracy": baseline_accuracy,
                        "final_accuracy": baseline_accuracy,
                        "accuracy_effect": 0.0,
                        "runtime_seconds": float(
                            baseline_row["runtime_seconds"]
                        ),
                    })

                    for condition, ids in [
                        ("harmful_missed", harmful_ids),
                        ("random_unseen", control_ids),
                    ]:
                        print(
                            f"Injection: seed={victim_seed}, "
                            f"B={small_block_size:,}, condition={condition}, "
                            f"epoch={injection_epoch}, n={len(ids)}"
                        )
                        result, runtime = _run_prbcd(
                            victim_seed=victim_seed,
                            block_size=small_block_size,
                            sampling_seed=sampling_seed,
                            injection_ids=ids,
                            injection_epoch=injection_epoch,
                            checkpoint_epochs=[],
                            store_resample_edge_details=False,
                        )
                        diagnostics = _diagnostics(result)
                        path = _save_diagnostics(
                            diagnostics,
                            f"inject_{condition}_epoch{injection_epoch}_"
                            f"seed{victim_seed}_rep{repeat}_"
                            f"B{small_block_size}",
                        )
                        final_accuracy = _final_accuracy(result)
                        injection_rows.append({
                            "seed": int(victim_seed),
                            "repeat": int(repeat),
                            "sampling_seed": int(sampling_seed),
                            "small_block_size": int(small_block_size),
                            "condition": condition,
                            "injection_epoch": int(injection_epoch),
                            "n_injected": len(ids),
                            "baseline_accuracy": baseline_accuracy,
                            "final_accuracy": final_accuracy,
                            # Positive means the injection strengthened the
                            # attack by reducing final attacked accuracy.
                            "accuracy_effect": (
                                baseline_accuracy - final_accuracy
                            ),
                            "runtime_seconds": float(runtime),
                            "diagnostics_path": str(path),
                        })
                        for edge_id in ids:
                            injection_edge_rows.append({
                                "seed": int(victim_seed),
                                "repeat": int(repeat),
                                "small_block_size": int(small_block_size),
                                "condition": condition,
                                "injection_epoch": int(injection_epoch),
                                "linear_id": int(edge_id),
                            })
                        del result, diagnostics
                        gc.collect()

    injection_df = pd.DataFrame(injection_rows)
    injection_edges_df = pd.DataFrame(injection_edge_rows)
    injection_df.to_csv(OUT_DIR / "injection_runs.csv", index=False)
    injection_edges_df.to_csv(
        OUT_DIR / "injection_edges.csv",
        index=False,
    )
else:
    injection_df = pd.read_csv(OUT_DIR / "injection_runs.csv")
    injection_edges_df = pd.read_csv(OUT_DIR / "injection_edges.csv")

display(injection_df)

# %% [cell 26]
if not injection_df.empty:
    injection_seed_df = (
        injection_df.groupby(
            [
                "seed",
                "repeat",
                "small_block_size",
                "injection_epoch",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            final_accuracy=("final_accuracy", "mean"),
            accuracy_effect=("accuracy_effect", "mean"),
        )
    )
    injection_summary_df = (
        injection_seed_df.groupby(
            ["small_block_size", "injection_epoch", "condition"],
            as_index=False,
        )
        .agg(
            accuracy_mean=("final_accuracy", "mean"),
            accuracy_std=("final_accuracy", "std"),
            effect_mean=("accuracy_effect", "mean"),
            effect_std=("accuracy_effect", "std"),
            n_runs=("accuracy_effect", "count"),
            n_seeds=("seed", "nunique"),
        )
    )
    injection_summary_df.to_csv(
        OUT_DIR / "injection_summary.csv",
        index=False,
    )

    plot_conditions = ["random_unseen", "harmful_missed"]
    short_condition = {
        "random_unseen": "random",
        "harmful_missed": "missed",
    }
    for block_size in SMALL_BLOCK_SIZES:
        block_part = injection_seed_df[
            injection_seed_df["small_block_size"] == block_size
        ]
        box_data = []
        box_labels = []
        box_positions = []
        position = 1
        for injection_epoch in CHECKPOINT_EPOCHS:
            for condition in plot_conditions:
                values = block_part.loc[
                    (block_part["injection_epoch"] == injection_epoch)
                    & (block_part["condition"] == condition),
                    "accuracy_effect",
                ].dropna().to_numpy()
                if values.size == 0:
                    continue
                box_data.append(values)
                box_labels.append(
                    f"{injection_epoch}\n{short_condition[condition]}"
                )
                box_positions.append(position)
                position += 1
            position += 0.5

        if box_data:
            plt.figure(figsize=(max(10, 0.8 * len(box_data)), 5.5))
            plt.boxplot(
                box_data,
                positions=box_positions,
                labels=box_labels,
                showfliers=False,
                widths=0.65,
            )
            # Show every raw run as well; this remains informative when only
            # one seed/repeat is configured and a box collapses to a line.
            for pos, values in zip(box_positions, box_data):
                plt.scatter(
                    np.full(values.shape, pos, dtype=float),
                    values,
                    s=22,
                    alpha=0.7,
                )
            plt.axhline(0, linestyle="--")
            plt.xlabel("Injection epoch and injected candidate set")
            plt.ylabel(
                "Final accuracy reduction vs. baseline "
                "(positive = stronger attack)"
            )
            plt.title(
                f"Injection effect by checkpoint — B={block_size:,}"
            )
            plt.xticks(rotation=35, ha="right")
            plt.grid(axis="y", alpha=0.3)
            _save_figure(
                f"08_injection_effect_by_checkpoint_B{block_size}.png"
            )

    paired = injection_seed_df.pivot_table(
        index=[
            "seed",
            "repeat",
            "small_block_size",
            "injection_epoch",
        ],
        columns="condition",
        values="accuracy_effect",
    ).reset_index()
    if {"harmful_missed", "random_unseen"}.issubset(paired):
        paired["missed_over_random_advantage"] = (
            paired["harmful_missed"] - paired["random_unseen"]
        )
        paired.to_csv(
            OUT_DIR / "injection_paired_effects.csv",
            index=False,
        )

        for block_size in SMALL_BLOCK_SIZES:
            block_part = paired[
                paired["small_block_size"] == block_size
            ]
            advantage_data = []
            advantage_labels = []
            for injection_epoch in CHECKPOINT_EPOCHS:
                values = block_part.loc[
                    block_part["injection_epoch"] == injection_epoch,
                    "missed_over_random_advantage",
                ].dropna().to_numpy()
                if values.size:
                    advantage_data.append(values)
                    advantage_labels.append(str(injection_epoch))
            if advantage_data:
                plt.figure(figsize=(9, 5))
                plt.boxplot(
                    advantage_data,
                    labels=advantage_labels,
                    showfliers=False,
                )
                for pos, values in enumerate(advantage_data, start=1):
                    plt.scatter(
                        np.full(values.shape, pos, dtype=float),
                        values,
                        s=22,
                        alpha=0.7,
                    )
                plt.axhline(0, linestyle="--")
                plt.xlabel("Injection epoch")
                plt.ylabel("Missed-edge effect minus random-control effect")
                plt.title(
                    f"Paired injection advantage by checkpoint — "
                    f"B={block_size:,}"
                )
                plt.grid(axis="y", alpha=0.3)
                _save_figure(
                    f"09_injection_advantage_by_checkpoint_B{block_size}.png"
                )

# %% [cell 28]
retention_ablation_rows = []

if RUN_RETENTION_ABLATION:
    for victim_seed in SEEDS:
        for repeat in range(REPEATS_PER_SEED):
            sampling_seed = _sampling_seed(victim_seed, repeat)
            for block_size in RETENTION_ABLATION_BLOCK_SIZES:
                for policy in RETENTION_POLICIES:
                    print(
                        f"Retention ablation: seed={victim_seed}, "
                        f"B={block_size:,}, policy={policy}"
                    )
                    result, runtime = _run_prbcd(
                        victim_seed=victim_seed,
                        block_size=block_size,
                        sampling_seed=sampling_seed,
                        retention_policy=policy,
                        checkpoint_epochs=[],
                        store_resample_edge_details=False,
                    )
                    diagnostics = _diagnostics(result)
                    retention_ablation_rows.append({
                        "seed": int(victim_seed),
                        "repeat": int(repeat),
                        "block_size": int(block_size),
                        "retention_policy": policy,
                        "final_accuracy": _final_accuracy(result),
                        "coverage_fraction": float(
                            diagnostics["metadata"]["coverage_fraction"]
                        ),
                        "n_ever_seen": int(
                            diagnostics["metadata"]["n_ever_seen"]
                        ),
                        "runtime_seconds": float(runtime),
                    })
                    del result, diagnostics
                    gc.collect()

    retention_ablation_df = pd.DataFrame(retention_ablation_rows)
    retention_ablation_df.to_csv(
        OUT_DIR / "retention_policy_ablation.csv",
        index=False,
    )
else:
    ablation_path = OUT_DIR / "retention_policy_ablation.csv"
    retention_ablation_df = (
        pd.read_csv(ablation_path)
        if ablation_path.exists()
        else pd.DataFrame()
    )

if not retention_ablation_df.empty:
    ablation_summary = (
        retention_ablation_df.groupby(
            ["block_size", "retention_policy"],
            as_index=False,
        )
        .agg(
            final_accuracy=("final_accuracy", "mean"),
            coverage_fraction=("coverage_fraction", "mean"),
        )
    )
    policies = RETENTION_POLICIES
    x = np.arange(len(RETENTION_ABLATION_BLOCK_SIZES))
    width = 0.8 / len(policies)
    plt.figure(figsize=(10, 5))
    for idx, policy in enumerate(policies):
        part = (
            ablation_summary[
                ablation_summary["retention_policy"] == policy
            ]
            .set_index("block_size")
            .reindex(RETENTION_ABLATION_BLOCK_SIZES)
        )
        plt.bar(
            x + (idx - (len(policies) - 1) / 2) * width,
            part["final_accuracy"],
            width=width,
            label=policy,
        )
    plt.xticks(
        x,
        [f"{b:,}" for b in RETENTION_ABLATION_BLOCK_SIZES],
    )
    plt.xlabel("Block size")
    plt.ylabel("Final attacked accuracy")
    plt.title("Retention-policy ablation")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _save_figure("10_retention_policy_ablation.png")

# %% [cell 30]
report = {
    "dataset": DATASET,
    "seeds": SEEDS,
    "model_name": MODEL_NAME,
    "model_label": MODEL_LABEL,
    "model_storage_type": MODEL_STORAGE_TYPE,
    "run_id": RUN_ID,
    "epsilon": MISS_EPSILON,
    "attack_budget": ATTACK_BUDGET,
    "small_block_sizes": SMALL_BLOCK_SIZES,
    "large_block_size": LARGE_BLOCK_SIZE,
    "epochs": EPOCHS,
    "fine_tune_epochs": FINE_TUNE_EPOCHS,
    "checkpoint_epochs": CHECKPOINT_EPOCHS,
    "injection_checkpoint_epochs": CHECKPOINT_EPOCHS,
    "probe_effect_mode": "one_step_full_block_replacement",
    "probe_gradient_mode": EXPECTED_PROBE_GRADIENT_MODE,
    "probe_candidates_per_backward": 1,
    "positive_delta_tolerance": POSITIVE_DELTA_TOL,
    "robust_positive_probability": ROBUST_POSITIVE_PROBABILITY,
    "robust_median_delta": ROBUST_MEDIAN_DELTA,
}
with open(OUT_DIR / "experiment_config.json", "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("Finished. Results:", OUT_DIR.resolve())
print("Core files:")
for filename in [
    "discovery_summary.csv",
    "probe_results.csv",
    "candidate_contextual_effect_summary.csv",
    "injection_summary.csv",
]:
    path = OUT_DIR / filename
    if path.exists():
        print(" -", path)
