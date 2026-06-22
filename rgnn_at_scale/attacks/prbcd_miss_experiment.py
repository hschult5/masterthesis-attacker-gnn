"""PRBCD search-space miss experiment.

This module extracts the optimization loop from ``PRBCD._attack`` into a
standalone, instrumented function.  It is intended to compare:

1. an exhaustive-search-space PRBCD reference run, and
2. a normal sampled PRBCD run with freely configurable hyperparameters.

The reference run is not a mathematical global optimum.  It removes the
random block-sampling bottleneck by putting every possible edge flip into one
fixed block.  Therefore, differences can be classified as:

* ``never_sampled``: reference edge never entered the sampled PRBCD block;
* ``dropped_before_final_block``: it entered but was removed by resampling;
* ``in_final_block_not_selected``: it survived into the final relaxed block,
  but optimization/final discretization did not choose it;
* ``selected_by_both``: both runs selected it.

For large graphs, the exhaustive block can be too large for GPU memory.  Use a
large fixed reference block instead, or raise ``max_full_edges`` knowingly.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm
from torch_sparse import SparseTensor

from rgnn_at_scale.attacks.base_attack import Attack
from rgnn_at_scale.helper import utils


InitializationMode = Literal["random", "full", "fixed"]
ResamplingMode = Literal["random", "none"]
Initializer = Callable[[Any, int, "PRBCDLoopConfig"], None]
Resampler = Callable[[Any, int, int, "PRBCDLoopConfig"], None]


@dataclass
class PRBCDLoopConfig:
    """Hyperparameters controlled by the standalone PRBCD loop.

    ``lr_factor`` is the constructor-level PRBCD value.  The runner recomputes
    the effective value exactly as PRBCD.__init__ does after applying the
    chosen block size.
    """

    epochs: int = 400
    fine_tune_epochs: int = 100
    block_size: Optional[int] = 1_000_000
    lr_factor: float = 100.0
    eps: float = 1e-7
    with_early_stopping: bool = True
    max_final_samples: int = 20
    display_step: int = 20
    do_synchronize: bool = False
    keep_heuristic: str = "WeightOnly"

    initialization: InitializationMode = "random"
    resampling: ResamplingMode = "random"
    resample_every: int = 1

    seed: int = 0
    progress: bool = True

    # Safety guard for initialization="full".
    max_full_edges: int = 5_000_000
    allow_unsafe_full_block: bool = False

    def validate(self, n_perturbations: int) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 <= self.fine_tune_epochs <= self.epochs:
            raise ValueError("fine_tune_epochs must satisfy 0 <= value <= epochs")
        if self.block_size is not None and self.block_size <= n_perturbations:
            raise ValueError("block_size must be greater than n_perturbations")
        if self.lr_factor <= 0:
            raise ValueError("lr_factor must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.max_final_samples <= 0:
            raise ValueError("max_final_samples must be positive")
        if self.resample_every <= 0:
            raise ValueError("resample_every must be positive")


@dataclass
class ReferenceEdgeTrace:
    edge_id: int
    first_seen_epoch: Optional[int] = None
    last_seen_epoch: Optional[int] = None
    entry_epochs: List[int] = field(default_factory=list)
    dropped_after_epochs: List[int] = field(default_factory=list)
    epochs_used_for_gradient: int = 0
    max_abs_gradient: float = 0.0
    max_relaxed_weight: float = 0.0
    final_relaxed_weight: float = 0.0
    in_final_relaxed_block: bool = False
    selected_final: bool = False
    currently_present: bool = False


@dataclass
class PRBCDRunResult:
    name: str
    config: PRBCDLoopConfig
    clean_loss: float
    clean_accuracy: float
    final_accuracy: float
    best_epoch: int
    best_accuracy: float
    final_edge_index: Tensor
    final_edge_weight: Tensor
    final_perturbation_ids: Tensor
    final_relaxed_search_space: Tensor
    final_relaxed_weights: Tensor
    epoch_records: List[Dict[str, Any]]
    tracked_edge_traces: Dict[int, ReferenceEdgeTrace]


@dataclass
class PRBCDComparison:
    reference: PRBCDRunResult
    candidate: PRBCDRunResult
    reference_edge_ids: List[int]
    selected_by_both: List[int]
    never_sampled: List[int]
    dropped_before_final_block: List[int]
    in_final_block_not_selected: List[int]
    candidate_only: List[int]
    edge_rows: List[Dict[str, Any]]

    @property
    def reference_recall(self) -> float:
        if not self.reference_edge_ids:
            return 1.0
        return len(self.selected_by_both) / len(self.reference_edge_ids)

    @property
    def sampling_recall(self) -> float:
        if not self.reference_edge_ids:
            return 1.0
        return 1.0 - len(self.never_sampled) / len(self.reference_edge_ids)


@dataclass
class _Snapshot:
    search_space: Tensor
    modified_edge_index: Tensor
    perturbed_edge_weight: Tensor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _snapshot(attack: Any) -> _Snapshot:
    return _Snapshot(
        search_space=attack.current_search_space.detach().cpu().clone(),
        modified_edge_index=attack.modified_edge_index.detach().cpu().clone(),
        perturbed_edge_weight=attack.perturbed_edge_weight.detach().cpu().clone(),
    )


def _restore_snapshot(attack: Any, snapshot: _Snapshot) -> None:
    attack.current_search_space = snapshot.search_space.to(attack.device)
    attack.modified_edge_index = snapshot.modified_edge_index.to(attack.device)
    attack.perturbed_edge_weight = snapshot.perturbed_edge_weight.to(attack.device)
    attack.perturbed_edge_weight.requires_grad_(True)


def _effective_block_size(attack: Any, config: PRBCDLoopConfig) -> int:
    if config.initialization == "full":
        if (
            attack.n_possible_edges > config.max_full_edges
            and not config.allow_unsafe_full_block
        ):
            raise MemoryError(
                "The exhaustive reference would contain "
                f"{attack.n_possible_edges:,} candidates, exceeding "
                f"max_full_edges={config.max_full_edges:,}. Use a smaller graph, "
                "a large fixed reference block, or set allow_unsafe_full_block=True."
            )
        return int(attack.n_possible_edges)

    if config.block_size is None:
        raise ValueError("block_size may be None only for initialization='full'")
    return int(config.block_size)


def _configure_attack(
    attack: Any,
    config: PRBCDLoopConfig,
    n_perturbations: int,
) -> int:
    config.validate(n_perturbations)
    block_size = _effective_block_size(attack, config)
    if block_size <= n_perturbations:
        raise ValueError(
            f"Effective block size {block_size} must exceed budget {n_perturbations}."
        )

    attack.block_size = block_size
    attack.epochs = int(config.epochs)
    attack.fine_tune_epochs = int(config.fine_tune_epochs)
    attack.epochs_resampling = attack.epochs - attack.fine_tune_epochs
    attack.with_early_stopping = bool(config.with_early_stopping)
    attack.eps = float(config.eps)
    attack.max_final_samples = int(config.max_final_samples)
    attack.display_step = int(config.display_step)
    attack.do_synchronize = bool(config.do_synchronize)
    attack.keep_heuristic = config.keep_heuristic

    # Recreate the constructor's block-size correction.
    attack.lr_factor = float(config.lr_factor) * max(
        math.log2(attack.n_possible_edges / block_size), 1.0
    )

    attack.attack_statistics = defaultdict(list)
    attack.tried_mask = torch.zeros(
        attack.n_possible_edges,
        device=attack.device,
        dtype=torch.bool,
    )
    return block_size


def _initialize_full_block(attack: Any) -> None:
    search_space = torch.arange(
        attack.n_possible_edges,
        device=attack.device,
        dtype=torch.long,
    )

    if attack.make_undirected:
        modified_edge_index = attack.linear_to_triu_idx(attack.n, search_space)
    else:
        modified_edge_index = attack.linear_to_full_idx(attack.n, search_space)
        non_loop = modified_edge_index[0] != modified_edge_index[1]
        search_space = search_space[non_loop]
        modified_edge_index = modified_edge_index[:, non_loop]

    attack.current_search_space = search_space
    attack.modified_edge_index = modified_edge_index
    attack.perturbed_edge_weight = torch.full(
        (search_space.numel(),),
        float(attack.eps),
        device=attack.device,
        dtype=torch.float32,
        requires_grad=True,
    )


def _initialize_fixed_block(attack: Any, fixed_search_space: Tensor) -> None:
    search_space = torch.unique(
        fixed_search_space.to(device=attack.device, dtype=torch.long),
        sorted=True,
    )
    if search_space.numel() == 0:
        raise ValueError("fixed_search_space is empty")
    if int(search_space.min()) < 0 or int(search_space.max()) >= attack.n_possible_edges:
        raise ValueError("fixed_search_space contains an out-of-range edge id")

    if attack.make_undirected:
        modified_edge_index = attack.linear_to_triu_idx(attack.n, search_space)
    else:
        modified_edge_index = attack.linear_to_full_idx(attack.n, search_space)
        non_loop = modified_edge_index[0] != modified_edge_index[1]
        search_space = search_space[non_loop]
        modified_edge_index = modified_edge_index[:, non_loop]

    attack.current_search_space = search_space
    attack.modified_edge_index = modified_edge_index
    attack.perturbed_edge_weight = torch.full(
        (search_space.numel(),),
        float(attack.eps),
        device=attack.device,
        dtype=torch.float32,
        requires_grad=True,
    )


def _initialize_attack(
    attack: Any,
    config: PRBCDLoopConfig,
    n_perturbations: int,
    fixed_search_space: Optional[Tensor],
    initializer: Optional[Initializer],
) -> None:
    if initializer is not None:
        initializer(attack, n_perturbations, config)
    elif config.initialization == "random":
        attack.sample_random_block(n_perturbations)
    elif config.initialization == "full":
        _initialize_full_block(attack)
    elif config.initialization == "fixed":
        if fixed_search_space is None:
            raise ValueError(
                "fixed_search_space is required for initialization='fixed'"
            )
        _initialize_fixed_block(attack, fixed_search_space)
    else:
        raise ValueError(f"Unknown initialization mode: {config.initialization}")

    if attack.current_search_space.numel() <= n_perturbations:
        raise RuntimeError(
            "Initialized search space must contain more candidates than the budget."
        )


def _resample_attack(
    attack: Any,
    epoch: int,
    n_perturbations: int,
    config: PRBCDLoopConfig,
    resampler: Optional[Resampler],
) -> None:
    if resampler is not None:
        resampler(attack, epoch, n_perturbations, config)
    elif config.resampling == "random":
        attack.resample_random_block(n_perturbations)
    elif config.resampling == "none":
        return
    else:
        raise ValueError(f"Unknown resampling mode: {config.resampling}")


def _locate_ids(sorted_space: Tensor, ids: Tensor) -> Tuple[Tensor, Tensor]:
    """Return membership and positions of ids in a sorted 1-D search space."""
    if ids.numel() == 0:
        empty_bool = torch.empty(0, dtype=torch.bool, device=ids.device)
        empty_long = torch.empty(0, dtype=torch.long, device=ids.device)
        return empty_bool, empty_long
    if sorted_space.numel() == 0:
        return (
            torch.zeros(ids.numel(), dtype=torch.bool, device=ids.device),
            torch.zeros(ids.numel(), dtype=torch.long, device=ids.device),
        )

    positions = torch.searchsorted(sorted_space, ids)
    valid = positions < sorted_space.numel()
    clamped = positions.clamp(max=sorted_space.numel() - 1)
    present = valid & (sorted_space[clamped] == ids)
    return present, clamped


def _update_membership_traces(
    attack: Any,
    tracked_ids: Tensor,
    traces: Dict[int, ReferenceEdgeTrace],
    epoch_marker: int,
    dropped_after_epoch: Optional[int],
) -> Tuple[int, int]:
    if tracked_ids.numel() == 0:
        return 0, 0

    present, _ = _locate_ids(attack.current_search_space, tracked_ids)
    present_cpu = present.detach().cpu().tolist()
    ids_cpu = tracked_ids.detach().cpu().tolist()

    entries = 0
    drops = 0
    for edge_id, is_present in zip(ids_cpu, present_cpu):
        trace = traces[int(edge_id)]
        if is_present and not trace.currently_present:
            trace.entry_epochs.append(int(epoch_marker))
            if trace.first_seen_epoch is None:
                trace.first_seen_epoch = int(epoch_marker)
            entries += 1
        elif not is_present and trace.currently_present:
            if dropped_after_epoch is not None:
                trace.dropped_after_epochs.append(int(dropped_after_epoch))
            drops += 1

        if is_present:
            trace.last_seen_epoch = int(epoch_marker)
        trace.currently_present = bool(is_present)

    return entries, drops


def _record_tracked_values(
    attack: Any,
    tracked_ids: Tensor,
    traces: Dict[int, ReferenceEdgeTrace],
    gradient: Tensor,
) -> int:
    if tracked_ids.numel() == 0:
        return 0

    present, positions = _locate_ids(attack.current_search_space, tracked_ids)
    tracked_positions = positions[present]
    if tracked_positions.numel() == 0:
        return 0

    present_ids = tracked_ids[present].detach().cpu().tolist()
    abs_grad = gradient.detach().abs()[tracked_positions].cpu().tolist()
    weights = (
        attack.perturbed_edge_weight.detach()[tracked_positions].cpu().tolist()
    )

    for edge_id, grad_value, weight_value in zip(
        present_ids, abs_grad, weights
    ):
        trace = traces[int(edge_id)]
        trace.epochs_used_for_gradient += 1
        trace.max_abs_gradient = max(trace.max_abs_gradient, float(grad_value))
        trace.max_relaxed_weight = max(
            trace.max_relaxed_weight, float(weight_value)
        )

    return len(present_ids)


def _evaluate_accuracy(attack: Any) -> float:
    edge_index, edge_weight = attack.get_modified_adj()
    logits = attack._get_logits(attack.attr, edge_index, edge_weight)
    return _to_float(utils.accuracy(logits, attack.labels, attack.idx_attack))


def run_prbcd_loop(
    attack: Any,
    n_perturbations: int,
    config: PRBCDLoopConfig,
    *,
    name: str = "run",
    tracked_edge_ids: Optional[Sequence[int] | Tensor] = None,
    fixed_search_space: Optional[Tensor] = None,
    initializer: Optional[Initializer] = None,
    resampler: Optional[Resampler] = None,
) -> PRBCDRunResult:
    """Run an instrumented PRBCD optimization loop on an initialized attack.

    The ``attack`` object must already contain the victim model, graph tensors,
    labels, attack indices, device, and PRBCD helper methods.  This function
    mutates that attack object, exactly like ``PRBCD._attack`` does.

    Custom selector-based initialization/resampling can be supplied through
    callbacks without putting experiment logic back into the PRBCD class.
    """
    _set_seed(config.seed)
    _configure_attack(attack, config, n_perturbations)

    attack.semi = False
    attack.use_cert = "standalone_experiment"
    attack.seed = config.seed

    _initialize_attack(
        attack,
        config,
        n_perturbations,
        fixed_search_space,
        initializer,
    )

    if tracked_edge_ids is None:
        tracked_ids = torch.empty(0, dtype=torch.long, device=attack.device)
    else:
        tracked_ids = torch.as_tensor(
            tracked_edge_ids,
            dtype=torch.long,
            device=attack.device,
        )
        tracked_ids = torch.unique(tracked_ids, sorted=True)
        if tracked_ids.numel() > 0:
            if int(tracked_ids.min()) < 0 or int(tracked_ids.max()) >= attack.n_possible_edges:
                raise ValueError("tracked_edge_ids contains an out-of-range edge id")

    traces = {
        int(edge_id): ReferenceEdgeTrace(edge_id=int(edge_id))
        for edge_id in tracked_ids.detach().cpu().tolist()
    }
    _update_membership_traces(
        attack,
        tracked_ids,
        traces,
        epoch_marker=0,
        dropped_after_epoch=None,
    )

    with torch.no_grad():
        clean_logits = attack._get_logits(
            attack.attr,
            attack.edge_index,
            attack.edge_weight,
        )
        clean_loss_tensor = attack.calculate_loss(
            clean_logits[attack.idx_attack],
            attack.labels[attack.idx_attack],
        )
        clean_accuracy = _to_float(
            utils.accuracy(clean_logits, attack.labels, attack.idx_attack)
        )
        clean_loss = _to_float(clean_loss_tensor)
        del clean_logits, clean_loss_tensor

    best_accuracy = float("inf")
    best_epoch = -1
    best_snapshot: Optional[_Snapshot] = None
    epoch_records: List[Dict[str, Any]] = []

    iterator: Iterable[int] = range(config.epochs)
    if config.progress:
        iterator = tqdm(iterator, desc=name)

    for epoch in iterator:
        attack.perturbed_edge_weight = attack.perturbed_edge_weight.detach()
        attack.perturbed_edge_weight.requires_grad_(True)

        tracked_before, _ = _locate_ids(
            attack.current_search_space,
            tracked_ids,
        )
        tracked_before_count = int(tracked_before.sum().item())

        edge_index, edge_weight = attack.get_modified_adj()

        if torch.cuda.is_available() and config.do_synchronize:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logits = attack._get_logits(attack.attr, edge_index, edge_weight)
        loss = attack.calculate_loss(
            logits[attack.idx_attack],
            attack.labels[attack.idx_attack],
        )
        gradient = utils.grad_with_checkpoint(
            loss,
            attack.perturbed_edge_weight,
        )[0]
        attack.gradient = gradient

        if torch.cuda.is_available() and config.do_synchronize:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with torch.no_grad():
            attack.update_edge_weights(n_perturbations, epoch, gradient)
            probability_mass_update = _to_float(
                attack.perturbed_edge_weight.sum()
            )

            attack.perturbed_edge_weight = Attack.project(
                n_perturbations,
                attack.perturbed_edge_weight,
                attack.eps,
            )
            probability_mass_projected = _to_float(
                attack.perturbed_edge_weight.sum()
            )

            _record_tracked_values(
                attack,
                tracked_ids,
                traces,
                gradient,
            )

            accuracy = _evaluate_accuracy(attack)
            loss_value = _to_float(loss)

            if epoch % max(1, config.display_step) == 0:
                logging.info(
                    "[%s] epoch=%d loss=%.6f accuracy=%.6f",
                    name,
                    epoch,
                    loss_value,
                    accuracy,
                )

            if config.with_early_stopping and accuracy < best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                best_snapshot = _snapshot(attack)

            if hasattr(attack, "_append_attack_statistics"):
                attack._append_attack_statistics(
                    loss_value,
                    accuracy,
                    probability_mass_update,
                    probability_mass_projected,
                )

            resampled = False
            restored_for_fine_tuning = False
            should_resample = (
                (resampler is not None or config.resampling != "none")
                and epoch < attack.epochs_resampling - 1
                and epoch % config.resample_every == 0
            )

            if should_resample:
                _resample_attack(
                    attack,
                    epoch,
                    n_perturbations,
                    config,
                    resampler,
                )
                resampled = True
            elif (
                config.with_early_stopping
                and epoch == attack.epochs_resampling - 1
                and best_snapshot is not None
            ):
                _restore_snapshot(attack, best_snapshot)
                restored_for_fine_tuning = True

            entries, drops = _update_membership_traces(
                attack,
                tracked_ids,
                traces,
                epoch_marker=epoch + 1,
                dropped_after_epoch=epoch,
            )
            tracked_after, _ = _locate_ids(
                attack.current_search_space,
                tracked_ids,
            )
            tracked_after_count = int(tracked_after.sum().item())

            epoch_records.append(
                {
                    "epoch": epoch,
                    "loss": loss_value,
                    "accuracy": accuracy,
                    "block_size": int(attack.current_search_space.numel()),
                    "nonzero_weights": int(
                        (attack.perturbed_edge_weight > attack.eps).sum().item()
                    ),
                    "probability_mass_update": probability_mass_update,
                    "probability_mass_projected": probability_mass_projected,
                    "tracked_total": int(tracked_ids.numel()),
                    "tracked_in_block_before": tracked_before_count,
                    "tracked_in_block_after": tracked_after_count,
                    "tracked_recall_before": (
                        tracked_before_count / tracked_ids.numel()
                        if tracked_ids.numel() > 0
                        else 1.0
                    ),
                    "tracked_recall_after": (
                        tracked_after_count / tracked_ids.numel()
                        if tracked_ids.numel() > 0
                        else 1.0
                    ),
                    "tracked_entries": entries,
                    "tracked_drops": drops,
                    "resampled": resampled,
                    "restored_for_fine_tuning": restored_for_fine_tuning,
                }
            )

        del edge_index, edge_weight, logits, loss, gradient

    if config.with_early_stopping and best_snapshot is not None:
        _restore_snapshot(attack, best_snapshot)
        _update_membership_traces(
            attack,
            tracked_ids,
            traces,
            epoch_marker=config.epochs,
            dropped_after_epoch=config.epochs - 1,
        )
    elif best_epoch < 0:
        best_epoch = config.epochs - 1
        best_accuracy = epoch_records[-1]["accuracy"]

    final_relaxed_search_space = attack.current_search_space.detach().cpu().clone()
    final_relaxed_weights = attack.perturbed_edge_weight.detach().cpu().clone()

    if tracked_ids.numel() > 0:
        present, positions = _locate_ids(
            attack.current_search_space,
            tracked_ids,
        )
        ids_cpu = tracked_ids.detach().cpu().tolist()
        present_cpu = present.detach().cpu().tolist()
        for idx, edge_id in enumerate(ids_cpu):
            trace = traces[int(edge_id)]
            trace.in_final_relaxed_block = bool(present_cpu[idx])
            if present_cpu[idx]:
                weight = float(
                    attack.perturbed_edge_weight[positions[idx]].detach().cpu().item()
                )
                trace.final_relaxed_weight = weight
                trace.max_relaxed_weight = max(trace.max_relaxed_weight, weight)

    final_edge_index, final_edge_weight = attack.sample_final_edges(
        n_perturbations
    )
    final_perturbation_ids = attack.current_search_space[
        attack.perturbed_edge_weight > 0.5
    ].detach().cpu().clone()

    final_selected_set = set(final_perturbation_ids.tolist())
    for edge_id, trace in traces.items():
        trace.selected_final = edge_id in final_selected_set

    attack.adj_adversary = SparseTensor.from_edge_index(
        final_edge_index,
        torch.ones_like(final_edge_index[0], dtype=torch.float32),
        (attack.n, attack.n),
    ).coalesce().detach()
    attack.attr_adversary = attack.attr

    with torch.no_grad():
        final_logits = attack._get_logits(
            attack.attr,
            final_edge_index,
            final_edge_weight,
        )
        final_accuracy = _to_float(
            utils.accuracy(final_logits, attack.labels, attack.idx_attack)
        )

    return PRBCDRunResult(
        name=name,
        config=config,
        clean_loss=clean_loss,
        clean_accuracy=clean_accuracy,
        final_accuracy=final_accuracy,
        best_epoch=best_epoch,
        best_accuracy=float(best_accuracy),
        final_edge_index=final_edge_index.detach().cpu(),
        final_edge_weight=final_edge_weight.detach().cpu(),
        final_perturbation_ids=final_perturbation_ids,
        final_relaxed_search_space=final_relaxed_search_space,
        final_relaxed_weights=final_relaxed_weights,
        epoch_records=epoch_records,
        tracked_edge_traces=traces,
    )


def _decode_edge_ids(attack: Any, edge_ids: Sequence[int]) -> Dict[int, Tuple[int, int]]:
    if not edge_ids:
        return {}
    ids = torch.tensor(edge_ids, device=attack.device, dtype=torch.long)
    if attack.make_undirected:
        pairs = attack.linear_to_triu_idx(attack.n, ids)
    else:
        pairs = attack.linear_to_full_idx(attack.n, ids)
    pairs = pairs.detach().cpu()
    return {
        int(edge_id): (int(pairs[0, i]), int(pairs[1, i]))
        for i, edge_id in enumerate(edge_ids)
    }


def _clean_edge_set(attack: Any) -> set[Tuple[int, int]]:
    edge_index = attack.edge_index.detach().cpu()
    result: set[Tuple[int, int]] = set()
    for u, v in edge_index.t().tolist():
        if u == v:
            continue
        if attack.make_undirected:
            result.add((min(int(u), int(v)), max(int(u), int(v))))
        else:
            result.add((int(u), int(v)))
    return result


def compare_prbcd_runs(
    reference_attack: Any,
    candidate_attack: Any,
    n_perturbations: int,
    *,
    reference_config: Optional[PRBCDLoopConfig] = None,
    candidate_config: Optional[PRBCDLoopConfig] = None,
    reference_fixed_search_space: Optional[Tensor] = None,
    candidate_fixed_search_space: Optional[Tensor] = None,
    reference_initializer: Optional[Initializer] = None,
    reference_resampler: Optional[Resampler] = None,
    candidate_initializer: Optional[Initializer] = None,
    candidate_resampler: Optional[Resampler] = None,
    output_dir: Optional[str | Path] = None,
) -> PRBCDComparison:
    """Run the reference first, then trace its selected edges in the candidate.

    Use two freshly initialized PRBCD objects with identical victim weights,
    data, labels, and attack indices.  The objects are intentionally supplied
    separately because deepcopying CUDA models/attack state is fragile.
    """
    if reference_attack.n != candidate_attack.n:
        raise ValueError("Reference and candidate attacks use different graphs")
    if reference_attack.make_undirected != candidate_attack.make_undirected:
        raise ValueError("Reference and candidate disagree on directedness")

    if reference_config is None:
        reference_config = PRBCDLoopConfig(
            epochs=400,
            fine_tune_epochs=100,
            block_size=None,
            initialization="full",
            resampling="none",
            seed=0,
        )
    if candidate_config is None:
        candidate_config = PRBCDLoopConfig(seed=0)

    reference = run_prbcd_loop(
        reference_attack,
        n_perturbations,
        reference_config,
        name="reference",
        fixed_search_space=reference_fixed_search_space,
        initializer=reference_initializer,
        resampler=reference_resampler,
    )

    reference_ids = sorted(
        int(x) for x in reference.final_perturbation_ids.tolist()
    )

    candidate = run_prbcd_loop(
        candidate_attack,
        n_perturbations,
        candidate_config,
        name="candidate",
        tracked_edge_ids=reference_ids,
        fixed_search_space=candidate_fixed_search_space,
        initializer=candidate_initializer,
        resampler=candidate_resampler,
    )

    reference_set = set(reference_ids)
    candidate_set = set(int(x) for x in candidate.final_perturbation_ids.tolist())
    candidate_final_block = set(
        int(x) for x in candidate.final_relaxed_search_space.tolist()
    )

    selected_by_both: List[int] = []
    never_sampled: List[int] = []
    dropped_before_final_block: List[int] = []
    in_final_block_not_selected: List[int] = []

    for edge_id in reference_ids:
        trace = candidate.tracked_edge_traces[edge_id]
        if edge_id in candidate_set:
            selected_by_both.append(edge_id)
        elif trace.first_seen_epoch is None:
            never_sampled.append(edge_id)
        elif edge_id not in candidate_final_block:
            dropped_before_final_block.append(edge_id)
        else:
            in_final_block_not_selected.append(edge_id)

    candidate_only = sorted(candidate_set - reference_set)

    pair_map = _decode_edge_ids(candidate_attack, reference_ids)
    clean_edges = _clean_edge_set(candidate_attack)
    edge_rows: List[Dict[str, Any]] = []

    for edge_id in reference_ids:
        u, v = pair_map[edge_id]
        canonical = (min(u, v), max(u, v)) if candidate_attack.make_undirected else (u, v)
        action = "delete" if canonical in clean_edges else "add"
        trace = candidate.tracked_edge_traces[edge_id]

        if edge_id in selected_by_both:
            category = "selected_by_both"
        elif edge_id in never_sampled:
            category = "never_sampled"
        elif edge_id in dropped_before_final_block:
            category = "dropped_before_final_block"
        else:
            category = "in_final_block_not_selected"

        row = asdict(trace)
        row.update(
            {
                "u": u,
                "v": v,
                "action": action,
                "category": category,
            }
        )
        edge_rows.append(row)

    comparison = PRBCDComparison(
        reference=reference,
        candidate=candidate,
        reference_edge_ids=reference_ids,
        selected_by_both=selected_by_both,
        never_sampled=never_sampled,
        dropped_before_final_block=dropped_before_final_block,
        in_final_block_not_selected=in_final_block_not_selected,
        candidate_only=candidate_only,
        edge_rows=edge_rows,
    )

    if output_dir is not None:
        save_comparison(comparison, output_dir)

    return comparison


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def save_comparison(
    comparison: PRBCDComparison,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "reference_config": asdict(comparison.reference.config),
        "candidate_config": asdict(comparison.candidate.config),
        "n_reference_edges": len(comparison.reference_edge_ids),
        "selected_by_both": comparison.selected_by_both,
        "never_sampled": comparison.never_sampled,
        "dropped_before_final_block": comparison.dropped_before_final_block,
        "in_final_block_not_selected": comparison.in_final_block_not_selected,
        "candidate_only": comparison.candidate_only,
        "reference_recall": comparison.reference_recall,
        "sampling_recall": comparison.sampling_recall,
        "reference_clean_accuracy": comparison.reference.clean_accuracy,
        "reference_final_accuracy": comparison.reference.final_accuracy,
        "candidate_clean_accuracy": comparison.candidate.clean_accuracy,
        "candidate_final_accuracy": comparison.candidate.final_accuracy,
        "reference_best_epoch": comparison.reference.best_epoch,
        "candidate_best_epoch": comparison.candidate.best_epoch,
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    _write_csv(
        output_path / "reference_epochs.csv",
        comparison.reference.epoch_records,
    )
    _write_csv(
        output_path / "candidate_epochs.csv",
        comparison.candidate.epoch_records,
    )
    _write_csv(
        output_path / "reference_edge_trace_in_candidate.csv",
        comparison.edge_rows,
    )


def concise_report(comparison: PRBCDComparison) -> str:
    """Return a compact human-readable experiment summary."""
    total = len(comparison.reference_edge_ids)
    return (
        f"Reference edges: {total}\n"
        f"Selected by both: {len(comparison.selected_by_both)}\n"
        f"Never sampled: {len(comparison.never_sampled)}\n"
        "Dropped before final relaxed block: "
        f"{len(comparison.dropped_before_final_block)}\n"
        "In final block but not discretely selected: "
        f"{len(comparison.in_final_block_not_selected)}\n"
        f"Reference-edge recall: {comparison.reference_recall:.3f}\n"
        f"Sampling recall: {comparison.sampling_recall:.3f}\n"
        "Final accuracy (reference/candidate): "
        f"{comparison.reference.final_accuracy:.6f} / "
        f"{comparison.candidate.final_accuracy:.6f}"
    )

def graph_to_attack_inputs(
    graph: Any,
    data_device: str | torch.device = "cpu",
) -> Tuple[SparseTensor, Tensor, Tensor]:
    """Convert graph.adj_matrix, graph.attr_matrix and graph.labels."""

    device = torch.device(data_device)

    # Adjacency: scipy sparse matrix -> torch_sparse.SparseTensor
    adj_coo = graph.adj_matrix.tocoo()

    adj = SparseTensor(
        row=torch.as_tensor(
            adj_coo.row,
            dtype=torch.long,
            device=device,
        ),
        col=torch.as_tensor(
            adj_coo.col,
            dtype=torch.long,
            device=device,
        ),
        value=torch.as_tensor(
            adj_coo.data,
            dtype=torch.float32,
            device=device,
        ),
        sparse_sizes=adj_coo.shape,
    ).coalesce()

    # Attributes: scipy sparse or NumPy -> dense float tensor
    attr_source = graph.attr_matrix

    if hasattr(attr_source, "toarray"):
        attr_source = attr_source.toarray()

    attr = torch.as_tensor(
        np.asarray(attr_source),
        dtype=torch.float32,
        device=device,
    )

    # Labels -> long tensor
    labels = torch.as_tensor(
        np.asarray(graph.labels),
        dtype=torch.long,
        device=device,
    )

    # Handle labels shaped as (N, 1) or one-hot labels shaped as (N, C)
    if labels.ndim == 2:
        if labels.size(1) == 1:
            labels = labels.squeeze(1)
        else:
            labels = labels.argmax(dim=1)

    labels = labels.reshape(-1)

    if attr.size(0) != labels.numel():
        raise ValueError(
            f"Feature rows ({attr.size(0)}) do not match "
            f"number of labels ({labels.numel()})."
        )

    if adj.sparse_size(0) != attr.size(0):
        raise ValueError(
            f"Adjacency has {adj.sparse_size(0)} nodes, "
            f"but attributes have {attr.size(0)} rows."
        )

    return adj, attr, labels