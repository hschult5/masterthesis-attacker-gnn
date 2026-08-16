import logging
import math

from collections import defaultdict
from typing import List, Tuple, Optional, Dict, DefaultDict, Any

from tqdm import tqdm

import numpy as np
import torch
import torch_sparse

from torch_sparse import SparseTensor

from rgnn_at_scale.helper import utils
from rgnn_at_scale.attacks.base_attack import (
    Attack,
    SparseAttack,
)

class PRBCD(SparseAttack):
    """Sampled and hence scalable PGD attack for graph data.
    """

    def __init__(
            self,
            keep_heuristic: str = "WeightOnly",
            lr_factor: float = 100,
            display_step: int = 20,
            epochs: int = 400,
            fine_tune_epochs: int = 100,
            block_size: int = 1_000_000,
            with_early_stopping: bool = True,
            do_synchronize: bool = False,
            eps: float = 1e-7,
            max_final_samples: int = 20,
            lp_model: Optional[torch.nn.Module] = None,

            # Fixed blocks from RQ2
            initial_block_linear_ids: Optional[torch.Tensor] = None,
            initial_block_label: str = "",
            resampling_enabled: bool = True,
            block_diagnostics_enabled: bool = False,
            attack_sampling_seed: Optional[int] = None,

            # RQ1
            rq1_enabled: bool = False,
            rq1_is_reference: bool = False,
            rq1_sampling_seed: Optional[int] = None,

            **kwargs,
    ):
        super().__init__(**kwargs)

        # Existing initialization
        self.lp_model = lp_model

        # Initial Block from RQ2 initial block Experiment
        self.initial_block_linear_ids = (
            initial_block_linear_ids
            if initial_block_linear_ids is not None
            else None
        )
        self.initial_block_label = str(initial_block_label)

        # indicates whether resampling is enabled
        self.resampling_enabled = resampling_enabled
        self.block_diagnostics_enabled = bool(block_diagnostics_enabled)
        self.attack_sampling_seed = (
            int(attack_sampling_seed)
            if attack_sampling_seed is not None
            else None
        )

        # RQ1 configuration
        self.rq1_enabled = bool(rq1_enabled)
        self.rq1_is_reference = bool(rq1_is_reference)
        self.rq1_sampling_seed = (
            int(rq1_sampling_seed)
            if rq1_sampling_seed is not None
            else 0
        )

        self.keep_heuristic = keep_heuristic
        self.display_step = display_step
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.epochs_resampling = epochs - fine_tune_epochs
        self.block_size = block_size
        self.with_early_stopping = with_early_stopping
        self.eps = eps
        self.do_synchronize = do_synchronize
        self.max_final_samples = max_final_samples
        self.device = kwargs.get('device')

        self.current_search_space: torch.Tensor = None
        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        self.semi = None

        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        # RQ1 reference runs use the complete edge-flip space as one fixed block.
        # This makes all epochs behave like fine-tuning epochs: no block sampling
        # or resampling is performed, and the learning-rate schedule starts in
        # the fine-tuning regime immediately.
        if self.rq1_is_reference:
            self.block_size = int(self.n_possible_edges)
            self.fine_tune_epochs = int(self.epochs)
            self.epochs_resampling = 0

        self.lr_factor = lr_factor * max(
            math.log2(self.n_possible_edges / self.block_size),
            1.,
        )

    def _attack(self, ads_mode, graph, n_perturbations, semi=False, use_cert="none", grid_radii: Optional[np.ndarray] = None, grid_binary_class: Optional[np.ndarray] = None, **kwargs):
        """Perform attack (`n_perturbations` is increasing as it was a greedy attack).

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        self.semi = semi
        self.use_cert = use_cert
        self.dataset = kwargs.get('dataset')
        self.seed = kwargs.get('seed')

        if self.attack_sampling_seed is not None:
            attack_sampling_seed = int(self.attack_sampling_seed)
        elif self.rq1_enabled or self.rq1_is_reference:
            attack_sampling_seed = int(self.rq1_sampling_seed)
        else:
            attack_sampling_seed = int(self.seed or 0)

        # Seed before constructing either a random block or any later refill.
        if (
            self.attack_sampling_seed is not None
            or self.rq1_enabled
            or self.rq1_is_reference
            or self.block_diagnostics_enabled
        ):
            torch.manual_seed(attack_sampling_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(attack_sampling_seed)

        selector_params = kwargs.get("selector_params")

        # Load selector/resampling configuration before initial-block selection.
        # This also makes selector settings available for custom initial blocks.
        if use_cert in ("accuracy_drop_selector","accuracy_drop_selector_with_resampling"):
            self._load_selector_params(selector_params)

        # For early stopping (not explicitly covered by pesudo code)
        best_accuracy = float('Inf')
        best_epoch = float('-Inf')

        # For collecting attack statistics
        self.attack_statistics: DefaultDict[str, Any] = defaultdict(list)

        if self.rq1_enabled:
            self.attack_statistics["rq1"] = {
                "final_linear_ids": torch.empty(
                    0,
                    dtype=torch.long,
                ),
                "initial_block": None,
                "epoch_blocks": {},
                "resample_events": [],
                "positive_gradient_sum": None,
                "max_weight": None,
                "times_seen": None,
                "metadata": {
                    "block_size": int(self.block_size),
                    "n_perturbations": int(n_perturbations),
                    "sampling_seed": int(self.rq1_sampling_seed),
                    "is_reference": bool(self.rq1_is_reference),
                    "use_cert": str(use_cert),
                    "full_space_reference": bool(self.rq1_is_reference),
                    "n_possible_edges": int(self.n_possible_edges),
                    "epochs_resampling": int(self.epochs_resampling),
                },
            }

            if self.rq1_is_reference:
                rq1 = self.attack_statistics["rq1"]

                rq1["positive_gradient_sum"] = torch.zeros(
                    self.n_possible_edges,
                    dtype=torch.float32,
                )

                rq1["max_weight"] = torch.zeros(
                    self.n_possible_edges,
                    dtype=torch.float32,
                )

                rq1["times_seen"] = torch.zeros(
                    self.n_possible_edges,
                    dtype=torch.int32,
                )

        if self.block_diagnostics_enabled:
            self.attack_statistics["block_diagnostics"] = {
                "initial_block": None,
                "epoch_blocks": {},
                "resample_events": [],
                "final_block": None,
                "final_linear_ids": torch.empty(0, dtype=torch.long),
                "metadata": {
                    "block_size": int(self.block_size),
                    "n_perturbations": int(n_perturbations),
                    "sampling_seed": int(attack_sampling_seed),
                    "initial_block_label": self.initial_block_label,
                    "custom_initial_block": bool(self.initial_block_linear_ids is not None),
                    "resampling_enabled": bool(self.resampling_enabled),
                    "epochs": int(self.epochs),
                    "fine_tune_epochs": int(self.fine_tune_epochs),
                    "epochs_resampling": int(self.epochs_resampling),
                    "n_possible_edges": int(self.n_possible_edges),
                },
            }

        #tried_mask for selector exclusion
        self.tried_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)

        # Sample initial search space (Algorithm 1, line 3-4).
        # Supplied Block takes prescedent over sampling
        if self.initial_block_linear_ids is not None:
            self.init_search_space_from_linear_ids(self.initial_block_linear_ids)

        # RQ1 reference: use the complete edge-flip space as a fixed block.
        elif self.rq1_is_reference:
            print(
                "[RQ1] reference run -> using full edge-flip search space "
                "and disabling resampling"
            )
            self.sample_full_search_space(
                n_perturbations=n_perturbations,
            )

        elif use_cert in ("accuracy_drop_selector","accuracy_drop_selector_with_resampling"):
            if self.lp_model is None:
                raise ValueError(
                    f"use_cert='{use_cert}' requires a supplied lp_model."
                )
            print(use_cert,"-> selector guided initial block")
            self.lp_model = self.lp_model.to(self.device)
            self.lp_model.eval()
            self.sample_block_from_linkpred_threshold(
                graph=graph,
                tau=self.tau,
                score_batch_size=self.score_batch_size,
                max_sampling_tries=self.max_sampling_tries,
                rng_seed=attack_sampling_seed,
                exclude_tried=self.exclude_tried,
            )
        elif use_cert == "none":
            print("Standard PRBCD -> random initial block")
            self.sample_random_block(n_perturbations,self.block_size)
        else:
            raise ValueError(f"Unknown use_cert mode: {use_cert!r}"
                             )
        # Accuracy and attack statistics before the attack even started
        with torch.no_grad():

            logits = self._get_logits(self.attr, self.edge_index, self.edge_weight)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

            logging.info(f'\nBefore the attack - Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

            self._append_attack_statistics(
                loss=loss.item(),
                accuracy=accuracy,
                probability_mass_update=0.0,
                probability_mass_projected=0.0,
                epoch=None,
                gradient=None,
            )

            del logits, loss

        # Loop over the epochs (Algorithm 1, line 5)
        for epoch in tqdm(range(self.epochs)):
            self.perturbed_edge_weight.requires_grad = True

            # Retreive sparse perturbed adjacency matrix `A \oplus p_{t-1}` (Algorithm 1, line 6)
            edge_index, edge_weight = self.get_modified_adj()

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Calculate logits for each node (Algorithm 1, line 6)
            logits = self._get_logits(self.attr, edge_index, edge_weight)
            # Calculate loss combining all each node (Algorithm 1, line 7)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            # Retreive gradient towards the current block (Algorithm 1, line 7)
            gradient = utils.grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
            self.gradient = gradient

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                # Gradient update step (Algorithm 1, line 7)
                edge_weight = self.update_edge_weights(n_perturbations, epoch, gradient)[1]
                # For monitoring
                probability_mass_update = self.perturbed_edge_weight.sum().item()
                # Projection to stay within relaxed `L_0` budget (Algorithm 1, line 8)
                self.perturbed_edge_weight = Attack.project(
                    n_perturbations, self.perturbed_edge_weight, self.eps)
                # For monitoring
                probability_mass_projected = self.perturbed_edge_weight.sum().item()

                # Calculate accuracy after the current epoch (overhead for monitoring and early stopping)
                edge_index, edge_weight = self.get_modified_adj()
                logits = self.attacked_model(data=self.attr.to(self.device), adj=(edge_index, edge_weight))
                accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

                del edge_index, edge_weight, logits

                if epoch % self.display_step == 0:
                    logging.info(f'\nEpoch: {epoch} Loss: {loss} Accuracy: {100 * accuracy:.3f} %\n')

                # Save best epoch for early stopping (not explicitly covered by pesudo code)
                if self.with_early_stopping and best_accuracy > accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
                    best_search_space = self.current_search_space.clone().cpu()
                    best_edge_index = self.modified_edge_index.clone().cpu()
                    best_edge_weight_diff = self.perturbed_edge_weight.detach().clone().cpu()

                self._append_attack_statistics(
                    loss=loss,
                    accuracy=accuracy,
                    probability_mass_update=probability_mass_update,
                    probability_mass_projected=probability_mass_projected,
                    epoch=epoch,
                    gradient=gradient,
                )

                # Skip resampling for RQ1 reference runs or when explicitly disabled
                if self.rq1_is_reference or not self.resampling_enabled:
                    pass

                # Resampling of search space (Algorithm 1, line 9-14)
                elif epoch < self.epochs_resampling - 1:

                    if self.rq1_enabled and not self.rq1_is_reference:
                        rq1_before_resampling = (
                            self.current_search_space
                            .detach()
                            .cpu()
                            .long()
                            .clone()
                        )

                    if self.block_diagnostics_enabled:
                        diagnostic_before_resampling = (
                            self.current_search_space
                            .detach()
                            .cpu()
                            .long()
                            .clone()
                        )

                    if use_cert in ("accuracy_drop_selector","accuracy_drop_selector_with_resampling"):
                        if use_cert == "accuracy_drop_selector_with_resampling":
                            # Thesis resampling function uses the selector
                            self.resample_block_from_linkpred_topk(
                                graph=graph,
                                top_k_per_batch=self.top_k_per_batch,
                                score_batch_size=self.score_batch_size,
                                exclude_tried=self.exclude_tried,
                                rng_seed=(attack_sampling_seed + epoch + 1)
                            )
                        else:
                            # Resample Random block if only initial block is supposed to be sampled from selector
                            self.resample_random_block(
                                n_perturbations=n_perturbations,
                                mod_block_size=self.block_size,
                            )
                    else:
                        # Original PR-BCD resampling function
                        self.resample_random_block(
                            n_perturbations=n_perturbations,
                            mod_block_size=self.block_size,
                        )

                    # ==========================================================
                    # Record the post-resampling block
                    # ==========================================================

                    if self.rq1_enabled and not self.rq1_is_reference:
                        rq1_after_resampling = (
                            self.current_search_space
                            .detach()
                            .cpu()
                            .long()
                            .clone()
                        )

                        self.attack_statistics["rq1"][
                            "resample_events"
                        ].append({
                            "epoch": int(epoch),
                            "before": rq1_before_resampling,
                            "after": rq1_after_resampling,
                        })

                    if self.block_diagnostics_enabled:
                        diagnostic_after_resampling = (
                            self.current_search_space
                            .detach()
                            .cpu()
                            .long()
                            .clone()
                        )
                        self.attack_statistics["block_diagnostics"][
                            "resample_events"
                        ].append({
                            "epoch": int(epoch),
                            "before": diagnostic_before_resampling,
                            "after": diagnostic_after_resampling,
                        })
                elif self.with_early_stopping and epoch == self.epochs_resampling - 1:
                    # Retreive best epoch if early stopping is active (not explicitly covered by pesudo code)
                    logging.info(
                        f'Loading search space of epoch {best_epoch} (accuarcy={best_accuracy}) for fine tuning\n')
                    self.current_search_space = best_search_space.to(self.device)
                    self.modified_edge_index = best_edge_index.to(self.device)
                    self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)
                    self.perturbed_edge_weight.requires_grad = True

        # Retreive best epoch if early stopping is active (not explicitly covered by pesudo code)
        if self.with_early_stopping:
            self.current_search_space = best_search_space.to(self.device)
            self.modified_edge_index = best_edge_index.to(self.device)
            self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)

        # Sample final discrete graph (Algorithm 1, line 16)
        edge_index = self.sample_final_edges(n_perturbations)[0]

        if self.rq1_enabled:
            final_space = (
                self.current_search_space
                .detach()
                .cpu()
                .long()
            )

            final_weights = (
                self.perturbed_edge_weight
                .detach()
                .cpu()
                .float()
            )

            if final_space.numel() != final_weights.numel():
                raise RuntimeError(
                    "Final current_search_space and "
                    "perturbed_edge_weight are not aligned."
                )

            final_mask = final_weights > 0.5

            self.attack_statistics["rq1"][
                "final_linear_ids"
            ] = torch.unique(
                final_space[final_mask],
                sorted=True,
            )

        if self.block_diagnostics_enabled:
            final_space = (
                self.current_search_space
                .detach()
                .cpu()
                .long()
                .clone()
            )
            final_weights = (
                self.perturbed_edge_weight
                .detach()
                .cpu()
                .float()
                .clone()
            )
            if final_space.numel() != final_weights.numel():
                raise RuntimeError(
                    "Final current_search_space and perturbed_edge_weight "
                    "are not aligned for block diagnostics."
                )
            diagnostics = self.attack_statistics["block_diagnostics"]
            diagnostics["final_block"] = final_space
            diagnostics["final_linear_ids"] = torch.unique(
                final_space[final_weights > 0.5],
                sorted=True,
            )

        self.adj_adversary = SparseTensor.from_edge_index(
            edge_index,
            torch.ones_like(edge_index[0], dtype=torch.float32),
            (self.n, self.n)
        ).coalesce().detach()
        self.attr_adversary = self.attr

        # TODO: Don't we want to switch to returning things? Haha yeah me too

    def _get_logits(self, x, edge_index, edge_weight):
        return self.attacked_model(
            data=x.to(self.device),
            adj=(edge_index.to(self.device), edge_weight.to(self.device))
        )

    # From original PR-BCD
    @torch.no_grad()
    def sample_final_edges(self, n_perturbations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        best_accuracy = float('Inf')
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        # TODO: potentially convert to assert
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0

        for i in range(self.max_final_samples):
            if best_accuracy == float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                logging.info(f'{i}-th sampling: too many samples {n_samples}')
                continue
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            logits = self._get_logits(self.attr, edge_index, edge_weight)
            accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

            # Save best sample
            if best_accuracy > accuracy:
                best_accuracy = accuracy
                best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.device))

        edge_index, edge_weight = self.get_modified_adj()
        edge_mask = edge_weight == 1

        allowed_perturbations = 2 * n_perturbations if self.make_undirected else n_perturbations
        edges_after_attack = edge_mask.sum()
        clean_edges = self.edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    # From original PR-BCD implementation
    def get_modified_adj(self):
        # --- Fallback: no perturbations yet ---
        if getattr(self, "perturbed_edge_weight", None) is None:
            edge_index = self.edge_index.to(self.device)
            edge_weight = (
                self.edge_weight.to(self.device).float()
                if getattr(self, "edge_weight", None) is not None
                else torch.ones(edge_index.size(1), device=self.device)
            )
            return edge_index, edge_weight

        # --- Original logic below ---
        if (
                not self.perturbed_edge_weight.requires_grad
                or not hasattr(self.attacked_model, 'do_checkpoint')
                or not self.attacked_model.do_checkpoint
        ):
            if self.make_undirected:
                modified_edge_index, modified_edge_weight = utils.to_symmetric(
                    self.modified_edge_index, self.perturbed_edge_weight, self.n
                )
            else:
                modified_edge_index, modified_edge_weight = (
                    self.modified_edge_index,
                    self.perturbed_edge_weight,
                )
            edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
            edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

            edge_index, edge_weight = torch_sparse.coalesce(
                edge_index, edge_weight, m=self.n, n=self.n, op='sum'
            )
        else:
            # TODO: test with pytorch 1.9.0
            # Currently (1.6.0) PyTorch does not support return arguments of `checkpoint` that do not require gradient.
            # For this reason we need this extra code and to execute it twice (due to checkpointing in fact 3 times...)
            from torch.utils import checkpoint

            def fuse_edges_run(perturbed_edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if self.make_undirected:
                    modified_edge_index, modified_edge_weight = utils.to_symmetric(
                        self.modified_edge_index, perturbed_edge_weight, self.n
                    )
                else:
                    modified_edge_index, modified_edge_weight = (
                        self.modified_edge_index,
                        self.perturbed_edge_weight,
                    )
                edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
                edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

                edge_index, edge_weight = torch_sparse.coalesce(
                    edge_index, edge_weight, m=self.n, n=self.n, op='sum'
                )
                return edge_index, edge_weight

            # Hack: for very large graphs the block needs to be added on CPU to save memory
            if len(self.edge_weight) > 100_000_000:
                device = self.device
                self.device = 'cpu'
                self.modified_edge_index = self.modified_edge_index.to(self.device)
                edge_index, edge_weight = fuse_edges_run(self.perturbed_edge_weight.cpu())
                self.device = device
                self.modified_edge_index = self.modified_edge_index.to(self.device)
                return edge_index.to(self.device), edge_weight.to(self.device)

            with torch.no_grad():
                edge_index = fuse_edges_run(self.perturbed_edge_weight)[0]

            edge_weight = checkpoint.checkpoint(
                lambda *input: fuse_edges_run(*input)[1],
                self.perturbed_edge_weight,
            )

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]

        return edge_index, edge_weight

    def update_edge_weights(self, n_perturbations: int, epoch: int,
                            gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the edge weights and adaptively, heuristically refined the learning rate such that (1) it is
        independent of the number of perturbations (assuming an undirected adjacency matrix) and (2) to decay learning
        rate during fine-tuning (i.e. fixed search space).

        Parameters
        ----------
            n_perturbations : int
            Number of perturbations.
            epoch : int
            Number of epochs until fine tuning.
            gradient : torch.Tensor
            The current gradient.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated edge indices and weights.
        """
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        self.perturbed_edge_weight.data.add_(lr * gradient)

        # We require for technical reasons that all edges in the block have at least a small positive value
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

        return self.get_modified_adj()

    # Init PR-BCD internal tensors from supplied block
    def init_search_space_from_linear_ids(self, linear_ids: torch.Tensor):
        self.current_search_space = linear_ids
        # Logic from Original PR-BCD
        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(
                self.n, self.current_search_space
            )
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(
                self.n, self.current_search_space
            )
            is_not_self_loop = (
                self.modified_edge_index[0] != self.modified_edge_index[1]
            )
            self.current_search_space = self.current_search_space[is_not_self_loop]
            self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.numel(),),
            float(self.eps),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        return

    def sample_full_search_space(self, n_perturbations: int = 0):
        """Initialize the PRBCD block with every possible edge-flip variable.

        This is intended for RQ1 reference runs. For undirected attacks, the
        search space is exactly all upper-triangular node pairs. For directed
        attacks, self-loops are removed after decoding, matching the existing
        random-block behavior.
        """
        self.current_search_space = torch.arange(
            self.n_possible_edges,
            device=self.device,
            dtype=torch.long,
        )

        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(
                self.n,
                self.current_search_space,
            )
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(
                self.n,
                self.current_search_space,
            )
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            self.current_search_space = self.current_search_space[is_not_self_loop]
            self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        self.perturbed_edge_weight = torch.full_like(
            self.current_search_space,
            self.eps,
            dtype=torch.float32,
            requires_grad=True,
        )

        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError(
                f"Full RQ1 reference search space has fewer edges "
                f"({self.current_search_space.size(0)}) than "
                f"n_perturbations={n_perturbations}."
            )

    def sample_random_block(self, n_perturbations: int = 0, mod_block_size: int = 0):
        for _ in range(self.max_final_samples):
            self.current_search_space = torch.randint(
                self.n_possible_edges, (mod_block_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def sample_block_from_linkpred_threshold(
            self,
            graph,
            tau: float,
            max_sampling_tries: int,
            score_batch_size: int,
            rng_seed: int,
            exclude_tried: bool,
    ):
        X, edge_index = self.extract_X_and_edge_index_from_sparsegraph(graph)

        self.lp_model.eval()
        with torch.no_grad():
            h = self.lp_model.encoder(X, edge_index)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(rng_seed))

        # Parameters to keep track of already accepted edges
        accepted = []
        accepted_count = 0
        accepted_mask = torch.zeros(
            self.n_possible_edges,
            dtype=torch.bool,
            device=self.device,
        )

        # Try repeatedly to fill the block until max_sampling_tries
        for _ in range(max_sampling_tries):
            if accepted_count >= self.block_size:
                break

            # Sample a random candidate set for scoring with size score_batch_size
            cand_lin = torch.unique(
                torch.randint(
                    self.n_possible_edges,
                    (score_batch_size,),
                    device=self.device,
                    generator=generator,
                ),
                sorted=False,
            )

            blocked_mask = self.tried_mask if exclude_tried else accepted_mask
            cand_lin = cand_lin[~blocked_mask[cand_lin]]

            # If all candidates were blocked sample again
            if cand_lin.numel() == 0:
                continue

            if self.make_undirected:
                cand_ei = PRBCD.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCD.linear_to_full_idx(self.n, cand_lin)
                valid = cand_ei[0] != cand_ei[1]
                cand_lin = cand_lin[valid]
                cand_ei = cand_ei[:, valid]

            if cand_lin.numel() == 0:
                continue

            with torch.no_grad():
                scores = torch.sigmoid(self.lp_model.edge_head(h, cand_ei).view(-1))

            if exclude_tried:
                self.tried_mask[cand_lin] = True

            # Select only those candidates that are above the tau threshold
            selected = cand_lin[scores >= tau]
            remaining = self.block_size - accepted_count
            # Prevents overfilling the block
            selected = selected[:remaining]

            if selected.numel() > 0:
                accepted.append(selected)
                accepted_mask[selected] = True
                accepted_count += selected.numel()

        if accepted_count < self.block_size:
            raise RuntimeError(
                f"Could not fill selector block: "
                f"selected={accepted_count}/{self.block_size}, tau={tau}."
            )

        self.current_search_space = torch.cat(accepted)

        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(
                self.n,
                self.current_search_space,
            )
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(
                self.n,
                self.current_search_space,
            )

        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.numel(),),
            self.eps,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

    def resample_random_block(self, n_perturbations: int, mod_block_size: int): #TODO: still work to be split_by_eps
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(self.max_final_samples):
            n_edges_resample = mod_block_size - self.current_search_space.size(0)
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
            self.perturbed_edge_weight[
                unique_idx[:perturbed_edge_weight_old.size(0)]
            ] = perturbed_edge_weight_old

            if not self.make_undirected:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

            if self.current_search_space.size(0) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def resample_block_from_linkpred_topk(
            self,
            graph,
            top_k_per_batch: int = 100,
            score_batch_size: int = 10_000,
            rng_seed: int = 0,
            exclude_tried: bool = True,
    ):
        """Refill the PRBCD block using top_k_per_batch from a batch of size score_batch_size."""

        # Keep step from PR-BCD, keeps at most half of the block
        if self.keep_heuristic == "WeightOnly":
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError(
                "Only keep_heuristic=`WeightOnly` supported"
            )

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # Number of edge ids needed to refill the block
        n_needed = self.block_size - self.current_search_space.numel()

        X, _ = self.extract_X_and_edge_index_from_sparsegraph(graph)

        # Encode current relaxed perturbations, GCN Conv natively accepts that.
        with torch.no_grad():
            edge_index_struct, edge_weight = self.get_modified_adj()
            h = self.lp_model.encoder(X, edge_index_struct, edge_weight)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(rng_seed))

        blocked_mask = torch.zeros(
            self.n_possible_edges,
            device=self.device,
            dtype=torch.bool,
        )

        # Keeping this blocked mask is sadly O(N^2)
        blocked_mask[self.current_search_space] = True
        if exclude_tried:
            blocked_mask |= self.tried_mask

        # Create a pool of all edges that are not blocked
        allowed_pool = torch.nonzero(~blocked_mask, as_tuple=False).view(-1)

        # Throw exception when allowed pool gets too small for refilling the block
        if allowed_pool.numel() < n_needed:
            raise RuntimeError(
                "Not enough candidates for top-k resampling: "
                f"needed={n_needed}, eligible={allowed_pool.numel()}, "
                f"exclude_tried={exclude_tried}."
            )

        # Order the allowed pool to subsequently sample score batches from them
        sampling_order = torch.randperm(
            allowed_pool.numel(),
            device=self.device,
            generator=generator,
        )

        # Let a counter run over all allowed edges
        sampling_cursor = 0
        accepted_ids = []
        accepted_count = 0

        while accepted_count < n_needed and sampling_cursor < sampling_order.numel():
            batch_end = min(sampling_cursor + score_batch_size, sampling_order.numel())
            batch_positions = sampling_order[sampling_cursor:batch_end]
            sampling_cursor = batch_end
            cand_lin = allowed_pool[batch_positions]

            # From original PR-BCD, in this thesis, we only evaluate undirected graphs.
            if self.make_undirected:
                cand_ei = PRBCD.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCD.linear_to_full_idx(self.n, cand_lin)
                is_not_self = cand_ei[0] != cand_ei[1]
                cand_lin = cand_lin[is_not_self]
                cand_ei = cand_ei[:, is_not_self]
                if cand_lin.numel() == 0:
                    continue

            # Score batch with selector
            with torch.no_grad():
                logits = self.lp_model.edge_head(h, cand_ei).view(-1)
                scores = torch.sigmoid(logits)

            # Exclude every scored candidate across later epochs.
            if exclude_tried:
                self.tried_mask[cand_lin] = True

            current_k = min(top_k_per_batch, scores.numel())
            top_positions = torch.topk(scores, k=current_k).indices
            batch_top_ids = cand_lin[top_positions]

            remaining = n_needed - accepted_count
            batch_top_ids = batch_top_ids[:remaining]
            accepted_ids.append(batch_top_ids)
            accepted_count += int(batch_top_ids.numel())

        if accepted_count < n_needed:
            raise RuntimeError(
                f"Could not refill block: needed={n_needed}, accepted={accepted_count}."
            )

        fill_lin = torch.cat(accepted_ids)

        # Init edge weights of the new block with epsilon
        new_weights = torch.full(
            (fill_lin.numel(),),
            self.eps,
            dtype=torch.float32,
            device=self.device,
        )

        # Fill PR-BCD optimization block
        self.current_search_space = torch.cat((self.current_search_space, fill_lin))
        self.perturbed_edge_weight = torch.cat((self.perturbed_edge_weight, new_weights))

        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(
                self.n, self.current_search_space
            )
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(
                self.n, self.current_search_space
            )

    @staticmethod
    def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = (
            n
            - 2
            - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
        ).long()
        col_idx = (
            lin_idx
            + row_idx
            + 1 - n * (n - 1) // 2
            + (n - row_idx) * ((n - row_idx) - 1) // 2
        )
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def linear_to_full_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = lin_idx // n
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def full_to_linear_idx(n: int, full_idx: torch.Tensor) -> torch.Tensor:
        # this function made errors in the undirected case, maybe this will be useful for the directed case
        row_idx, col_idx = full_idx[0], full_idx[1]
        lin_idx = row_idx * n + col_idx
        return lin_idx

    @staticmethod
    def triu_idx_to_linear_idx(n: int, full_idx: torch.Tensor) -> torch.Tensor:
        # if im correct in the undirected case, the indexing of the block matrix is different to the directed case (see my note: (1) )
        row_idx, col_idx = full_idx[0], full_idx[1]
        lin_idx = (n * row_idx - row_idx * (row_idx + 1) // 2) + (col_idx - row_idx - 1)
        return lin_idx

    def _append_attack_statistics(
            self,
            loss,
            accuracy,
            probability_mass_update,
            probability_mass_projected,
            *,
            epoch=None,
            gradient=None,
    ):
        epoch_id = -1 if epoch is None else int(epoch)

        # Normal PRBCD metrics
        self.attack_statistics["epoch"].append(epoch_id)
        self.attack_statistics["loss"].append(float(loss))
        self.attack_statistics["accuracy"].append(float(accuracy))
        self.attack_statistics["weights_above_eps"].append(
            int((self.perturbed_edge_weight.detach() > self.eps).sum())
        )
        self.attack_statistics["probability_mass_update"].append(
            float(probability_mass_update)
        )
        self.attack_statistics["probability_mass_projected"].append(
            float(probability_mass_projected)
        )

        # Nothing else required
        if not self.block_diagnostics_enabled and not self.rq1_enabled:
            return

        current_ids = (
            self.current_search_space
            .detach()
            .cpu()
            .long()
            .clone()
        )

        # Block history
        if self.block_diagnostics_enabled:
            diagnostics = self.attack_statistics["block_diagnostics"]

            if epoch is None:
                diagnostics["initial_block"] = current_ids
            else:
                diagnostics["epoch_blocks"][int(epoch)] = current_ids

        # RQ1 reference metrics
        if not self.rq1_enabled or epoch is None:
            return

        if self.rq1_is_reference:
            if gradient is None:
                raise RuntimeError(
                    "RQ1 reference recording requires gradient."
                )

            current_gradient = (
                gradient.detach().cpu().float()
            )

            current_weights = (
                self.perturbed_edge_weight
                .detach()
                .cpu()
                .float()
            )

            rq1 = self.attack_statistics["rq1"]

            rq1["positive_gradient_sum"].index_add_(
                0,
                current_ids,
                current_gradient.clamp_min(0.0),
            )

            rq1["max_weight"][current_ids] = torch.maximum(
                rq1["max_weight"][current_ids],
                current_weights,
            )

    # Directly extracts edge index and attribute matrix from graph
    #Generated by ChatGPT 5.1
    def extract_X_and_edge_index_from_sparsegraph(self, graph):
        N, d = graph.attr_matrix.shape

        # ---- Features X: (N, d) dense ----
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(
            np.vstack([X_coo.row, X_coo.col]),
            dtype=torch.long,
            device=self.device,
        )
        X_val = torch.tensor(X_coo.data, dtype=torch.float32, device=self.device)
        X_sparse = torch.sparse_coo_tensor(
            X_idx, X_val, size=(N, d), device=self.device
        ).coalesce()
        X = X_sparse.to_dense()  # (N, d)

        # ---- Adjacency A → edge_index_struct: (2, E) ----
        A_coo = graph.adj_matrix.tocoo()
        A_idx = torch.tensor(
            np.vstack([A_coo.row, A_coo.col]),
            dtype=torch.long,
            device=self.device,
        )
        A_val = torch.tensor(A_coo.data, dtype=torch.float32, device=self.device)
        A_sparse = torch.sparse_coo_tensor(
            A_idx, A_val, size=(N, N), device=self.device
        ).coalesce()

        edge_index_struct = A_sparse.indices()  # (2, E)
        # optionally remove self-loops:
        # edge_index_struct = PRBCD.cut_diagonal_entries(edge_index_struct)

        return X, edge_index_struct

    @staticmethod
    def cut_matrix_idx_to_triu_idx(matrix: torch.tensor) -> torch.Tensor:
        # cut all entries of matrix where the entry (x,y) holds x >= y
        row_idx = matrix[0]
        col_idx = matrix[1]
        mask = row_idx < col_idx
        #returns undirected triu matrix
        return matrix[:, mask]

    @staticmethod
    def pairs_to_linear_uppertri(pairs: torch.Tensor, N: int, *, drop_self_loops: bool = True) -> torch.Tensor:
        """
        Convert 2×b node index pairs into a (b,) vector of linear indices over the
        upper-triangular part (u < v) of an N×N matrix, row-major over (u,v).

        Mapping (u < v) -> k:
            k = u*(2*N - u - 1)//2 + (v - u - 1)
        Range: k ∈ [0, N*(N-1)//2)

        Args:
            pairs: LongTensor of shape (2, b) or (b, 2). Each pair is (u, v).
            N:     number of nodes.
            drop_self_loops: if True, removes u==v; if False, they’re filtered anyway by u<v.

        Returns:
            lin: LongTensor of shape (b_valid,), linear indices for valid u<v pairs.
        """
        # Accept either (2,b) or (b,2)
        if pairs.dim() != 2 or not (pairs.size(0) == 2 or pairs.size(1) == 2):
            raise ValueError("pairs must be shape (2, b) or (b, 2) of longs")
        if pairs.size(0) == 2:
            u, v = pairs[0], pairs[1]
        else:
            u, v = pairs[:, 0], pairs[:, 1]

        # Make undirected by ordering (u,v) -> (min,max)
        uu = torch.minimum(u, v)
        vv = torch.maximum(u, v)

        # Keep only strict upper-tri (uu < vv)
        mask = uu < vv
        if drop_self_loops:
            # already excluded by uu < vv, but keeps intent explicit
            pass

        if mask.sum() == 0:
            return torch.empty(0, dtype=torch.long, device=pairs.device)

        uu = uu[mask]
        vv = vv[mask]

        # Apply the closed-form linearization for the upper triangle
        # k = uu*(2N - uu - 1)//2 + (vv - uu - 1)
        N = int(N)
        lin = uu * (2 * N - uu - 1) // 2 + (vv - uu - 1)
        return lin

    @staticmethod
    def flip_matrix_idx_to_triu_idx(matrix: torch.tensor) -> torch.tensor:
        matrix = PRBCD.cut_diagonal_entries(matrix)
        row_idx = matrix[0]
        col_idx = matrix[1]
        # flip all entries of matrix where the entry (x,y) holds x>y
        mask = row_idx > col_idx
        temp = col_idx[mask]
        col_idx[mask] = row_idx[mask]
        row_idx[mask] = temp
        return matrix

    @staticmethod
    def cut_diagonal_entries(matrix: torch.tensor) -> torch.Tensor:
        row_idx = matrix[0]
        col_idx = matrix[1]
        mask = row_idx != col_idx
        # returns directed matrix (diagonal is cut)
        return matrix[:, mask]

    def _load_selector_params(self, selector_params: dict):
        selector_params = selector_params or {}

        self.tau = selector_params.get("tau", 0.8)
        self.score_batch_size = selector_params.get("score_batch_size", 1000)
        self.max_sampling_tries = selector_params.get("max_sampling_tries", 2_000_000)
        self.exclude_tried = selector_params.get("exclude_tried", True)
        self.top_k_per_batch = selector_params.get("top_k_per_batch",100)
