"""LP-guided extension of the clean PRBCD attack.

Place this file next to ``prbcd_clean.py`` and adjust the import below if your
clean PRBCD module has a different name.

Expected LP-model interface
---------------------------
The selector model must expose:

    h = lp_model.encoder(x, edge_index)
    logits = lp_model.edge_head(h, candidate_edge_index)

where ``candidate_edge_index`` has shape ``[2, M]``. The logits are converted
with sigmoid, matching a model trained with BCEWithLogitsLoss.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from rgnn_at_scale.attacks.prbcd import PRBCD


TensorLike = Union[torch.Tensor, list, tuple]


class LPPRBCD(PRBCD):
    """PRBCD with an externally supplied or LP-selected initial search block.

    Initialization precedence:

    1. ``initial_block`` passed to the constructor.
    2. A block sampled and thresholded by ``lp_model``.
    3. The original random PRBCD initialization.

    The clean PRBCD implementation itself remains unchanged. Its ``_attack``
    method calls ``sample_random_block`` once for initialization; this subclass
    overrides that method and inserts the requested block instead.

    Parameters
    ----------
    initial_block:
        Optional initial PRBCD block. Accepted formats are:

        - shape ``[2, B]`` containing node pairs;
        - shape ``[B, 2]`` containing node pairs;
        - shape ``[B]`` containing PRBCD linear edge indices.

        For an undirected graph, pair endpoints are normalized to ``u < v``.
        Duplicate pairs are removed. The resulting number of unique candidates
        must equal ``block_size``.

    lp_model:
        Optional trained LP selector. It must expose ``encoder`` and
        ``edge_head`` as described in the module docstring.

    lp_threshold:
        Only randomly sampled pairs with ``sigmoid(logit) >= threshold`` enter
        the PRBCD block.

    lp_candidate_batch_size:
        Number of random edge candidates generated in one sampling round.

    lp_score_batch_size:
        Number of candidate pairs scored by the LP head in one forward batch.

    lp_max_candidates:
        Maximum number of random candidates that may be scored while trying to
        fill one block. A RuntimeError is raised if the threshold is too strict.

    lp_random_seed:
        Random seed used for candidate generation.

    lp_resample:
        If False, only the initial block is LP-guided and ordinary PRBCD random
        resampling is used afterward. If True, every PRBCD refill also uses the
        LP threshold.

    lp_use_current_graph_for_resampling:
        If True and ``lp_resample=True``, node embeddings for a refill are
        computed using the current perturbed graph. Otherwise the clean graph is
        always used by the LP encoder.
    """

    def __init__(
        self,
        *,
        initial_block: Optional[TensorLike] = None,
        lp_model: Optional[torch.nn.Module] = None,
        lp_threshold: float = 0.8,
        lp_candidate_batch_size: int = 100_000,
        lp_score_batch_size: int = 100_000,
        lp_max_candidates: int = 5_000_000,
        lp_random_seed: int = 0,
        lp_resample: bool = False,
        lp_use_current_graph_for_resampling: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if not 0.0 <= float(lp_threshold) <= 1.0:
            raise ValueError("lp_threshold must lie in [0, 1].")
        if int(lp_candidate_batch_size) <= 0:
            raise ValueError("lp_candidate_batch_size must be positive.")
        if int(lp_score_batch_size) <= 0:
            raise ValueError("lp_score_batch_size must be positive.")
        if int(lp_max_candidates) <= 0:
            raise ValueError("lp_max_candidates must be positive.")

        self.initial_block = initial_block
        self.lp_model = lp_model
        self.lp_threshold = float(lp_threshold)
        self.lp_candidate_batch_size = int(lp_candidate_batch_size)
        self.lp_score_batch_size = int(lp_score_batch_size)
        self.lp_max_candidates = int(lp_max_candidates)
        self.lp_random_seed = int(lp_random_seed)
        self.lp_resample = bool(lp_resample)
        self.lp_use_current_graph_for_resampling = bool(
            lp_use_current_graph_for_resampling
        )

        self._initial_block_consumed = False
        self._lp_sampling_call = 0

    # ------------------------------------------------------------------
    # Public block initializer
    # ------------------------------------------------------------------

    def initialize_from_block(
        self,
        block: TensorLike,
        n_perturbations: int = 0,
    ) -> None:
        """Initialize PRBCD tensors from an external block.

        This initializes the same three tensors as clean PRBCD random sampling:

        - ``current_search_space``
        - ``modified_edge_index``
        - ``perturbed_edge_weight``

        Note that clean PRBCD has no persistent ``modified_edge_weight`` field.
        The corresponding optimization variable is ``perturbed_edge_weight``;
        the actual modified weights are assembled by ``get_modified_adj``.
        """

        linear_block = self._block_to_linear_indices(block)

        if linear_block.numel() != int(self.block_size):
            raise ValueError(
                "The external block must contain exactly block_size unique "
                f"candidates after normalization. Got {linear_block.numel()}, "
                f"expected {self.block_size}."
            )

        if linear_block.numel() < int(n_perturbations):
            raise ValueError(
                f"The input block contains {linear_block.numel()} candidates, "
                f"but n_perturbations={n_perturbations}."
            )

        self._set_search_space(linear_block)

    # ------------------------------------------------------------------
    # Hook used by the unchanged clean PRBCD._attack implementation
    # ------------------------------------------------------------------

    def sample_random_block(
            self,
            n_perturbations: int = 0,
            block_size: int | None = None,
    ) -> None:
        """
        Initialize the PRBCD search space from:

        1. An externally supplied block;
        2. The LP-GNN thresholded candidate pool;
        3. The parent PRBCD random sampler.

        `block_size` is accepted because the modified parent PRBCD calls:

            sample_random_block(n_perturbations, self.block_size)
        """

        target_size = (
            int(self.block_size)
            if block_size is None
            else int(block_size)
        )

        if target_size <= int(n_perturbations):
            raise ValueError(
                f"block_size ({target_size}) must be greater than "
                f"n_perturbations ({n_perturbations})."
            )

        # --------------------------------------------------------
        # Externally supplied initial block
        # --------------------------------------------------------

        if (
                self.initial_block is not None
                and not self._initial_block_consumed
        ):
            logging.info(
                "[LPPRBCD] Initializing from supplied edge block."
            )

            linear_block = self._block_to_linear_indices(
                self.initial_block
            )

            if linear_block.numel() != target_size:
                raise ValueError(
                    "The supplied initial block must contain exactly "
                    f"{target_size} unique candidates after normalization. "
                    f"Got {linear_block.numel()}."
                )

            self._set_search_space(linear_block)

            self._initial_block_consumed = True
            return

        # --------------------------------------------------------
        # LP-guided initial block
        # --------------------------------------------------------

        if self.lp_model is not None:
            logging.info(
                "[LPPRBCD] Building LP-guided block: "
                "target_size=%d, threshold=%.4f",
                target_size,
                self.lp_threshold,
            )

            linear_block = self._sample_lp_threshold_block(
                target_size=target_size,
                exclude_linear=None,
                use_current_graph=False,
            )

            self._set_search_space(linear_block)

            if self.current_search_space.numel() < target_size:
                raise RuntimeError(
                    "LP-guided sampling did not fill the requested block. "
                    f"Created {self.current_search_space.numel()} candidates, "
                    f"but target_size={target_size}."
                )

            if (
                    self.current_search_space.numel()
                    < int(n_perturbations)
            ):
                raise RuntimeError(
                    "LP-selected block is smaller than the attack budget: "
                    f"{self.current_search_space.numel()} < "
                    f"{n_perturbations}."
                )

            logging.info(
                "[LPPRBCD] LP-guided block initialized with %d candidates.",
                self.current_search_space.numel(),
            )

            return

        # --------------------------------------------------------
        # Random fallback for your modified parent PRBCD
        # --------------------------------------------------------

        super().sample_random_block(
            n_perturbations,
            target_size,
        )

    # ------------------------------------------------------------------
    # Optional LP-guided PRBCD resampling
    # ------------------------------------------------------------------

    def resample_random_block(
            self,
            n_perturbations: int,
            mod_block_size: int | None = None,
    ) -> None:
        """
        Resample the PRBCD block.

        Supports the modified parent PRBCD interface:

            resample_random_block(
                n_perturbations,
                mod_block_size=self.block_size,
            )

        If lp_resample=False, use normal PRBCD resampling.
        If lp_resample=True, refill the block with LP-selected candidates.
        """

        target_size = (
            int(self.block_size)
            if mod_block_size is None
            else int(mod_block_size)
        )

        # ========================================================
        # Normal PRBCD resampling
        # ========================================================

        if not self.lp_resample or self.lp_model is None:
            return super().resample_random_block(
                n_perturbations,
                mod_block_size=target_size,
            )

        # ========================================================
        # LP-guided resampling
        # ========================================================

        if self.keep_heuristic != "WeightOnly":
            raise NotImplementedError(
                "Only keep_heuristic='WeightOnly' is supported."
            )

        # Sort candidates by their optimized PRBCD weights.
        sorted_idx = torch.argsort(
            self.perturbed_edge_weight
        )

        # Remove candidates whose weights remained at epsilon.
        idx_keep = (
                self.perturbed_edge_weight <= self.eps
        ).sum().long()

        # Keep at most the strongest half of the existing block.
        if idx_keep < sorted_idx.size(0) // 2:
            idx_keep = sorted_idx.size(0) // 2

        keep_idx = sorted_idx[idx_keep:]

        kept_linear = (
            self.current_search_space[keep_idx]
            .detach()
            .to(self.device)
        )

        kept_weights = (
            self.perturbed_edge_weight[keep_idx]
            .detach()
            .to(self.device)
        )

        n_needed = target_size - int(
            kept_linear.numel()
        )

        # Nothing needs to be refilled.
        if n_needed <= 0:
            self._set_search_space(
                kept_linear,
                existing_weights=kept_weights,
            )
            return

        logging.info(
            "[LPPRBCD] Refilling %d candidates with "
            "LP threshold %.4f.",
            n_needed,
            self.lp_threshold,
        )

        # Sample new random pairs and retain those whose LP score
        # exceeds lp_threshold.
        new_linear = self._sample_lp_threshold_block(
            target_size=n_needed,
            exclude_linear=kept_linear,
            use_current_graph=(
                self.lp_use_current_graph_for_resampling
            ),
        )

        combined_linear = torch.cat(
            [
                kept_linear,
                new_linear.to(self.device),
            ],
            dim=0,
        )

        # Sort the new search space and obtain the mapping needed
        # to restore the retained PRBCD weights.
        new_search_space, inverse = torch.unique(
            combined_linear,
            sorted=True,
            return_inverse=True,
        )

        if new_search_space.numel() != target_size:
            raise RuntimeError(
                "LP-guided resampling did not create the requested "
                f"block size. Got {new_search_space.numel()}, "
                f"expected {target_size}."
            )

        # New candidates start at epsilon. Retained candidates keep
        # their existing optimized weights.
        new_weights = torch.full(
            (new_search_space.numel(),),
            float(self.eps),
            dtype=torch.float32,
            device=self.device,
        )

        old_positions = inverse[
                        :kept_linear.numel()
                        ]

        new_weights[old_positions] = kept_weights

        self._set_search_space(
            new_search_space,
            existing_weights=new_weights,
        )

        if (
                self.current_search_space.numel()
                <= int(n_perturbations)
        ):
            raise RuntimeError(
                "The resampled block must contain more candidates "
                "than the attack perturbation budget."
            )

    # ------------------------------------------------------------------
    # LP candidate generation and scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_lp_threshold_block(
        self,
        *,
        target_size: int,
        exclude_linear: Optional[torch.Tensor],
        use_current_graph: bool,
    ) -> torch.Tensor:
        """Randomly sample pairs and keep those passing the LP threshold.

        Sampling continues until exactly ``target_size`` unique candidates have
        passed the threshold. If more candidates pass than needed, the highest
        scoring passing candidates are retained.
        """

        if self.lp_model is None:
            raise RuntimeError("lp_model is not configured.")
        if target_size <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        lp_device = self._model_device(self.lp_model)
        self.lp_model.to(lp_device)
        self.lp_model.eval()

        x = self.attr.to(lp_device)
        structure = self._lp_structure(use_current_graph).to(lp_device)
        node_embeddings = self.lp_model.encoder(x, structure)

        # CPU RNG keeps candidate generation independent of the LP device and
        # works for CPU and CUDA models alike.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.lp_random_seed + self._lp_sampling_call
        )
        self._lp_sampling_call += 1

        excluded_cpu = None
        if exclude_linear is not None and exclude_linear.numel() > 0:
            excluded_cpu = torch.unique(
                exclude_linear.detach().to("cpu", dtype=torch.long),
                sorted=False,
            )

        accepted_linear = torch.empty(0, dtype=torch.long)
        accepted_scores = torch.empty(0, dtype=torch.float32)
        n_scored = 0

        while (
            accepted_linear.numel() < target_size
            and n_scored < self.lp_max_candidates
        ):
            remaining_budget = self.lp_max_candidates - n_scored
            current_batch_size = min(
                self.lp_candidate_batch_size,
                remaining_budget,
            )

            candidate_linear = torch.randint(
                low=0,
                high=int(self.n_possible_edges),
                size=(current_batch_size,),
                generator=generator,
                dtype=torch.long,
                device="cpu",
            )
            candidate_linear = torch.unique(candidate_linear, sorted=False)

            # Exclude edges already retained by PRBCD and edges already accepted
            # in previous rounds.
            if excluded_cpu is not None and excluded_cpu.numel() > 0:
                candidate_linear = candidate_linear[
                    ~torch.isin(candidate_linear, excluded_cpu)
                ]
            if accepted_linear.numel() > 0:
                candidate_linear = candidate_linear[
                    ~torch.isin(candidate_linear, accepted_linear)
                ]

            if candidate_linear.numel() == 0:
                n_scored += current_batch_size
                continue

            candidate_pairs = self._linear_to_pairs(
                candidate_linear.to(lp_device)
            )

            # Directed full indexing includes self-loops. They are not valid
            # PRBCD candidates, so filter them before scoring.
            if not self.make_undirected:
                non_self = candidate_pairs[0] != candidate_pairs[1]
                candidate_pairs = candidate_pairs[:, non_self]
                candidate_linear = candidate_linear[non_self.cpu()]

            if candidate_linear.numel() == 0:
                n_scored += current_batch_size
                continue

            batch_scores = torch.empty(
                candidate_linear.numel(),
                dtype=torch.float32,
                device="cpu",
            )

            for start in range(
                0,
                candidate_linear.numel(),
                self.lp_score_batch_size,
            ):
                end = min(
                    start + self.lp_score_batch_size,
                    candidate_linear.numel(),
                )
                logits = self.lp_model.edge_head(
                    node_embeddings,
                    candidate_pairs[:, start:end],
                ).reshape(-1)
                batch_scores[start:end] = torch.sigmoid(logits).to("cpu")

            passing = batch_scores >= self.lp_threshold
            if passing.any():
                accepted_linear = torch.cat(
                    [accepted_linear, candidate_linear[passing]],
                    dim=0,
                )
                accepted_scores = torch.cat(
                    [accepted_scores, batch_scores[passing]],
                    dim=0,
                )

            n_scored += current_batch_size

            logging.debug(
                "[LPPRBCD] scored=%d accepted=%d/%d threshold=%.4f",
                n_scored,
                accepted_linear.numel(),
                target_size,
                self.lp_threshold,
            )

        if accepted_linear.numel() < target_size:
            acceptance_rate = (
                accepted_linear.numel() / max(1, n_scored)
            )
            raise RuntimeError(
                "Could not fill the LP-guided PRBCD block. "
                f"Accepted {accepted_linear.numel()} of {n_scored} scored "
                f"candidates (rate={acceptance_rate:.6f}) with "
                f"threshold={self.lp_threshold}, but target_size={target_size}. "
                "Lower lp_threshold or increase lp_max_candidates."
            )

        # If the last round produced more than needed, retain the strongest
        # threshold-passing candidates.
        top_indices = torch.topk(
            accepted_scores,
            k=target_size,
            largest=True,
            sorted=False,
        ).indices
        selected_linear = accepted_linear[top_indices]
        selected_linear = torch.unique(selected_linear, sorted=True)

        if selected_linear.numel() != target_size:
            # This should be exceptionally rare because duplicates are filtered,
            # but fail loudly rather than silently returning a short block.
            raise RuntimeError(
                "LP candidate deduplication produced a short block: "
                f"{selected_linear.numel()} != {target_size}."
            )

        return selected_linear.to(self.device)

    def _lp_structure(self, use_current_graph: bool) -> torch.Tensor:
        """Return the graph structure used by the LP encoder."""

        if not use_current_graph:
            return self.edge_index

        with torch.no_grad():
            edge_index, edge_weight = self.get_modified_adj()
            present = edge_weight > 0.5
            return edge_index[:, present]

    # ------------------------------------------------------------------
    # Tensor conversion and validation
    # ------------------------------------------------------------------

    def _set_search_space(
        self,
        linear_indices: torch.Tensor,
        existing_weights: Optional[torch.Tensor] = None,
    ) -> None:
        linear_indices = linear_indices.to(self.device, dtype=torch.long)

        if existing_weights is None:
            linear_indices = torch.unique(linear_indices, sorted=True)
        else:
            existing_weights = existing_weights.to(
                self.device,
                dtype=torch.float32,
            )
            if existing_weights.numel() != linear_indices.numel():
                raise ValueError(
                    "existing_weights and linear_indices must have equal length."
                )

            # Preserve the edge-to-weight alignment while restoring PRBCD's
            # sorted search-space convention.
            linear_indices, order = torch.sort(linear_indices)
            existing_weights = existing_weights[order]

            if linear_indices.numel() > 1 and torch.any(
                linear_indices[1:] == linear_indices[:-1]
            ):
                raise ValueError(
                    "Weighted search-space initialization contains duplicate "
                    "linear edge indices."
                )

        if linear_indices.numel() == 0:
            raise ValueError("The PRBCD search block cannot be empty.")
        if linear_indices.min().item() < 0:
            raise ValueError("Linear edge indices must be non-negative.")
        if linear_indices.max().item() >= int(self.n_possible_edges):
            raise ValueError(
                "A linear edge index lies outside the possible-edge space."
            )

        modified_edge_index = self._linear_to_pairs(linear_indices)

        if not self.make_undirected:
            non_self = modified_edge_index[0] != modified_edge_index[1]
            linear_indices = linear_indices[non_self]
            modified_edge_index = modified_edge_index[:, non_self]
            if existing_weights is not None:
                existing_weights = existing_weights[non_self]

        self.current_search_space = linear_indices
        self.modified_edge_index = modified_edge_index

        if existing_weights is None:
            self.perturbed_edge_weight = torch.full(
                (linear_indices.numel(),),
                float(self.eps),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
        else:
            self.perturbed_edge_weight = (
                existing_weights
                .clone()
                .detach()
                .requires_grad_(True)
            )

    def _block_to_linear_indices(self, block: TensorLike) -> torch.Tensor:
        tensor = torch.as_tensor(block, dtype=torch.long, device=self.device)

        if tensor.ndim == 1:
            return torch.unique(tensor, sorted=True)

        if tensor.ndim != 2:
            raise ValueError(
                "initial_block must be a 1D linear-index tensor or a 2D "
                "edge-pair tensor."
            )

        if tensor.size(0) == 2:
            pairs = tensor
        elif tensor.size(1) == 2:
            pairs = tensor.t().contiguous()
        else:
            raise ValueError(
                "A pair block must have shape [2, B] or [B, 2]."
            )

        if pairs.numel() == 0:
            raise ValueError("initial_block cannot be empty.")
        if pairs.min().item() < 0 or pairs.max().item() >= int(self.n):
            raise ValueError("initial_block contains an invalid node index.")

        if self.make_undirected:
            u = torch.minimum(pairs[0], pairs[1])
            v = torch.maximum(pairs[0], pairs[1])
            non_self = u != v
            u = u[non_self]
            v = v[non_self]
            if u.numel() == 0:
                raise ValueError("initial_block contains only self-loops.")
            linear = self.triu_idx_to_linear(
                self.n,
                torch.stack([u, v], dim=0),
            )
        else:
            non_self = pairs[0] != pairs[1]
            pairs = pairs[:, non_self]
            if pairs.numel() == 0:
                raise ValueError("initial_block contains only self-loops.")
            linear = pairs[0] * int(self.n) + pairs[1]

        return torch.unique(linear, sorted=True)

    def _linear_to_pairs(self, linear_indices: torch.Tensor) -> torch.Tensor:
        if self.make_undirected:
            return self.linear_to_triu_idx(self.n, linear_indices)
        return self.linear_to_full_idx(self.n, linear_indices)

    @staticmethod
    def triu_idx_to_linear(n: int, edge_index: torch.Tensor) -> torch.Tensor:
        """Map undirected pairs ``u < v`` to PRBCD upper-triangle indices."""

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, M].")

        u = edge_index[0].long()
        v = edge_index[1].long()
        if not torch.all(u < v):
            raise ValueError("triu_idx_to_linear requires u < v.")

        row_start = u * (2 * int(n) - u - 1) // 2
        return row_start + (v - u - 1)

    @staticmethod
    def _model_device(model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
