import copy
import logging
import math
from typing import Tuple, Any, Literal

import numpy as np
import torch
import torch_sparse
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn, Tensor
from tqdm import tqdm

from AttackerGNN.ShadowModelLinkPredictor import LinkPredictionGNN
from rgnn_at_scale.helper import utils
from rgnn_at_scale.attacks.base_attack import Attack


class EndpointPRBCDV4Scorer:
    """
    PRBCD candidate mining + V4-style subset accuracy-drop scoring.

    Main output:
        src, dst, labels, exists

    where labels are continuous V4-style scores in [0, 1].
    """

    def __init__(
        self,
        *,
        n: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        attr: torch.Tensor,
        labels: torch.Tensor,
        attacked_model,
        idx_attack=None,
        test_idx=None,
        device=None,
        block_size=None,
        n_epochs_resampling=None,

        # PRBCD / attack state
        make_undirected: bool = True,
        eps: float = 1e-7,
        max_final_samples: int = 20,
        display_step: int = 20,
        do_synchronize: bool = False,

        # V4-style defaults
        n_subsets: int = 50,
        subset_fraction: float = 0.1,
        balance_ratio: float | None = None,
        store_candidates: bool = False,
    ):
        if device is None:
            device = attr.device if hasattr(attr, "device") else "cpu"

        self.device = device
        self.n = int(n)

        self.edge_index = edge_index.to(device=device, dtype=torch.long).contiguous()

        if edge_weight is None:
            self.edge_weight = torch.ones(
                self.edge_index.size(1),
                device=device,
                dtype=torch.float32,
            )
        else:
            self.edge_weight = edge_weight.to(device=device).float().contiguous()

        self.attr = attr.to(device)
        self.labels = labels.to(device)
        self.attacked_model = attacked_model

        self.keep_heuristic = 'WeightOnly'
        self.loss_type = 'tanhMargin'

        self.idx_attack = idx_attack
        self.test_idx = test_idx

        self.make_undirected = bool(make_undirected)
        self.eps = float(eps)
        self.max_final_samples = int(max_final_samples)
        self.display_step = int(display_step)
        self.do_synchronize = bool(do_synchronize)

        self.n_subsets = int(n_subsets)
        self.subset_fraction = float(subset_fraction)
        self.balance_ratio = balance_ratio
        self.store_candidates = bool(store_candidates)
        self.block_size = int(block_size)
        self.n_epochs_resampling = int(n_epochs_resampling)

        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n * self.n

        self.lr_factor = 100 * max(math.log2(self.n_possible_edges / self.block_size), 1.)

        self.current_search_space = None
        self.modified_edge_index = None
        self.perturbed_edge_weight = None
        self.gradient = None
        self._all_candidates = None

    # ------------------------------------------------------------------
    # index helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pairs_to_linear_uppertri(
        pairs: torch.Tensor,
        N: int,
        *,
        drop_self_loops: bool = True,
    ) -> torch.Tensor:
        """
        Convert 2×b node pairs into linear upper-triangle ids.
        Mapping:
            k = u * (2N - u - 1) // 2 + (v - u - 1)
        """
        if pairs.dim() != 2 or not (pairs.size(0) == 2 or pairs.size(1) == 2):
            raise ValueError("pairs must be shape (2, b) or (b, 2)")

        if pairs.size(0) == 2:
            u, v = pairs[0], pairs[1]
        else:
            u, v = pairs[:, 0], pairs[:, 1]

        uu = torch.minimum(u, v)
        vv = torch.maximum(u, v)

        mask = uu < vv
        if mask.sum() == 0:
            return torch.empty(0, dtype=torch.long, device=pairs.device)

        uu = uu[mask]
        vv = vv[mask]

        N = int(N)
        return uu * (2 * N - uu - 1) // 2 + (vv - uu - 1)

    @staticmethod
    def triu_idx_to_linear_idx(n: int, full_idx: torch.Tensor) -> torch.Tensor:
        row_idx, col_idx = full_idx[0], full_idx[1]
        return (n * row_idx - row_idx * (row_idx + 1) // 2) + (col_idx - row_idx - 1)

    @staticmethod
    def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = (
            n
            - 2
            - torch.floor(
                torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5
            )
        ).long()

        col_idx = (
            lin_idx
            + row_idx
            + 1
            - n * (n - 1) // 2
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
        row_idx, col_idx = full_idx[0], full_idx[1]
        return row_idx * n + col_idx

    @classmethod
    def _build_uppertri_bitset(cls, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        num_pairs = n * (n - 1) // 2

        present = torch.zeros(
            num_pairs,
            dtype=torch.bool,
            device=edge_index.device,
        )

        lin = cls.pairs_to_linear_uppertri(edge_index, n)

        if lin.numel() > 0:
            lin = torch.unique(lin, sorted=False)
            present[lin] = True

        return present

    # ------------------------------------------------------------------
    # PRBCD-like graph manipulation
    # ------------------------------------------------------------------

    def sample_random_block(self, n_perturbations: int = 0, mod_block_size: int = 0):
        if mod_block_size <= 0:
            mod_block_size = self.n_possible_edges

        for _ in range(self.max_final_samples):
            self.current_search_space = torch.randint(
                self.n_possible_edges,
                (mod_block_size,),
                device=self.device,
            )

            self.current_search_space = torch.unique(
                self.current_search_space,
                sorted=True,
            )

            if self.make_undirected:
                self.modified_edge_index = self.linear_to_triu_idx(
                    self.n,
                    self.current_search_space,
                )
            else:
                self.modified_edge_index = self.linear_to_full_idx(
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

            if self.current_search_space.size(0) >= n_perturbations:
                return

        raise RuntimeError(
            "Sampling random block was not successful. "
            "Please decrease n_perturbations."
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
                self.modified_edge_index = self.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = self.linear_to_full_idx(self.n, self.current_search_space)

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

    def _get_logits(self, x, edge_index, edge_weight):
        return self.attacked_model(
            data=x.to(self.device),
            adj=(edge_index.to(self.device), edge_weight.to(self.device)),
        )

    def calculate_loss(self, logits, labels):
        """
        TODO: maybe add formal definition for all losses? or maybe don't
        """
        if self.loss_type == 'CW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                    logits[np.arange(logits.size(0)), labels]
                    - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -torch.clamp(margin, min=0).mean()
        elif self.loss_type == 'LCW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                    logits[np.arange(logits.size(0)), labels]
                    - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.leaky_relu(margin, negative_slope=0.1).mean()
        elif self.loss_type == 'tanhMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                    logits[np.arange(logits.size(0)), labels]
                    - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == 'Margin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                    logits[np.arange(logits.size(0)), labels]
                    - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -margin.mean()
        elif self.loss_type.startswith('tanhMarginCW-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                    logits[np.arange(logits.size(0)), labels]
                    - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = (alpha * torch.tanh(-margin) - (1 - alpha) * torch.clamp(margin, min=0)).mean()
        elif self.loss_type.startswith('tanhMarginMCE-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'

            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                    logits[np.arange(logits.size(0)), labels]
                    - logits[np.arange(logits.size(0)), best_non_target_class]
            )

            not_flipped = logits.argmax(-1) == labels

            loss = alpha * torch.tanh(-margin).mean() + (1 - alpha) * \
                   F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'eluMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                    logits[np.arange(logits.size(0)), labels]
                    - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.elu(margin).mean()
        elif self.loss_type == 'MCE':
            not_flipped = logits.argmax(-1) == labels
            loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'NCE':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            loss = -F.cross_entropy(logits, best_non_target_class)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    def update_edge_weights(self, n_perturbations: int, epoch: int,
                            gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.n_epochs_resampling) + 1)

        self.perturbed_edge_weight.data.add_(lr * gradient)

        # We require for technical reasons that all edges in the block have at least a small positive value
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

        return self.get_modified_adj()

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

    # ------------------------------------------------------------------
    # PRBCD endpoint candidate miner
    # ------------------------------------------------------------------

    def _run_prbcd_for_endpoint_candidate_block(
            self,
            n_perturbations,
            n_candidates: int = 2_000_000,
            rng_seed: int = 0,
            epochs: int = 5,
            epochs_resampling: int | None = None,
            **kwargs,
    ) -> list[tuple[int, int]]:
        """
        Runs lightweight PRBCD-style block optimization and returns the actual
        edge flips selected by the final discrete PRBCD attack.

        Logic:
            1. initialize random candidate block
            2. optimize relaxed perturbation weights
            3. resample only during the early resampling phase
            4. fine-tune final block without further resampling
            5. sample/select final discrete attacked graph
            6. return clean-vs-attacked symmetric difference
        """
        torch.manual_seed(int(rng_seed))

        n_perturbations = int(n_perturbations)
        n_candidates = int(n_candidates)
        epochs = int(epochs)

        if epochs_resampling is None:
            # Roughly first half explores/resamples, second half fine-tunes fixed block.
            epochs_resampling = max(1, epochs // 2)
        else:
            epochs_resampling = int(epochs_resampling)

        # Safety clamp
        epochs_resampling = max(1, min(epochs_resampling, epochs))

        if n_perturbations <= 0:
            raise ValueError("n_perturbations must be > 0.")

        if n_candidates <= n_perturbations:
            raise ValueError(
                f"n_candidates must be > n_perturbations, got "
                f"n_candidates={n_candidates}, n_perturbations={n_perturbations}."
            )

        print(
            f"[endpointPRBCD] epochs={epochs}, "
            f"epochs_resampling={epochs_resampling}, "
            f"fine_tuning_epochs={epochs - epochs_resampling}"
        )

        fixed_block_path = None
        if fixed_block_path is not None:
            print("Using fixed PRBCD block:", fixed_block_path)

            block_state = torch.load(fixed_block_path, map_location="cpu")

            self.init_from_fixed_block_state(
                block_state,
                n_perturbations=n_perturbations,
            )
        else:
            print("run sampling with no certificate")
            self.sample_random_block(n_perturbations, n_candidates)

        print(self.attacked_model)

        with torch.no_grad():
            logits = self._get_logits(
                self.attr,
                self.edge_index,
                self.edge_weight,
            )

            if self.idx_attack is not None:
                loss = self.calculate_loss(
                    logits[self.idx_attack],
                    self.labels[self.idx_attack],
                )
                accuracy = utils.accuracy(
                    logits,
                    self.labels,
                    self.idx_attack,
                )
            else:
                loss = self.calculate_loss(logits, self.labels)
                accuracy = utils.accuracy(
                    logits,
                    self.labels,
                    torch.arange(self.n, device=self.device),
                )

            logging.info(
                f"\nBefore mining - Loss: {loss.item()} "
                f"Accuracy: {100 * accuracy:.3f} %\n"
            )

            del logits, loss

        for epoch in tqdm(range(epochs)):
            self.perturbed_edge_weight.requires_grad = True

            edge_index, edge_weight = self.get_modified_adj()

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logits = self._get_logits(
                self.attr,
                edge_index,
                edge_weight,
            )

            if self.idx_attack is not None:
                loss = self.calculate_loss(
                    logits[self.idx_attack],
                    self.labels[self.idx_attack],
                )
            else:
                loss = self.calculate_loss(logits, self.labels)

            gradient = utils.grad_with_checkpoint(
                loss,
                self.perturbed_edge_weight,
            )[0]

            self.gradient = gradient

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                self.update_edge_weights(
                    n_perturbations,
                    epoch,
                    gradient,
                )

                self.perturbed_edge_weight = Attack.project(
                    n_perturbations,
                    self.perturbed_edge_weight,
                    self.eps,
                )

                if epoch % self.display_step == 0:
                    edge_index_tmp, edge_weight_tmp = self.get_modified_adj()

                    logits_tmp = self.attacked_model(
                        data=self.attr.to(self.device),
                        adj=(edge_index_tmp, edge_weight_tmp),
                    )

                    if self.idx_attack is not None:
                        accuracy = utils.accuracy(
                            logits_tmp,
                            self.labels,
                            self.idx_attack,
                        )
                    else:
                        accuracy = utils.accuracy(
                            logits_tmp,
                            self.labels,
                            torch.arange(self.n, device=self.device),
                        )

                    logging.info(
                        f"\nEpoch: {epoch} Loss: {loss.item():.4f} "
                        f"Accuracy: {100 * accuracy:.3f} %\n"
                    )

                    del edge_index_tmp, edge_weight_tmp, logits_tmp

                # ----------------------------------------------------------
                # Resampling phase only:
                # Do not resample during fine-tuning.
                # Do not resample after the last resampling epoch.
                # ----------------------------------------------------------
                if epoch < epochs_resampling - 1:
                    self.resample_random_block(n_perturbations, n_candidates)

        final_edge_index, _final_edge_weight = self.sample_final_edges(
            n_perturbations,
        )

        orig_edges = _edge_set_from_edge_index(
            self.edge_index,
            undirected=True,
        )

        new_edges = _edge_set_from_edge_index(
            final_edge_index,
            undirected=True,
        )

        actual_flips = sorted(orig_edges.symmetric_difference(new_edges))

        print(
            f"[endpointPRBCD] Final discrete attack selected "
            f"{len(actual_flips)} actual flips."
        )

        return actual_flips

    # ------------------------------------------------------------------
    # Main V4-style scorer
    # ------------------------------------------------------------------

    def mine_prbcd_endpoint_candidates(
            self,
            n_perturbations,
            tag: str | None = None,
            rng_seed: int = 0,
            prev_tried_set=None,
            training_data_node_cap: int = 0,
            n_candidates: int = 2_000_000,
            prbcd_epochs: int = 5,
            n_prbcd_runs: int = 1,
            n_epochs_resampling: int = 10,
    ) -> list[tuple[int, int]]:
        """
        Run endpointPRBCD multiple times and return the union of actual flipped edges.

        Returns:
            candidates: sorted list of (u, v) pairs with u < v
        """

        if tag is None:
            tag = type(self).__name__

        training_data_node_cap = int(training_data_node_cap)
        if training_data_node_cap < 0:
            raise ValueError("training_data_node_cap must be >= 0.")

        n_prbcd_runs = int(n_prbcd_runs)
        if n_prbcd_runs <= 0:
            raise ValueError("n_prbcd_runs must be >= 1.")

        print(
            f"[{tag}] Mining actual endpointPRBCD flips: "
            f"budget={n_perturbations}, "
            f"block={n_candidates}, epochs={prbcd_epochs}, "
            f"runs={n_prbcd_runs}"
        )

        # --------------------------------------------------------------
        # Run PRBCD multiple times and collect union of actual flips
        # --------------------------------------------------------------
        candidate_set: set[tuple[int, int]] = set()

        for run_id in range(n_prbcd_runs):
            run_seed = int(rng_seed) + run_id

            print(
                f"[{tag}] PRBCD run {run_id + 1}/{n_prbcd_runs} "
                f"(seed={run_seed})"
            )

            run_candidates = self._run_prbcd_for_endpoint_candidate_block(
                rng_seed=run_seed,
                n_perturbations=n_perturbations,
                n_candidates=n_candidates,
                epochs=prbcd_epochs,
                n_epochs_resampling=self.n_epochs_resampling,
            )

            run_candidates = {
                (min(int(u), int(v)), max(int(u), int(v)))
                for u, v in run_candidates
                if int(u) != int(v)
            }

            before = len(candidate_set)
            candidate_set.update(run_candidates)
            added = len(candidate_set) - before

            print(
                f"[{tag}] Run {run_id + 1}: "
                f"{len(run_candidates)} flips, {added} new, "
                f"{len(candidate_set)} total unique"
            )

        candidates = sorted(candidate_set)

        if len(candidates) == 0:
            raise RuntimeError(f"[{tag}] endpointPRBCD produced no actual flips.")

        print(f"[{tag}] Mined {len(candidates)} unique actual PRBCD flips.")

        # --------------------------------------------------------------
        # Remove previously tried candidates
        # --------------------------------------------------------------
        if prev_tried_set:
            num_pairs = self.n * (self.n - 1) // 2

            prev_indices = {
                int(idx)
                for (idx, _drop) in prev_tried_set
                if 0 <= int(idx) < num_pairs
            }

            if prev_indices:
                before = len(candidates)
                kept_candidates = []

                for u, v in candidates:
                    uv = torch.tensor(
                        [[u], [v]],
                        device=self.device,
                        dtype=torch.long,
                    )

                    k_lin = int(
                        self.triu_idx_to_linear_idx(self.n, uv).item()
                    )

                    if k_lin not in prev_indices:
                        kept_candidates.append((u, v))

                candidates = kept_candidates

                print(
                    f"[{tag}] Previous-tried filter: "
                    f"dropped {before - len(candidates)} "
                    f"→ {len(candidates)} remaining"
                )

        if len(candidates) == 0:
            raise RuntimeError(
                f"[{tag}] endpointPRBCD flips were all previously tried."
            )

        # --------------------------------------------------------------
        # Optional node cap
        # --------------------------------------------------------------
        if training_data_node_cap > 0:
            node_counts = np.zeros(self.n, dtype=np.int64)

            before = len(candidates)
            capped_candidates: list[tuple[int, int]] = []

            for u, v in candidates:
                if (
                        node_counts[u] < training_data_node_cap
                        and node_counts[v] < training_data_node_cap
                ):
                    capped_candidates.append((u, v))
                    node_counts[u] += 1
                    node_counts[v] += 1

            candidates = capped_candidates

            print(
                f"[{tag}] Node cap: max {training_data_node_cap} candidates/node "
                f"dropped {before - len(candidates)} "
                f"→ {len(candidates)} remaining"
            )

        return sorted(candidates)

    def init_from_fixed_block_state(self, block_state: dict, n_perturbations: int = 0):
        assert int(self.n) == int(block_state["n"])
        assert int(self.block_size) == int(block_state["block_size"])
        assert bool(self.make_undirected) == bool(block_state["make_undirected"])
        assert int(self.n_possible_edges) == int(block_state["n_possible_edges"])

        self.current_search_space = block_state["current_search_space"].to(self.device).long()
        self.modified_edge_index = block_state["modified_edge_index"].to(self.device).long()

        self.perturbed_edge_weight = (
            block_state["perturbed_edge_weight"]
            .to(self.device)
            .float()
            .detach()
            .clone()
            .requires_grad_(True)
        )

        assert self.current_search_space.dim() == 1
        assert self.modified_edge_index.dim() == 2
        assert self.modified_edge_index.size(0) == 2
        assert self.modified_edge_index.size(1) == self.current_search_space.size(0)
        assert self.perturbed_edge_weight.size(0) == self.current_search_space.size(0)

        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError(
                f"Fixed block too small: "
                f"{self.current_search_space.size(0)} < {n_perturbations}"
            )

    def score_prbcd_endpoint_candidates_v4(
            self,
            candidates: list[tuple[int, int]],
            *,
            n_subsets: int | None = None,
            subset_fraction: float | None = None,
            balance_ratio: float | None = None,
            eval_idx=None,
            rng_seed: int = 0,
            store_candidates: bool | None = None,
    ):
        """
        Score mined endpointPRBCD candidates by querying the target model.

        Args:
            candidates:
                Sorted list of (u, v) pairs with u < v.

        Returns:
            src, dst, labels_out, exists
        """

        if n_subsets is None:
            n_subsets = self.n_subsets

        if subset_fraction is None:
            subset_fraction = self.subset_fraction

        if balance_ratio is None:
            balance_ratio = self.balance_ratio

        if store_candidates is None:
            store_candidates = self.store_candidates

        if n_subsets < 1:
            raise ValueError(f"n_subsets must be >= 1, got {n_subsets}")

        if not (0.0 < float(subset_fraction) <= 1.0):
            raise ValueError(
                f"subset_fraction must be in (0, 1], got {subset_fraction}"
            )

        if not candidates:
            raise ValueError("No candidates were provided for scoring.")

        # ensure sorted u < v
        candidates = sorted({
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in candidates
            if int(u) != int(v)
        })

        n_cands = len(candidates)

        ei_base = self.edge_index.to(
            device=self.device,
            dtype=torch.long,
        ).contiguous()

        ew_base = self.edge_weight.to(self.device).float().contiguous()

        present = self._build_uppertri_bitset(ei_base, self.n)

        src = torch.tensor(
            [u for u, _v in candidates],
            device=self.device,
            dtype=torch.long,
        )

        dst = torch.tensor(
            [v for _u, v in candidates],
            device=self.device,
            dtype=torch.long,
        )

        candidate_linear_idx = self.triu_idx_to_linear_idx(
            self.n,
            torch.stack([src, dst], dim=0),
        ).to(device=self.device, dtype=torch.long)

        exists = present[candidate_linear_idx].float().to(self.device)

        # directed edge lookup for deletion
        E = ei_base.size(1)

        dir_pos = torch.full(
            (self.n, self.n),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

        dir_pos[ei_base[0], ei_base[1]] = torch.arange(
            E,
            device=self.device,
            dtype=torch.int32,
        )

        def _build_perturbed_adj(flips):
            ew_use = ew_base.clone()
            del_indices = []
            add_edges = []

            for u, v, action, _k_lin in flips:
                u = int(u)
                v = int(v)

                if action == "del":
                    idx = int(dir_pos[u, v].item())
                    if idx >= 0:
                        del_indices.append(idx)

                    idx2 = int(dir_pos[v, u].item())
                    if idx2 >= 0:
                        del_indices.append(idx2)

                else:
                    add_edges.append((u, v))
                    add_edges.append((v, u))

            if del_indices:
                del_idx_tensor = torch.tensor(
                    del_indices,
                    device=self.device,
                    dtype=torch.long,
                )
                ew_use[del_idx_tensor] = 0.0

            if add_edges:
                extra_ei = torch.tensor(
                    add_edges,
                    device=self.device,
                    dtype=torch.long,
                ).t()

                extra_ew = torch.ones(
                    extra_ei.size(1),
                    device=self.device,
                    dtype=torch.float32,
                )

                ei_use = torch.cat([ei_base, extra_ei], dim=1)
                ew_use = torch.cat([ew_use, extra_ew], dim=0)
            else:
                ei_use = ei_base

            return ei_use, ew_use

        def _candidate_to_flip(i: int):
            u, v = candidates[i]
            k_lin = int(candidate_linear_idx[i].item())
            action = "del" if bool(exists[i].item()) else "add"
            return u, v, action, k_lin

        # choose evaluation nodes
        if eval_idx is None:
            if self.test_idx is not None:
                eval_idx = self.test_idx
            elif self.idx_attack is not None:
                eval_idx = self.idx_attack
            else:
                eval_idx = torch.arange(
                    self.n,
                    dtype=torch.long,
                    device=self.device,
                )

        eval_idx = torch.as_tensor(
            eval_idx,
            dtype=torch.long,
            device=self.device,
        )

        if eval_idx.numel() == 0:
            raise ValueError("eval_idx is empty; cannot compute labels.")

        y_eval = self.labels[eval_idx]

        self.attacked_model.eval()

        with torch.no_grad():
            logits_clean = self.attacked_model(
                data=self.attr,
                adj=(ei_base, ew_base),
            )

            clean_preds = logits_clean.argmax(dim=-1)

            clean_acc = float(
                (clean_preds[eval_idx] == y_eval).float().mean().item()
            )

        subset_size = max(1, round(float(subset_fraction) * n_cands))

        print(
            f"[endpointPRBCD/V4-style] Subset scoring:"
            f"  {n_subsets} subsets × {subset_size} edges/subset"
            f"  (subset_fraction={subset_fraction})"
            f"  clean_acc={clean_acc:.4f}"
        )

        rng = np.random.default_rng(int(rng_seed))
        drop_sum = np.zeros(n_cands, dtype=np.float64)
        drop_per_subset = []

        for s_i in range(int(n_subsets)):
            chosen = rng.choice(
                n_cands,
                size=min(subset_size, n_cands),
                replace=False,
            )

            batch = [_candidate_to_flip(int(i)) for i in chosen]
            ei_use, ew_use = _build_perturbed_adj(batch)

            with torch.no_grad():
                logits_pert = self.attacked_model(
                    data=self.attr,
                    adj=(ei_use, ew_use),
                )

                pert_preds = logits_pert.argmax(dim=-1)

                pert_acc = float(
                    (pert_preds[eval_idx] == y_eval).float().mean().item()
                )

            drop = max(0.0, clean_acc - pert_acc)

            drop_per_subset.append(drop)
            drop_sum[chosen] += drop

            if (s_i + 1) % 10 == 0 or s_i == int(n_subsets) - 1:
                print(
                    f"  Subset {s_i + 1:>3}/{int(n_subsets)}"
                    f"  last_drop={drop:.4f}"
                    f"  running_mean_drop={float(np.mean(drop_per_subset)):.4f}"
                )

        max_raw = float(drop_sum.max())

        if max_raw > 1e-8:
            labels_np = (drop_sum / max_raw).astype(np.float32)
        else:
            labels_np = drop_sum.astype(np.float32)

        labels_out = torch.tensor(
            labels_np,
            dtype=torch.float32,
            device=self.device,
        )

        n_pos_raw = int((labels_out > 0).sum().item())
        n_neg_raw = n_cands - n_pos_raw

        print(
            f"[endpointPRBCD/V4-style] Labels:"
            f"  {n_pos_raw}/{n_cands} edges with score > 0"
            f"  ({100 * n_pos_raw / max(1, n_cands):.1f}%)"
            f"  mean_label={float(labels_out.mean().item()) if labels_out.numel() else 0.0:.4f}"
            f"  max_raw_drop_sum={max_raw:.4f}"
        )

        if store_candidates:
            self._all_candidates = {
                "src": src.cpu().clone(),
                "dst": dst.cpu().clone(),
                "labels": labels_out.cpu().clone(),
                "exists": exists.cpu().clone(),
                "n_raw": n_cands,
                "n_pos_raw": n_pos_raw,
                "n_neg_raw": n_neg_raw,
                "drop_sum_raw": drop_sum.copy(),
                "drop_per_subset": np.array(drop_per_subset, dtype=np.float64),
                "clean_acc": clean_acc,
                "max_raw": max_raw,
                "subset_size": subset_size,
                "candidate_linear_idx": candidate_linear_idx.detach().cpu().clone(),
            }

        if balance_ratio is not None:
            pos_mask = labels_out > 0
            neg_mask = ~pos_mask

            n_pos = int(pos_mask.sum().item())
            n_neg_target = max(1, round(n_pos * float(balance_ratio)))

            if n_pos == 0:
                print(
                    "[endpointPRBCD/V4-style] No positive/scored candidates found; "
                    "skipping balancing and returning all zero-label candidates."
                )

            elif n_neg_target < int(neg_mask.sum().item()):
                g_cpu = torch.Generator(device="cpu").manual_seed(int(rng_seed))

                neg_idx = neg_mask.nonzero(as_tuple=True)[0].cpu()
                perm = torch.randperm(len(neg_idx), generator=g_cpu)[:n_neg_target]
                neg_keep = neg_idx[perm].to(device=self.device)

                pos_keep = pos_mask.nonzero(as_tuple=True)[0]

                keep = torch.cat([pos_keep, neg_keep], dim=0)

                src = src[keep]
                dst = dst[keep]
                labels_out = labels_out[keep]
                exists = exists[keep]

                print(
                    f"[endpointPRBCD/V4-style] After balancing:"
                    f"  {n_pos} scored + {n_neg_target} zero-label"
                    f"  (ratio 1:{float(balance_ratio):.1f})"
                )

        return src, dst, labels_out, exists

def train_val_test_split(
        labels,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        mode="stratified",
        seed=0,
):
    """
    Create train/val/test indices from labels.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
    """
    labels = np.asarray(labels)
    n = len(labels)

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    rng = np.random.default_rng(seed)

    if mode == "stratified":
        train_idx = []
        val_idx = []
        test_idx = []

        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)

            n_cls = len(cls_idx)
            n_train = int(round(train_size * n_cls))
            n_val = int(round(val_size * n_cls))

            train_idx.extend(cls_idx[:n_train])
            val_idx.extend(cls_idx[n_train:n_train + n_val])
            test_idx.extend(cls_idx[n_train + n_val:])

        train_idx = np.array(train_idx, dtype=np.int64)
        val_idx = np.array(val_idx, dtype=np.int64)
        test_idx = np.array(test_idx, dtype=np.int64)

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        return train_idx, val_idx, test_idx

    elif mode == "random":
        all_idx = np.arange(n)
        rng.shuffle(all_idx)

        n_train = int(round(train_size * n))
        n_val = int(round(val_size * n))

        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train:n_train + n_val]
        test_idx = all_idx[n_train + n_val:]

        return train_idx, val_idx, test_idx

    else:
        raise ValueError("mode must be 'stratified' or 'random'")

def _edge_set_from_edge_index(
        edge_index: torch.Tensor,
        *,
        undirected: bool = True,
) -> set[tuple[int, int]]:
    """
    Convert edge_index to a Python set of edge pairs.

    For undirected graphs, each edge is stored once as (u, v) with u < v.
    Self-loops are ignored.
    """
    ei = edge_index.detach().cpu().long()

    edges: set[tuple[int, int]] = set()

    for u, v in ei.t().tolist():
        u = int(u)
        v = int(v)

        if u == v:
            continue

        if undirected and u > v:
            u, v = v, u

        edges.add((u, v))

    return edges

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
    torch = _require_torch()
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=device)
    if edge_index.numel():
        adj[edge_index[0].to(device), edge_index[1].to(device)] = 1.0
    adj.fill_diagonal_(0.0)
    return torch.maximum(adj, adj.t())

def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyG attacks require torch. Install optional dependencies with `python -m pip install -e '.[pyg]'`."
        ) from exc
    return torch

def accuracy(model, x, labels, idx, edge_index=None):
    model.eval()

    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)

    device = x.device

    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    else:
        labels = labels.to(device)

    if not torch.is_tensor(idx):
        idx = torch.as_tensor(idx, dtype=torch.long, device=device)
    else:
        idx = idx.to(device)

    if edge_index is not None:
        edge_index = edge_index.to(device)

    with torch.no_grad():
        logits_tmp = model(
            data=x,
            adj=edge_index
        )
        pred = logits_tmp[idx].argmax(dim=1)
        return (pred == labels[idx]).float().mean().item()

def _edge_index_from_dense(adj):
    return (adj > 0.5).nonzero(as_tuple=False).t().contiguous()

def _confusion_counts(preds: torch.Tensor, y: torch.Tensor):
    tp = int(((preds == 1) & (y == 1)).sum().item())
    tn = int(((preds == 0) & (y == 0)).sum().item())
    fp = int(((preds == 1) & (y == 0)).sum().item())
    fn = int(((preds == 0) & (y == 1)).sum().item())
    return tp, fp, tn, fn

def _safe_auc_ap_sklearn(probs: torch.Tensor, y: torch.Tensor):
    y_np = y.detach().cpu().numpy()
    p_np = probs.detach().cpu().numpy()

    if len(set(y_np.tolist())) < 2:
        return None, None

    try:
        auc = float(roc_auc_score(y_np, p_np))
        ap = float(average_precision_score(y_np, p_np))
        return auc, ap
    except Exception:
        return None, None

import copy
import math

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train_link_prediction_gnn(
    x: torch.Tensor,
    edge_index_struct: torch.Tensor,
    edge_index_lab: torch.Tensor,
    y_label: torch.Tensor,
    device: str = "cpu",
    num_epochs: int = 200,
    hidden_dim: int = 64,
    out_dim: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    use_tqdm: bool = True,
    verbose: bool = True,
    log_every: int = 20,
    log_grad_norm: bool = False,
    csv_path: str | None = None,
    csv_append: bool = False,
    # ---- optional auxiliary endpoint supervision ----
    y_src_label: torch.Tensor | None = None,
    y_dst_label: torch.Tensor | None = None,
    aux_loss_weight: float = 0.5,
    # ---- early stopping / best checkpoint ----
    min_epochs_before_early_stop: int = 40,
    early_stop: bool = True,
    early_stop_metric: str = "val_ap",
    # allowed:
    # classification/thresholded metrics:
    # "val_auc" | "val_ap" | "val_acc" |
    # "val_f1" | "val_recall"
    # continuous-target metrics:
    # "val_loss" | "val_mae" | "val_mse" |
    # "val_rmse" | "val_r2" | "val_pearson"
    early_stop_patience: int = 15,
    early_stop_min_delta: float = 1e-4,
    restore_best: bool = True,
    # ---- split ratios ----
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    threshold: float = 0.5,
):
    """Train a binary link-prediction GNN with hard or soft targets.

    ``y_label`` may contain either hard binary labels {0, 1} or soft labels
    anywhere in [0, 1]. In both cases the edge head must return one logit per
    labeled edge and training uses ``BCEWithLogitsLoss``.

    For soft labels, MAE/MSE/RMSE/R2/Pearson are computed directly against the
    sigmoid probabilities. Binary metrics are still available, but they use
    ``threshold`` to convert both targets and probabilities to hard classes.

    The auxiliary source/destination labels, when supplied, may also be hard or
    soft values in [0, 1].
    """
    import copy
    import math

    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tqdm import tqdm

    # ======================================================
    # Validate configuration
    # ======================================================

    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0.")

    if min_epochs_before_early_stop < 1:
        raise ValueError(
            "min_epochs_before_early_stop must be at least 1."
        )

    if early_stop_patience < 1:
        raise ValueError(
            "early_stop_patience must be at least 1."
        )

    if early_stop_min_delta < 0:
        raise ValueError(
            "early_stop_min_delta must be non-negative."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between 0 and 1.")

    if aux_loss_weight < 0:
        raise ValueError("aux_loss_weight must be non-negative.")

    if (y_src_label is None) != (y_dst_label is None):
        raise ValueError(
            "y_src_label and y_dst_label must either both be "
            "provided or both be None."
        )

    # Backward-compatible aliases.
    metric_aliases = {
        "auc": "val_auc",
        "ap": "val_ap",
        "acc": "val_acc",
        "f1": "val_f1",
        "recall": "val_recall",
        "loss": "val_loss",
        "mae": "val_mae",
        "mse": "val_mse",
        "rmse": "val_rmse",
        "r2": "val_r2",
        "pearson": "val_pearson",
    }
    early_stop_metric = metric_aliases.get(
        early_stop_metric,
        early_stop_metric,
    )

    metric_mode = {
        "val_auc": "max",
        "val_ap": "max",
        "val_acc": "max",
        "val_f1": "max",
        "val_recall": "max",
        "val_loss": "min",
        "val_mae": "min",
        "val_mse": "min",
        "val_rmse": "min",
        "val_r2": "max",
        "val_pearson": "max",
    }

    if early_stop_metric not in metric_mode:
        raise ValueError(
            f"early_stop_metric must be one of "
            f"{list(metric_mode.keys())}, or one of the aliases "
            f"{list(metric_aliases.keys())}."
        )

    ratio_sum = train_ratio + val_ratio + test_ratio

    if abs(ratio_sum - 1.0) >= 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0."
        )

    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError(
            "train_ratio, val_ratio and test_ratio must all be positive."
        )

    if log_every < 1:
        raise ValueError("log_every must be at least 1.")

    # ======================================================
    # Move tensors to device and validate targets
    # ======================================================

    x = x.to(device)
    edge_index_struct = edge_index_struct.long().to(device)
    edge_index_lab = edge_index_lab.long().to(device)
    y_label = y_label.float().view(-1).to(device)

    if y_src_label is not None:
        y_src_label = y_src_label.float().view(-1).to(device)

    if y_dst_label is not None:
        y_dst_label = y_dst_label.float().view(-1).to(device)

    if edge_index_struct.ndim != 2 or edge_index_struct.size(0) != 2:
        raise ValueError(
            "edge_index_struct must have shape (2, E)."
        )

    if edge_index_lab.ndim != 2 or edge_index_lab.size(0) != 2:
        raise ValueError(
            "edge_index_lab must have shape (2, M)."
        )

    M = edge_index_lab.size(1)

    def _validate_unit_interval(
        values: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"{name} contains NaN or infinite values.")

        tolerance = 1e-7
        min_value = float(values.min().item()) if values.numel() else 0.0
        max_value = float(values.max().item()) if values.numel() else 1.0

        if min_value < -tolerance or max_value > 1.0 + tolerance:
            raise ValueError(
                f"{name} must contain values in [0, 1]. "
                f"Found range [{min_value}, {max_value}]."
            )

        # Remove harmless floating-point spillover such as 1.00000001.
        return values.clamp(0.0, 1.0)

    y_label = _validate_unit_interval(y_label, "y_label")

    if y_src_label is not None:
        y_src_label = _validate_unit_interval(
            y_src_label,
            "y_src_label",
        )

    if y_dst_label is not None:
        y_dst_label = _validate_unit_interval(
            y_dst_label,
            "y_dst_label",
        )

    def _contains_only_hard_labels(values: torch.Tensor) -> bool:
        if values.numel() == 0:
            return True

        is_zero = torch.isclose(
            values,
            torch.zeros_like(values),
            atol=1e-7,
            rtol=0.0,
        )
        is_one = torch.isclose(
            values,
            torch.ones_like(values),
            atol=1e-7,
            rtol=0.0,
        )
        return bool((is_zero | is_one).all().item())

    hard_label_mode = _contains_only_hard_labels(y_label)
    label_mode = "hard_binary" if hard_label_mode else "soft_binary"

    # ======================================================
    # Empty input
    # ======================================================

    if M == 0:
        if verbose:
            print(
                "[LP-GNN] No labeled pairs. Returning untrained model."
            )

        model = LinkPredictionGNN(
            in_dim=x.size(1),
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        ).to(device)

        model.training_history = {
            "train_losses": [],
            "train_objective_losses": [],
            "val_losses": [],
            "test_losses": [],
            "val_ap": [],
            "val_auc": [],
            "val_f1": [],
            "val_mae": [],
            "val_mse": [],
            "val_rmse": [],
            "val_r2": [],
            "val_pearson": [],
            "early_stop_metric_history": [],
            "patience_history": [],
            "stopped_at": 0,
            "best_epoch": None,
            "best_metric": None,
            "early_stopped": False,
            "early_stop_enabled": early_stop,
            "early_stop_metric": early_stop_metric,
            "early_stop_patience": early_stop_patience,
            "min_epochs_before_early_stop": (
                min_epochs_before_early_stop
            ),
            "restored_best": False,
            "stop_reason": "no_labeled_pairs",
            "train_idx": torch.empty(0, dtype=torch.long),
            "val_idx": torch.empty(0, dtype=torch.long),
            "test_idx": torch.empty(0, dtype=torch.long),
            "label_mode": label_mode,
            "threshold": float(threshold),
            "split_strategy": None,
        }

        return model

    if M != y_label.numel():
        raise ValueError(
            "edge_index_lab and y_label must contain the same "
            f"number of examples, got {M} and {y_label.numel()}."
        )

    if y_src_label is not None and M != y_src_label.numel():
        raise ValueError(
            "edge_index_lab and y_src_label must contain the "
            "same number of examples."
        )

    if y_dst_label is not None and M != y_dst_label.numel():
        raise ValueError(
            "edge_index_lab and y_dst_label must contain the "
            "same number of examples."
        )

    use_aux = (
        y_src_label is not None
        and y_dst_label is not None
    )

    # Soft targets are thresholded only for stratification and binary metrics.
    y_hard = (y_label >= threshold).long()
    n_neg = int((y_hard == 0).sum().item())
    n_pos = int((y_hard == 1).sum().item())

    if verbose:
        if hard_label_mode:
            msg = (
                f"[LP-GNN] Labeled pairs: M={M} | mode=hard_binary | "
                f"pos={n_pos} | neg={n_neg}"
            )
        else:
            msg = (
                f"[LP-GNN] Labeled pairs: M={M} | mode=soft_binary | "
                f"target_min={y_label.min().item():.4f} | "
                f"target_mean={y_label.mean().item():.4f} | "
                f"target_max={y_label.max().item():.4f} | "
                f"thresholded_pos={n_pos} | thresholded_neg={n_neg}"
            )

        if use_aux:
            src_mode = (
                "hard"
                if _contains_only_hard_labels(y_src_label)
                else "soft"
            )
            dst_mode = (
                "hard"
                if _contains_only_hard_labels(y_dst_label)
                else "soft"
            )
            msg += (
                f" | src_labels={src_mode} | dst_labels={dst_mode} "
                f"| aux_loss_weight={aux_loss_weight}"
            )

        print(msg)

    # ======================================================
    # Train / validation / test split
    # ======================================================

    split_generator = torch.Generator(
        device="cpu"
    ).manual_seed(42)

    def _split_counts(count: int) -> tuple[int, int, int]:
        n_train_local = int(train_ratio * count)
        n_val_local = int(val_ratio * count)
        n_test_local = count - n_train_local - n_val_local
        return n_train_local, n_val_local, n_test_local

    def _split_one_group(
        indices: torch.Tensor,
        group_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = int(indices.numel())
        n_train_local, n_val_local, n_test_local = _split_counts(count)

        if min(n_train_local, n_val_local, n_test_local) <= 0:
            raise ValueError(
                f"Group '{group_name}' needs at least one example in "
                "train, validation and test. "
                f"Group size={count}, resulting split="
                f"{n_train_local}/{n_val_local}/{n_test_local}."
            )

        order = torch.randperm(
            count,
            generator=split_generator,
        ).to(device)
        shuffled = indices[order]

        train_local = shuffled[:n_train_local]
        val_local = shuffled[
            n_train_local:n_train_local + n_val_local
        ]
        test_local = shuffled[n_train_local + n_val_local:]

        return train_local, val_local, test_local

    classification_metric_names = {
        "val_auc",
        "val_ap",
        "val_acc",
        "val_f1",
        "val_recall",
    }

    threshold_groups_can_be_stratified = False
    if n_neg > 0 and n_pos > 0:
        neg_counts = _split_counts(n_neg)
        pos_counts = _split_counts(n_pos)
        threshold_groups_can_be_stratified = (
            min(*neg_counts, *pos_counts) > 0
        )

    if threshold_groups_can_be_stratified:
        neg_idx_all = torch.nonzero(
            y_hard == 0,
            as_tuple=True,
        )[0]
        pos_idx_all = torch.nonzero(
            y_hard == 1,
            as_tuple=True,
        )[0]

        neg_train, neg_val, neg_test = _split_one_group(
            neg_idx_all,
            "target<threshold",
        )
        pos_train, pos_val, pos_test = _split_one_group(
            pos_idx_all,
            "target>=threshold",
        )

        train_idx = torch.cat([neg_train, pos_train])
        val_idx = torch.cat([neg_val, pos_val])
        test_idx = torch.cat([neg_test, pos_test])
        split_strategy = "threshold_stratified"

    else:
        if hard_label_mode:
            if n_neg == 0 or n_pos == 0:
                raise ValueError(
                    f"Hard binary training requires both classes, got "
                    f"pos={n_pos}, neg={n_neg}."
                )

            raise ValueError(
                "Each hard class needs enough examples to place at least "
                "one item in train, validation and test. "
                f"Got pos={n_pos}, neg={n_neg}."
            )

        if early_stop_metric in classification_metric_names:
            if n_neg == 0 or n_pos == 0:
                raise ValueError(
                    f"{early_stop_metric} requires soft targets on both "
                    f"sides of threshold={threshold}, but got "
                    f"thresholded_pos={n_pos}, thresholded_neg={n_neg}. "
                    "Use early_stop_metric='val_loss', 'val_mae', "
                    "'val_rmse', 'val_r2', or 'val_pearson'."
                )

            raise ValueError(
                f"{early_stop_metric} requires enough thresholded positive "
                "and negative targets for all three splits. "
                f"Got thresholded_pos={n_pos}, thresholded_neg={n_neg}. "
                "Use a continuous early-stopping metric or provide more data."
            )

        n_train, n_val, n_test = _split_counts(M)

        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(
                "Not enough labeled pairs for train, validation and test. "
                f"M={M}, resulting split={n_train}/{n_val}/{n_test}."
            )

        # Rank-bin stratification preserves the soft-target distribution better
        # than a purely random split. Each bin is split independently.
        min_bin_size = max(
            3,
            math.ceil(1.0 / min(train_ratio, val_ratio, test_ratio)),
        )
        num_bins = max(1, min(10, M // min_bin_size))

        random_tiebreak = torch.rand(
            M,
            generator=split_generator,
        )
        # Stable sorting is not available in all supported torch versions;
        # a tiny random jitter only determines ordering among near-equal values.
        y_for_sort = y_label.detach().cpu() + 1e-12 * random_tiebreak
        sorted_cpu_idx = torch.argsort(y_for_sort)
        rank_bins = torch.tensor_split(sorted_cpu_idx, num_bins)

        train_parts: list[torch.Tensor] = []
        val_parts: list[torch.Tensor] = []
        test_parts: list[torch.Tensor] = []

        for bin_number, bin_cpu_idx in enumerate(rank_bins):
            bin_idx = bin_cpu_idx.to(device)
            count = int(bin_idx.numel())

            if count == 0:
                continue

            bin_train, bin_val, bin_test = _split_one_group(
                bin_idx,
                f"soft_rank_bin_{bin_number}",
            )
            train_parts.append(bin_train)
            val_parts.append(bin_val)
            test_parts.append(bin_test)

        train_idx = torch.cat(train_parts)
        val_idx = torch.cat(val_parts)
        test_idx = torch.cat(test_parts)
        split_strategy = f"soft_rank_bins_{num_bins}"

    # Shuffle final indices so examples are not grouped by stratum/bin.
    train_idx = train_idx[
        torch.randperm(
            train_idx.numel(),
            generator=split_generator,
        ).to(device)
    ]
    val_idx = val_idx[
        torch.randperm(
            val_idx.numel(),
            generator=split_generator,
        ).to(device)
    ]
    test_idx = test_idx[
        torch.randperm(
            test_idx.numel(),
            generator=split_generator,
        ).to(device)
    ]

    if train_idx.numel() + val_idx.numel() + test_idx.numel() != M:
        raise AssertionError("Split sizes do not add up to M.")

    if verbose:
        print(
            f"[LP-GNN] Split strategy={split_strategy} | "
            f"train={train_idx.numel()} | val={val_idx.numel()} | "
            f"test={test_idx.numel()}"
        )

    # ======================================================
    # Model and losses
    # ======================================================

    model = LinkPredictionGNN(
        in_dim=x.size(1),
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    def _effective_pos_weight(targets: torch.Tensor) -> float:
        """Return neg/pos mass; identical to count ratio for hard labels."""
        positive_mass = float(targets.sum().item())
        negative_mass = float((1.0 - targets).sum().item())

        if positive_mass <= 1e-12 or negative_mass <= 1e-12:
            return 1.0

        return negative_mass / positive_mass

    train_targets = y_label[train_idx]

    if hard_label_mode:
        pos_weight = _effective_pos_weight(train_targets)
    else:
        pos_weight = 1.0

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    if use_aux:
        src_train = y_src_label[train_idx]
        dst_train = y_dst_label[train_idx]

        src_pos_weight = _effective_pos_weight(src_train)
        dst_pos_weight = _effective_pos_weight(dst_train)

        loss_fn_src = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                src_pos_weight,
                dtype=torch.float32,
                device=device,
            )
        )
        loss_fn_dst = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                dst_pos_weight,
                dtype=torch.float32,
                device=device,
            )
        )
    else:
        src_pos_weight = None
        dst_pos_weight = None

    # ======================================================
    # Evaluation helpers
    # ======================================================

    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else 0.0

    def _confusion_counts_local(
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        predictions = predictions.long().view(-1)
        labels = labels.long().view(-1)

        tp = int(((predictions == 1) & (labels == 1)).sum().item())
        fp = int(((predictions == 1) & (labels == 0)).sum().item())
        tn = int(((predictions == 0) & (labels == 0)).sum().item())
        fn = int(((predictions == 0) & (labels == 1)).sum().item())
        return tp, fp, tn, fn

    def _safe_auc_ap(
        probabilities: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float | None, float | None]:
        probabilities_np = probabilities.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        if len(set(labels_np.tolist())) < 2:
            return None, None

        try:
            auc = float(roc_auc_score(labels_np, probabilities_np))
        except ValueError:
            auc = None

        try:
            ap = float(average_precision_score(labels_np, probabilities_np))
        except ValueError:
            ap = None

        return auc, ap

    def _compute_metrics_from_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        probs = torch.sigmoid(logits)
        hard_targets = (targets >= threshold).long()
        predictions = (probs >= threshold).long()

        loss_val = loss_fn(logits, targets)

        errors = probs - targets
        mae = torch.mean(torch.abs(errors))
        mse = torch.mean(errors.square())
        rmse = torch.sqrt(mse)

        target_centered = targets - targets.mean()
        prob_centered = probs - probs.mean()

        ss_res = torch.sum(errors.square())
        ss_tot = torch.sum(target_centered.square())

        if float(ss_tot.item()) > 1e-12:
            r2: float | None = float((1.0 - ss_res / ss_tot).item())
        else:
            r2 = None

        pearson_denominator = torch.sqrt(
            torch.sum(target_centered.square())
            * torch.sum(prob_centered.square())
        )

        if float(pearson_denominator.item()) > 1e-12:
            pearson: float | None = float(
                (
                    torch.sum(target_centered * prob_centered)
                    / pearson_denominator
                ).item()
            )
        else:
            pearson = None

        tp, fp, tn, fn = _confusion_counts_local(
            predictions,
            hard_targets,
        )

        total = tp + fp + tn + fn
        acc = _safe_div(tp + tn, total)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        auc, ap = _safe_auc_ap(probs, hard_targets)

        return {
            "loss": float(loss_val.item()),
            "mae": float(mae.item()),
            "mse": float(mse.item()),
            "rmse": float(rmse.item()),
            "r2": r2,
            "pearson": pearson,
            # For soft targets, the following are thresholded metrics.
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "auc": auc,
            "ap": ap,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "target_min": float(targets.min().item()),
            "target_mean": float(targets.mean().item()),
            "target_max": float(targets.max().item()),
            "p_min": float(probs.min().item()),
            "p_mean": float(probs.mean().item()),
            "p_max": float(probs.max().item()),
        }

    def _evaluate_split(
        split_idx: torch.Tensor,
    ) -> dict | None:
        model.eval()

        with torch.no_grad():
            if use_aux:
                output = model(
                    x,
                    edge_index_struct,
                    edge_index_lab[:, split_idx],
                    return_aux=True,
                )
                logits = output["edge_logits"].view(-1)
            else:
                logits = model(
                    x,
                    edge_index_struct,
                    edge_index_lab[:, split_idx],
                ).view(-1)

            if (
                torch.isnan(logits).any()
                or torch.isinf(logits).any()
            ):
                return None

            targets = y_label[split_idx]
            return _compute_metrics_from_logits(logits, targets)

    # ======================================================
    # Early stopping state
    # ======================================================

    want = metric_mode[early_stop_metric]
    best_metric = -float("inf") if want == "max" else float("inf")
    best_epoch = -1
    best_state = None
    patience_left = int(early_stop_patience)

    stopped_epoch = 0
    stopped_metric_value = None
    early_stopped = False
    stop_reason = "completed"

    def _is_improvement(current: float, best: float) -> bool:
        if want == "max":
            return current > best + early_stop_min_delta
        return current < best - early_stop_min_delta

    # ======================================================
    # Histories
    # ======================================================

    train_losses: list[float] = []
    val_losses: list[float] = []
    test_losses: list[float] = []
    train_objective_losses: list[float] = []

    val_ap_history: list[float | None] = []
    val_auc_history: list[float | None] = []
    val_f1_history: list[float] = []
    val_mae_history: list[float] = []
    val_mse_history: list[float] = []
    val_rmse_history: list[float] = []
    val_r2_history: list[float | None] = []
    val_pearson_history: list[float | None] = []

    early_stop_metric_history: list[float] = []
    patience_history: list[int] = []

    epoch_iter = (
        tqdm(range(num_epochs), desc="[LP-GNN] Training")
        if use_tqdm
        else range(num_epochs)
    )

    # ======================================================
    # Training loop
    # ======================================================

    for epoch in epoch_iter:
        current_epoch = epoch + 1

        model.train()
        optimizer.zero_grad()

        if use_aux:
            out_train = model(
                x,
                edge_index_struct,
                edge_index_lab[:, train_idx],
                return_aux=True,
            )

            logits_train = out_train["edge_logits"].view(-1)

            if (
                torch.isnan(logits_train).any()
                or torch.isinf(logits_train).any()
            ):
                print(
                    "[LP-GNN][ERROR] NaN/Inf in train edge logits "
                    f"at epoch {current_epoch}."
                )
                stop_reason = "non_finite_train_logits"
                break

            loss_edge = loss_fn(
                logits_train,
                y_label[train_idx],
            )
            loss_src = loss_fn_src(
                out_train["src_flip_logits"].view(-1),
                y_src_label[train_idx],
            )
            loss_dst = loss_fn_dst(
                out_train["dst_flip_logits"].view(-1),
                y_dst_label[train_idx],
            )

            loss = (
                loss_edge
                + aux_loss_weight * (loss_src + loss_dst)
            )

        else:
            logits_train = model(
                x,
                edge_index_struct,
                edge_index_lab[:, train_idx],
            ).view(-1)

            if (
                torch.isnan(logits_train).any()
                or torch.isinf(logits_train).any()
            ):
                print(
                    "[LP-GNN][ERROR] NaN/Inf in train logits "
                    f"at epoch {current_epoch}."
                )
                stop_reason = "non_finite_train_logits"
                break

            # Preserve the old hard-label smoothing behavior, but do not
            # smooth labels that are already soft.
            if hard_label_mode:
                label_smoothing = 0.1
                train_targets_for_loss = (
                    y_label[train_idx] * (1.0 - label_smoothing)
                    + 0.5 * label_smoothing
                )
            else:
                train_targets_for_loss = y_label[train_idx]

            loss = loss_fn(
                logits_train,
                train_targets_for_loss,
            )

        if not bool(torch.isfinite(loss).item()):
            print(
                "[LP-GNN][ERROR] Non-finite training loss "
                f"at epoch {current_epoch}."
            )
            stop_reason = "non_finite_train_loss"
            break

        loss.backward()

        grad_norm_val = None
        if log_grad_norm:
            total_norm_squared = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter_norm = (
                        parameter.grad.detach().norm(2).item()
                    )
                    total_norm_squared += parameter_norm ** 2
            grad_norm_val = total_norm_squared ** 0.5

        optimizer.step()

        # ==================================================
        # Evaluate train / validation / test
        # ==================================================

        train_metrics = _evaluate_split(train_idx)
        val_metrics = _evaluate_split(val_idx)
        test_metrics = _evaluate_split(test_idx)

        if train_metrics is None:
            print(
                "[LP-GNN][ERROR] NaN/Inf during train evaluation "
                f"at epoch {current_epoch}."
            )
            stop_reason = "non_finite_train_evaluation"
            break

        if val_metrics is None:
            print(
                "[LP-GNN][ERROR] NaN/Inf during validation evaluation "
                f"at epoch {current_epoch}."
            )
            stop_reason = "non_finite_validation_evaluation"
            break

        if test_metrics is None:
            print(
                "[LP-GNN][ERROR] NaN/Inf during test evaluation "
                f"at epoch {current_epoch}."
            )
            stop_reason = "non_finite_test_evaluation"
            break

        train_losses.append(float(train_metrics["loss"]))
        val_losses.append(float(val_metrics["loss"]))
        test_losses.append(float(test_metrics["loss"]))
        train_objective_losses.append(float(loss.item()))

        metric_key = early_stop_metric.replace("val_", "", 1)
        metric_val_raw = val_metrics[metric_key]

        if metric_val_raw is None:
            raise RuntimeError(
                f"{early_stop_metric} is unavailable at epoch "
                f"{current_epoch}. This usually means that the validation "
                "targets are constant for this metric. Choose val_loss, "
                "val_mae, val_mse, or val_rmse instead."
            )

        metric_val = float(metric_val_raw)

        if not math.isfinite(metric_val):
            raise RuntimeError(
                f"Non-finite {early_stop_metric} at epoch "
                f"{current_epoch}: {metric_val}"
            )

        stopped_epoch = current_epoch
        stopped_metric_value = metric_val

        # ==================================================
        # Best checkpoint and patience
        # ==================================================

        is_best = False

        if _is_improvement(metric_val, best_metric):
            best_metric = metric_val
            best_epoch = current_epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = int(early_stop_patience)
            is_best = True
        elif (
            early_stop
            and current_epoch >= min_epochs_before_early_stop
        ):
            patience_left -= 1

        val_ap_history.append(val_metrics["ap"])
        val_auc_history.append(val_metrics["auc"])
        val_f1_history.append(float(val_metrics["f1"]))
        val_mae_history.append(float(val_metrics["mae"]))
        val_mse_history.append(float(val_metrics["mse"]))
        val_rmse_history.append(float(val_metrics["rmse"]))
        val_r2_history.append(val_metrics["r2"])
        val_pearson_history.append(val_metrics["pearson"])
        early_stop_metric_history.append(metric_val)
        patience_history.append(int(patience_left))

        # ==================================================
        # tqdm output
        # ==================================================

        if use_tqdm:
            postfix = {
                "tr_bce": f"{train_metrics['loss']:.4f}",
                "tr_obj": f"{loss.item():.4f}",
                "va_loss": f"{val_metrics['loss']:.4f}",
                "va_mae": f"{val_metrics['mae']:.3f}",
                "va_rmse": f"{val_metrics['rmse']:.3f}",
                "va_ap": (
                    f"{val_metrics['ap']:.3f}"
                    if val_metrics["ap"] is not None
                    else "n/a"
                ),
                "va_auc": (
                    f"{val_metrics['auc']:.3f}"
                    if val_metrics["auc"] is not None
                    else "n/a"
                ),
                "va_f1": f"{val_metrics['f1']:.3f}",
                "pat": patience_left if early_stop else "off",
            }
            epoch_iter.set_postfix(postfix)

        # ==================================================
        # Console output
        # ==================================================

        if verbose and (
            current_epoch % log_every == 0
            or current_epoch == 1
            or current_epoch == num_epochs
        ):
            threshold_metric_prefix = (
                ""
                if hard_label_mode
                else f"thr@{threshold:.2f}_"
            )

            message = (
                f"[LP-GNN] Epoch {current_epoch:03d}/{num_epochs} | "
                f"train_bce={train_metrics['loss']:.4f} | "
                f"train_objective={loss.item():.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"{threshold_metric_prefix}train_acc="
                f"{train_metrics['acc']:.4f} | "
                f"{threshold_metric_prefix}val_acc="
                f"{val_metrics['acc']:.4f} | "
                f"{threshold_metric_prefix}test_acc="
                f"{test_metrics['acc']:.4f} | "
                f"{threshold_metric_prefix}val_precision="
                f"{val_metrics['precision']:.4f} | "
                f"{threshold_metric_prefix}val_recall="
                f"{val_metrics['recall']:.4f} | "
                f"{threshold_metric_prefix}val_f1="
                f"{val_metrics['f1']:.4f} | "
                f"val_TP/FP/TN/FN="
                f"{val_metrics['tp']}/{val_metrics['fp']}/"
                f"{val_metrics['tn']}/{val_metrics['fn']}"
            )

            if val_metrics["r2"] is not None:
                message += f" | val_R2={val_metrics['r2']:.4f}"

            if val_metrics["pearson"] is not None:
                message += (
                    f" | val_Pearson={val_metrics['pearson']:.4f}"
                )

            if val_metrics["auc"] is not None:
                message += f" | val_AUC={val_metrics['auc']:.4f}"

            if val_metrics["ap"] is not None:
                message += f" | val_AP={val_metrics['ap']:.4f}"

            message += (
                f" | best_{early_stop_metric}={best_metric:.4f} "
                f"(epoch {best_epoch})"
            )

            if early_stop:
                message += f" | patience_left={patience_left}"

            if grad_norm_val is not None:
                message += f" | grad_norm={grad_norm_val:.3e}"

            if use_aux:
                message += " | multitask_aux=on"

            if is_best:
                message += " | new_best"

            print(message)

        # ==================================================
        # Early stopping
        # ==================================================

        if (
            early_stop
            and current_epoch >= min_epochs_before_early_stop
            and patience_left <= 0
        ):
            early_stopped = True
            stop_reason = "early_stopping"

            if verbose:
                print(
                    f"[LP-GNN] Early stopping at epoch "
                    f"{current_epoch}. Best "
                    f"{early_stop_metric}={best_metric:.4f} "
                    f"at epoch {best_epoch}."
                )
            break

    # ======================================================
    # Validate early-stopping bookkeeping
    # ======================================================

    stopped_at = len(train_losses)

    assert stopped_at == stopped_epoch, (
        "Internal epoch bookkeeping mismatch: "
        f"history contains {stopped_at} epochs, "
        f"but stopped_epoch={stopped_epoch}."
    )

    expected_history_length = stopped_at
    histories_to_check = {
        "val_losses": val_losses,
        "test_losses": test_losses,
        "train_objective_losses": train_objective_losses,
        "val_ap": val_ap_history,
        "val_auc": val_auc_history,
        "val_f1": val_f1_history,
        "val_mae": val_mae_history,
        "val_mse": val_mse_history,
        "val_rmse": val_rmse_history,
        "val_r2": val_r2_history,
        "val_pearson": val_pearson_history,
        "early_stop_metric_history": early_stop_metric_history,
        "patience_history": patience_history,
    }

    for history_name, history_values in histories_to_check.items():
        assert len(history_values) == expected_history_length, (
            f"{history_name} does not match the number of completed epochs."
        )

    if stopped_at > 0:
        assert best_state is not None, (
            "At least one epoch completed but no best checkpoint was recorded."
        )
        assert 1 <= best_epoch <= stopped_at, (
            f"Invalid best_epoch={best_epoch} for stopped_at={stopped_at}."
        )
        assert math.isfinite(best_metric), (
            "The recorded best metric is not finite."
        )

    if early_stopped:
        assert early_stop
        assert stopped_at >= min_epochs_before_early_stop
        assert patience_left <= 0
        assert stop_reason == "early_stopping"

    if not early_stop:
        assert not early_stopped

    # ======================================================
    # Restore best validation checkpoint
    # ======================================================

    restored_best = False

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        restored_best = True

        if verbose:
            print(
                f"[LP-GNN] Restored best model from epoch "
                f"{best_epoch} ({early_stop_metric}="
                f"{best_metric:.4f})."
            )

    # ======================================================
    # Attach history to model
    # ======================================================

    model.training_history = {
        "train_losses": train_losses,
        "train_objective_losses": train_objective_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "val_ap": val_ap_history,
        "val_auc": val_auc_history,
        "val_f1": val_f1_history,
        "val_mae": val_mae_history,
        "val_mse": val_mse_history,
        "val_rmse": val_rmse_history,
        "val_r2": val_r2_history,
        "val_pearson": val_pearson_history,
        "early_stop_metric_history": early_stop_metric_history,
        "patience_history": patience_history,
        "stopped_at": stopped_at,
        "best_epoch": best_epoch if best_epoch >= 1 else None,
        "best_metric": (
            float(best_metric) if best_epoch >= 1 else None
        ),
        "early_stopped": early_stopped,
        "early_stop_enabled": early_stop,
        "early_stop_metric": early_stop_metric,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "min_epochs_before_early_stop": min_epochs_before_early_stop,
        "stopped_metric_value": stopped_metric_value,
        "restored_best": restored_best,
        "stop_reason": stop_reason,
        "train_idx": train_idx.detach().cpu(),
        "val_idx": val_idx.detach().cpu(),
        "test_idx": test_idx.detach().cpu(),
        "positive_class_weight": float(pos_weight),
        "src_positive_class_weight": (
            float(src_pos_weight) if src_pos_weight is not None else None
        ),
        "dst_positive_class_weight": (
            float(dst_pos_weight) if dst_pos_weight is not None else None
        ),
        "label_mode": label_mode,
        "hard_label_mode": hard_label_mode,
        "threshold": float(threshold),
        "split_strategy": split_strategy,
        "csv_path": csv_path,
        "csv_append": csv_append,
    }

    if verbose:
        print(
            f"[LP-GNN] Training complete after {stopped_at} epoch(s). "
            f"stop_reason={stop_reason} | "
            f"label_mode={label_mode} | "
            f"best_epoch={model.training_history['best_epoch']} | "
            f"best_{early_stop_metric}="
            f"{model.training_history['best_metric']}."
        )

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
    mode: Literal[
        "subset_accuracy_drop",
        "endpoint",
        "two_hop_correct_to_incorrect",
    ] = "subset_accuracy_drop",
    subset_fraction: float = 0.1,
    n_subsets: int = 100,

    # Endpoint sampling parameters
    endpoint_k_samples: int = 1000,
    endpoint_require_correct_to_incorrect: bool = True,
    endpoint_max_sampling_tries: int | None = None,

    # Two-hop neighborhood mining parameters
    two_hop_k_samples: int | None = None,

    seed: int = 0,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Mine supervision scores for candidate edge flips.

    Modes
    -----
    subset_accuracy_drop:
        Random subsets of the supplied candidate edges are flipped
        simultaneously. The observed evaluation-accuracy drop is assigned
        to all selected edges.

    endpoint:
        Builds a node pool from all nodes occurring in `candidates`.
        It then randomly samples unique undirected pairs from all possible
        pairs among those nodes.

        Each sampled pair is flipped individually:

            existing edge     -> removed
            non-existing edge -> added

        After a forward pass through the victim model, only the predictions
        of the two endpoints are evaluated.

    two_hop_correct_to_incorrect:
        Uses the supplied candidate edges directly.

        Each candidate edge is flipped individually. For an edge (u, v), the
        evaluated node set is the union of the clean two-hop neighborhoods
        of u and v, including the endpoints themselves.

        The raw score assigned to the candidate edge is the number of nodes
        in that neighborhood that:

            1. were correctly classified on the clean graph, and
            2. are incorrectly classified after flipping the edge.

        The raw scores are min-max normalized across all evaluated candidate
        edges to produce `labels_norm`.

    Parameters
    ----------
    endpoint_k_samples:
        Number of unique random node pairs to evaluate in endpoint mode.

    endpoint_require_correct_to_incorrect:
        If True, an endpoint hit requires that the endpoint was correctly
        classified before the flip and incorrectly classified afterward.

        If False, any change in the endpoint's predicted class is counted.

    endpoint_max_sampling_tries:
        Maximum number of attempts used to collect unique random pairs.
        If None, a suitable value is selected automatically.

    two_hop_k_samples:
        Maximum number of supplied candidate edges to evaluate in
        two-hop mode.

        If None, all unique supplied candidate edges are evaluated.
        If an integer is supplied, that many unique candidate edges are
        sampled without replacement.
    """
    if device is None:
        device = attr.device

    if not candidates:
        raise ValueError("candidates must not be empty.")

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

    if (
        (cand_src < 0).any()
        or (cand_src >= n_nodes).any()
        or (cand_dst < 0).any()
        or (cand_dst >= n_nodes).any()
    ):
        raise ValueError(
            "At least one candidate contains an invalid node ID."
        )

    clean_logits = model(attr, adj_orig)
    clean_preds = clean_logits.argmax(dim=-1)
    clean_correct = clean_preds == labels

    clean_accuracy = float(
        (
            clean_preds[eval_idx]
            == labels[eval_idx]
        )
        .float()
        .mean()
        .item()
    )

    if mode == "subset_accuracy_drop":
        # The original candidate edges are used directly.
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
            verbose=verbose,
        )

    if mode == "endpoint":
        # cand_src and cand_dst define a candidate-node pool.
        # The endpoint miner samples new random pairs from that pool.
        if endpoint_k_samples is None:
            endpoint_k_samples = len(candidates)
        return _mine_endpoint_flips(
            model=model,
            attr=attr,
            labels=labels,
            adj_orig=adj_orig,
            cand_src=cand_src,
            cand_dst=cand_dst,
            clean_preds=clean_preds,
            clean_correct=clean_correct,
            clean_accuracy=clean_accuracy,
            verbose=verbose,
        )

    if mode == "two_hop_correct_to_incorrect":
        # Unlike endpoint mode, the two-hop miner uses the supplied
        # candidate edges themselves. It does not construct new pairs
        # from the candidate-node pool.
        return _mine_two_hop_correct_to_incorrect_flips(
            model=model,
            attr=attr,
            labels=labels,
            adj_orig=adj_orig,
            cand_src=cand_src,
            cand_dst=cand_dst,
            clean_preds=clean_preds,
            clean_correct=clean_correct,
            clean_accuracy=clean_accuracy,
            k_samples=two_hop_k_samples,
            rng_seed=seed,
            verbose=verbose,
        )

    raise ValueError(
        f"Unknown mode {mode!r}. Expected one of: "
        "'subset_accuracy_drop', "
        "'endpoint', or "
        "'two_hop_correct_to_incorrect'."
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
    verbose: bool,
) -> dict[str, Any]:
    if not 0.0 < subset_fraction <= 1.0:
        raise ValueError(
            f"subset_fraction must be in (0, 1], got {subset_fraction}."
        )

    if n_subsets <= 0:
        raise ValueError(f"n_subsets must be positive, got {n_subsets}.")

    device = adj_orig.device
    n_cands = int(cand_src.numel())

    subset_size = max(
        1,
        min(n_cands, round(subset_fraction * n_cands)),
    )

    rng = np.random.default_rng(seed)

    drop_sum = np.zeros(n_cands, dtype=np.float64)
    inclusion_count = np.zeros(n_cands, dtype=np.int64)
    drop_per_subset = np.zeros(n_subsets, dtype=np.float64)

    adj_work = adj_orig.clone()
    y_eval = labels[eval_idx]

    for subset_idx in range(n_subsets):
        chosen = rng.choice(
            n_cands,
            size=subset_size,
            replace=False,
        )

        chosen_t = torch.as_tensor(
            chosen,
            device=device,
            dtype=torch.long,
        )

        src = cand_src[chosen_t]
        dst = cand_dst[chosen_t]
        original_values = exists[chosen_t]
        flipped_values = 1.0 - original_values

        adj_work[src, dst] = flipped_values
        adj_work[dst, src] = flipped_values

        pert_preds = model(attr, adj_work).argmax(dim=-1)
        pert_accuracy = float(
            (pert_preds[eval_idx] == y_eval)
            .float()
            .mean()
            .item()
        )

        drop = max(0.0, clean_accuracy - pert_accuracy)

        drop_per_subset[subset_idx] = drop
        drop_sum[chosen] += drop
        inclusion_count[chosen] += 1

        adj_work[src, dst] = original_values
        adj_work[dst, src] = original_values

    labels_raw = drop_sum.copy()
    max_raw = float(labels_raw.max())

    labels_norm = (
        (labels_raw / max_raw).astype(np.float32)
        if max_raw > 1e-8
        else labels_raw.astype(np.float32)
    )

    mean_drop_when_selected = np.divide(
        drop_sum,
        inclusion_count,
        out=np.zeros_like(drop_sum),
        where=inclusion_count > 0,
    )

    if verbose:
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
    model: torch.nn.Module,
    attr: torch.Tensor,
    labels: torch.Tensor,
    adj_orig: torch.Tensor,
    cand_src: torch.Tensor,
    cand_dst: torch.Tensor,
    clean_preds: torch.Tensor,
    clean_correct: torch.Tensor,
    clean_accuracy: float,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate every original candidate edge exactly once.

    Label 1:
        At least one endpoint was correctly classified before the flip
        and incorrectly classified after flipping that exact candidate edge.

    Label 0:
        Otherwise.
    """
    device = adj_orig.device

    attr = attr.to(device)
    labels = labels.to(device=device, dtype=torch.long)

    cand_src = cand_src.to(
        device=device,
        dtype=torch.long,
    ).view(-1)

    cand_dst = cand_dst.to(
        device=device,
        dtype=torch.long,
    ).view(-1)

    clean_preds = clean_preds.to(
        device=device,
        dtype=torch.long,
    )

    clean_correct = clean_correct.to(
        device=device,
        dtype=torch.bool,
    )

    if cand_src.numel() != cand_dst.numel():
        raise ValueError(
            "cand_src and cand_dst must have equal length."
        )

    n_samples = cand_src.numel()

    if n_samples == 0:
        raise ValueError(
            "The candidate list is empty."
        )

    model.eval()
    adj_work = adj_orig.clone()

    endpoint_labels = torch.zeros(
        n_samples,
        device=device,
        dtype=torch.float32,
    )

    src_hit_labels = torch.zeros_like(
        endpoint_labels
    )

    dst_hit_labels = torch.zeros_like(
        endpoint_labels
    )

    both_hit_labels = torch.zeros_like(
        endpoint_labels
    )

    exists_clean = torch.zeros(
        n_samples,
        device=device,
        dtype=torch.bool,
    )

    src_pert_predictions = torch.full(
        (n_samples,),
        -1,
        device=device,
        dtype=torch.long,
    )

    dst_pert_predictions = torch.full_like(
        src_pert_predictions,
        -1,
    )

    rows: list[dict[str, Any]] = []

    for i in range(n_samples):
        u = int(cand_src[i].item())
        v = int(cand_dst[i].item())

        original_uv = float(
            adj_orig[u, v].item()
        )

        original_vu = float(
            adj_orig[v, u].item()
        )

        edge_exists = (
            original_uv > 0.5
            or original_vu > 0.5
        )

        exists_clean[i] = edge_exists

        flipped_value = (
            0.0 if edge_exists else 1.0
        )

        # Flip exactly this original candidate edge.
        adj_work[u, v] = flipped_value
        adj_work[v, u] = flipped_value

        pert_preds = model(
            attr,
            adj_work,
        ).argmax(dim=-1)

        u_clean_pred = int(
            clean_preds[u].item()
        )

        v_clean_pred = int(
            clean_preds[v].item()
        )

        u_pert_pred = int(
            pert_preds[u].item()
        )

        v_pert_pred = int(
            pert_preds[v].item()
        )

        u_true = int(labels[u].item())
        v_true = int(labels[v].item())

        # Correct before, incorrect after.
        u_hit = (
            bool(clean_correct[u].item())
            and u_pert_pred != u_true
        )

        v_hit = (
            bool(clean_correct[v].item())
            and v_pert_pred != v_true
        )

        endpoint_hit = u_hit or v_hit
        both_hit = u_hit and v_hit

        endpoint_labels[i] = float(
            endpoint_hit
        )

        src_hit_labels[i] = float(
            u_hit
        )

        dst_hit_labels[i] = float(
            v_hit
        )

        both_hit_labels[i] = float(
            both_hit
        )

        src_pert_predictions[i] = (
            u_pert_pred
        )

        dst_pert_predictions[i] = (
            v_pert_pred
        )

        rows.append({
            "sample_index": i,
            "u": u,
            "v": v,
            "exists_clean": edge_exists,
            "action": (
                "del"
                if edge_exists
                else "add"
            ),
            "u_label": u_true,
            "v_label": v_true,
            "u_clean_pred": u_clean_pred,
            "v_clean_pred": v_clean_pred,
            "u_pert_pred": u_pert_pred,
            "v_pert_pred": v_pert_pred,
            "u_was_correct": bool(
                clean_correct[u].item()
            ),
            "v_was_correct": bool(
                clean_correct[v].item()
            ),
            "u_hit": u_hit,
            "v_hit": v_hit,
            "both_hit": both_hit,
            "endpoint_hit": endpoint_hit,
        })

        # Restore the graph exactly.
        adj_work[u, v] = original_uv
        adj_work[v, u] = original_vu

    endpoint_hits = int(
        endpoint_labels.sum().item()
    )

    if verbose:
        print(
            f"Clean evaluation accuracy: "
            f"{clean_accuracy:.4f}"
        )

        print(
            f"Original candidate edges "
            f"evaluated: {n_samples}"
        )

        print(
            f"Endpoint hits: "
            f"{endpoint_hits}/{n_samples} "
            f"({endpoint_hits / n_samples:.2%})"
        )

    edge_index_lab = torch.stack(
        [
            cand_src,
            cand_dst,
        ],
        dim=0,
    )

    return {
        "mode": "endpoint",
        "labels_raw": (
            endpoint_labels
            .detach()
            .cpu()
            .numpy()
        ),
        "labels_norm": (
            endpoint_labels
            .detach()
            .cpu()
            .numpy()
        ),
        "endpoint_labels": (
            endpoint_labels
            .detach()
            .cpu()
            .numpy()
        ),
        "sampled_edge_index": (
            edge_index_lab
            .detach()
            .cpu()
        ),
        "sampled_u": (
            cand_src
            .detach()
            .cpu()
            .numpy()
        ),
        "sampled_v": (
            cand_dst
            .detach()
            .cpu()
            .numpy()
        ),
        "u_hit_labels": (
            src_hit_labels
            .detach()
            .cpu()
            .numpy()
        ),
        "v_hit_labels": (
            dst_hit_labels
            .detach()
            .cpu()
            .numpy()
        ),
        "both_hit_labels": (
            both_hit_labels
            .detach()
            .cpu()
            .numpy()
        ),
        "u_pert_predictions": (
            src_pert_predictions
            .detach()
            .cpu()
            .numpy()
        ),
        "v_pert_predictions": (
            dst_pert_predictions
            .detach()
            .cpu()
            .numpy()
        ),
        "exists": (
            exists_clean
            .detach()
            .cpu()
            .numpy()
        ),
        "clean_accuracy": clean_accuracy,
        "endpoint_hits": endpoint_hits,
        "n_samples": n_samples,
        "rows": rows,
    }


@torch.no_grad()
def _mine_two_hop_correct_to_incorrect_flips(
    *,
    model: torch.nn.Module,
    attr: Tensor,
    labels: Tensor,
    adj_orig: Tensor,
    cand_src: Tensor,
    cand_dst: Tensor,
    clean_preds: Tensor,
    clean_correct: Tensor,
    clean_accuracy: float,
    k_samples: int | None = None,
    rng_seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate candidate edge flips by counting correct-to-incorrect prediction
    changes in the union of the endpoints' clean 2-hop neighborhoods.

    For every evaluated candidate edge (u, v):

        1. Compute N_2(u) union N_2(v) on the clean graph, including u and v.
        2. Flip only edge (u, v).
        3. Run the model on the perturbed graph.
        4. Count nodes w in the neighborhood for which:

               clean_preds[w] == labels[w]
               pert_preds[w]  != labels[w]

    This count becomes the edge's raw label.

    Normalization
    -------------
    Raw scores are min-max normalized across the evaluated candidate edges:

        labels_norm = (labels_raw - min) / (max - min)

    If all raw scores are identical, labels_norm is set to zero.

    Candidate sampling
    ------------------
    Candidate pairs are taken directly from cand_src/cand_dst. Undirected
    duplicates such as (u, v) and (v, u) are merged.

    If k_samples is None, all unique candidate edges are evaluated.
    Otherwise, at most k_samples unique candidates are sampled without
    replacement.
    """
    device = adj_orig.device

    attr = attr.to(device)
    labels = labels.to(device=device, dtype=torch.long).view(-1)
    cand_src = cand_src.to(device=device, dtype=torch.long).view(-1)
    cand_dst = cand_dst.to(device=device, dtype=torch.long).view(-1)
    clean_preds = clean_preds.to(device=device, dtype=torch.long).view(-1)
    clean_correct = clean_correct.to(
        device=device,
        dtype=torch.bool,
    ).view(-1)

    if cand_src.numel() != cand_dst.numel():
        raise ValueError(
            "cand_src and cand_dst must have equal length, got "
            f"{cand_src.numel()} and {cand_dst.numel()}."
        )

    if cand_src.numel() == 0:
        raise ValueError("The candidate set is empty.")

    if adj_orig.ndim != 2 or adj_orig.size(0) != adj_orig.size(1):
        raise ValueError(
            "adj_orig must be a square dense adjacency matrix."
        )

    num_nodes = int(adj_orig.size(0))

    if attr.size(0) != num_nodes:
        raise ValueError(
            f"attr contains {attr.size(0)} nodes, but adj_orig contains "
            f"{num_nodes} nodes."
        )

    if labels.numel() != num_nodes:
        raise ValueError(
            f"labels contains {labels.numel()} entries, but the graph has "
            f"{num_nodes} nodes."
        )

    if clean_preds.numel() != num_nodes:
        raise ValueError(
            "clean_preds must contain one prediction per graph node."
        )

    if clean_correct.numel() != num_nodes:
        raise ValueError(
            "clean_correct must contain one Boolean value per graph node."
        )

    if k_samples is not None and k_samples <= 0:
        raise ValueError(
            f"k_samples must be positive or None, got {k_samples}."
        )

    if torch.any(cand_src < 0) or torch.any(cand_src >= num_nodes):
        raise ValueError("cand_src contains an invalid node index.")

    if torch.any(cand_dst < 0) or torch.any(cand_dst >= num_nodes):
        raise ValueError("cand_dst contains an invalid node index.")

    # ---------------------------------------------------------------
    # Canonicalize candidates as undirected pairs: u < v.
    # ---------------------------------------------------------------
    canonical_u = torch.minimum(cand_src, cand_dst)
    canonical_v = torch.maximum(cand_src, cand_dst)

    non_self_loop = canonical_u != canonical_v
    canonical_u = canonical_u[non_self_loop]
    canonical_v = canonical_v[non_self_loop]

    if canonical_u.numel() == 0:
        raise ValueError(
            "No non-self-loop candidate edges remain after filtering."
        )

    # Linearization is only used to remove duplicate undirected pairs.
    candidate_keys = canonical_u * num_nodes + canonical_v
    unique_keys = torch.unique(candidate_keys, sorted=True)

    unique_u = torch.div(
        unique_keys,
        num_nodes,
        rounding_mode="floor",
    )
    unique_v = unique_keys.remainder(num_nodes)

    n_unique_candidates = int(unique_u.numel())

    # ---------------------------------------------------------------
    # Select all candidates or a reproducible random subset.
    # ---------------------------------------------------------------
    if k_samples is None:
        selected_indices = torch.arange(
            n_unique_candidates,
            device=device,
            dtype=torch.long,
        )
    else:
        n_selected = min(int(k_samples), n_unique_candidates)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(rng_seed))

        selected_indices = torch.randperm(
            n_unique_candidates,
            generator=generator,
        )[:n_selected].to(device)

    sampled_u = unique_u[selected_indices]
    sampled_v = unique_v[selected_indices]
    n_samples = int(sampled_u.numel())

    # ---------------------------------------------------------------
    # Build Boolean clean adjacency for neighborhood extraction.
    #
    # We treat the graph as undirected, even if only one direction is
    # present in adj_orig.
    # ---------------------------------------------------------------
    adjacency_bool = (adj_orig > 0.5)
    adjacency_bool = adjacency_bool | adjacency_bool.T

    # Self-connectivity makes distance <= 2 extraction convenient and
    # ensures that each endpoint belongs to its own neighborhood.
    adjacency_with_self = adjacency_bool.clone()
    adjacency_with_self.fill_diagonal_(True)

    # Integer matrix multiplication:
    # two_hop_reachability[i, j] > 0 means that j can be reached from i
    # using at most two steps in adjacency_with_self.
    #
    # Because self-connections are included, this covers distances 0, 1,
    # and 2.
    adjacency_numeric = adjacency_with_self.to(dtype=torch.float32)
    two_hop_reachability = (
        adjacency_numeric @ adjacency_numeric
    ) > 0

    model.eval()

    adj_work = adj_orig.clone()

    labels_raw_tensor = torch.zeros(
        n_samples,
        device=device,
        dtype=torch.float32,
    )

    neighborhood_sizes = torch.zeros(
        n_samples,
        device=device,
        dtype=torch.long,
    )

    clean_correct_counts = torch.zeros_like(neighborhood_sizes)

    exists_clean = torch.zeros(
        n_samples,
        device=device,
        dtype=torch.bool,
    )

    rows: list[dict[str, Any]] = []

    for sample_idx in range(n_samples):
        u = int(sampled_u[sample_idx].item())
        v = int(sampled_v[sample_idx].item())

        # The neighborhood is fixed using the clean graph. The flipped
        # graph is not used to redefine which nodes are evaluated.
        neighborhood_mask = (
            two_hop_reachability[u]
            | two_hop_reachability[v]
        )

        neighborhood_nodes = torch.nonzero(
            neighborhood_mask,
            as_tuple=False,
        ).flatten()

        neighborhood_size = int(neighborhood_nodes.numel())
        neighborhood_sizes[sample_idx] = neighborhood_size

        clean_correct_in_neighborhood = clean_correct[
            neighborhood_nodes
        ]

        n_clean_correct = int(
            clean_correct_in_neighborhood.sum().item()
        )
        clean_correct_counts[sample_idx] = n_clean_correct

        original_uv = float(adj_orig[u, v].item())
        original_vu = float(adj_orig[v, u].item())

        edge_exists = original_uv > 0.5 or original_vu > 0.5
        exists_clean[sample_idx] = edge_exists

        flipped_value = 0.0 if edge_exists else 1.0

        # Flip only the current undirected candidate edge.
        adj_work[u, v] = flipped_value
        adj_work[v, u] = flipped_value

        pert_logits = model(attr, adj_work)
        pert_preds = pert_logits.argmax(dim=-1)

        pert_incorrect = pert_preds != labels

        # A node is counted only when it was correct before the flip and
        # incorrect after the flip.
        correct_to_incorrect_mask = (
            clean_correct
            & pert_incorrect
            & neighborhood_mask
        )

        flipped_nodes = torch.nonzero(
            correct_to_incorrect_mask,
            as_tuple=False,
        ).flatten()

        raw_score = int(flipped_nodes.numel())
        labels_raw_tensor[sample_idx] = float(raw_score)

        rows.append({
            "sample_index": sample_idx,
            "u": u,
            "v": v,
            "exists_clean": edge_exists,
            "action": "del" if edge_exists else "add",
            "neighborhood_size": neighborhood_size,
            "clean_correct_in_neighborhood": n_clean_correct,
            "correct_to_incorrect_count": raw_score,
            "correct_to_incorrect_nodes": (
                flipped_nodes.detach().cpu().tolist()
            ),
        })

        # Restore the exact clean values before evaluating the next edge.
        adj_work[u, v] = original_uv
        adj_work[v, u] = original_vu

    # ---------------------------------------------------------------
    # Min-max normalization across the evaluated candidate edges.
    # ---------------------------------------------------------------
    raw_min = float(labels_raw_tensor.min().item())
    raw_max = float(labels_raw_tensor.max().item())

    if raw_max > raw_min:
        labels_norm_tensor = (
            labels_raw_tensor - raw_min
        ) / (raw_max - raw_min)
    else:
        labels_norm_tensor = torch.zeros_like(labels_raw_tensor)

    # Add normalized scores to the row representation.
    for sample_idx, row in enumerate(rows):
        row["label_raw"] = float(
            labels_raw_tensor[sample_idx].item()
        )
        row["label_norm"] = float(
            labels_norm_tensor[sample_idx].item()
        )

    total_correct_to_incorrect = int(
        labels_raw_tensor.sum().item()
    )
    positive_flips = int(
        (labels_raw_tensor > 0).sum().item()
    )

    mean_raw_score = float(
        labels_raw_tensor.mean().item()
    )

    mean_neighborhood_size = float(
        neighborhood_sizes.float().mean().item()
    )

    if verbose:
        print(f"Clean evaluation accuracy: {clean_accuracy:.4f}")
        print(
            f"Unique candidate edges available: "
            f"{n_unique_candidates}"
        )
        print(
            f"Candidate flips evaluated: {n_samples}"
        )
        print(
            "Score definition: number of clean-correct nodes becoming "
            "incorrect within the union of both endpoints' clean "
            "2-hop neighborhoods"
        )
        print(
            f"Flips with raw score > 0: "
            f"{positive_flips}/{n_samples} "
            f"({positive_flips / n_samples:.2%})"
        )
        print(
            f"Total correct-to-incorrect transitions: "
            f"{total_correct_to_incorrect}"
        )
        print(f"Mean raw score: {mean_raw_score:.4f}")
        print(
            f"Raw score range: [{raw_min:.0f}, {raw_max:.0f}]"
        )
        print(
            f"Mean union-neighborhood size: "
            f"{mean_neighborhood_size:.2f}"
        )

    sampled_edge_index = torch.stack(
        [sampled_u, sampled_v],
        dim=0,
    )

    return {
        "mode": "two_hop_correct_to_incorrect",

        "labels_raw": (
            labels_raw_tensor.detach().cpu().numpy()
        ),
        "labels_norm": (
            labels_norm_tensor.detach().cpu().numpy()
        ),

        "sampled_edge_index": (
            sampled_edge_index.detach().cpu()
        ),
        "sampled_u": sampled_u.detach().cpu().numpy(),
        "sampled_v": sampled_v.detach().cpu().numpy(),

        "exists": exists_clean.detach().cpu().numpy(),

        "neighborhood_sizes": (
            neighborhood_sizes.detach().cpu().numpy()
        ),
        "clean_correct_counts": (
            clean_correct_counts.detach().cpu().numpy()
        ),

        "clean_accuracy": clean_accuracy,

        "n_unique_candidates": n_unique_candidates,
        "n_samples": n_samples,

        "positive_flips": positive_flips,
        "total_correct_to_incorrect": (
            total_correct_to_incorrect
        ),
        "mean_raw_score": mean_raw_score,
        "raw_score_min": raw_min,
        "raw_score_max": raw_max,
        "mean_neighborhood_size": mean_neighborhood_size,

        "rows": rows,
    }

from pathlib import Path
from typing import Any, Optional

import torch


def _move_nested_tensors_to_cpu(value: Any) -> Any:
    """Recursively move tensors in nested structures to CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu()

    if isinstance(value, dict):
        return {
            key: _move_nested_tensors_to_cpu(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _move_nested_tensors_to_cpu(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _move_nested_tensors_to_cpu(item)
            for item in value
        )

    if isinstance(value, set):
        return {
            _move_nested_tensors_to_cpu(item)
            for item in value
        }

    return value


def _move_nested_tensors_to_device(
    value: Any,
    device: torch.device | str,
) -> Any:
    """Recursively move tensors in nested structures to a device."""
    if torch.is_tensor(value):
        return value.to(device)

    if isinstance(value, dict):
        return {
            key: _move_nested_tensors_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _move_nested_tensors_to_device(item, device)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _move_nested_tensors_to_device(item, device)
            for item in value
        )

    if isinstance(value, set):
        return {
            _move_nested_tensors_to_device(item, device)
            for item in value
        }

    return value


def save_mining_output(
    path: str | Path,
    mining_result: dict[str, Any],
    *,
    candidates: Optional[list[tuple[int, int]]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Save a candidate-mining result.

    Tensors are stored on CPU so that the file remains portable between
    CPU and CUDA environments.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": 1,
        "mining_result": _move_nested_tensors_to_cpu(mining_result),
        "candidates": candidates,
        "metadata": metadata or {},
    }

    torch.save(payload, path)
    print(f"[MINING CACHE] Saved mining output to: {path}")


def load_mining_output(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_metadata: Optional[dict[str, Any]] = None,
) -> tuple[
    dict[str, Any],
    Optional[list[tuple[int, int]]],
    dict[str, Any],
]:
    """
    Load a saved candidate-mining result.

    Returns:
        mining_result, candidates, metadata
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Mining cache does not exist: {path}"
        )

    # weights_only=False is needed because the payload can contain
    # ordinary Python structures such as candidate tuples and metadata.
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid mining cache format in {path}"
        )

    if payload.get("format_version") != 1:
        raise ValueError(
            "Unsupported mining-cache version: "
            f"{payload.get('format_version')}"
        )

    metadata = payload.get("metadata", {})

    if expected_metadata is not None:
        mismatches = {
            key: {
                "expected": expected_value,
                "stored": metadata.get(key),
            }
            for key, expected_value in expected_metadata.items()
            if metadata.get(key) != expected_value
        }

        if mismatches:
            raise ValueError(
                "Mining cache metadata does not match the current run: "
                f"{mismatches}"
            )

    mining_result = _move_nested_tensors_to_device(
        payload["mining_result"],
        device,
    )

    candidates = payload.get("candidates")

    print(f"[MINING CACHE] Loaded mining output from: {path}")

    return mining_result, candidates, metadata