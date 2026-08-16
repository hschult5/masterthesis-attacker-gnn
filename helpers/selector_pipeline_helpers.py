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
from sklearn.model_selection import train_test_split

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

# Computes Accuracy on model
def accuracy(model, x, labels, idx, edge_index=None):
    model.eval()
    with torch.no_grad():
        logits = model(data=x, adj=edge_index)
        predictions = logits[idx].argmax(dim=1)
        return (predictions == labels[idx]).float().mean().item()

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
    import copy

    import torch
    import torch.nn as nn
    from tqdm import tqdm

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
    train_losses: list[float] = []
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

        train_loss = loss_fn(logits_train, train_y_labels_smooth)
        train_loss.backward()

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

        # Compute validation loss
        with torch.no_grad():
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