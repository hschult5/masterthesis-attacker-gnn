import copy
import logging

from collections import defaultdict
import math
from typing import List, Tuple, Optional, Set, Dict

from torch import Tensor
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import torch_sparse
from torch_sparse import SparseTensor
from AttackerGNN.PriorSelector import PriorSelector
from AttackerGNN.NodeBlockScorer import NodeBlockScorer
import AttackerGNN.gnn_least_likely_edge as lle
import AttackerGNN.gnn_score_all as sall
from AttackerGNN.GCNLinkPredictor import GCNLinkPredictor
from AttackerGNN.GCNMarginGradientPredictor import TinyGCN, tanh_margin_loss_label_free
from AttackerGNN.ShadowModelLinkPredictor import LinkPredictionGNN
from rgnn_at_scale.helper import utils
from rgnn_at_scale.attacks.base_attack import Attack, SparseAttack
import os
import csv
from datetime import datetime

from rgnn_at_scale.helper.csvLogger import CSVMetricLogger


class PRBCD(SparseAttack):
    """Sampled and hence scalable PGD attack for graph data.
    """

    def __init__(self,
                 keep_heuristic: str = 'WeightOnly',
                 lr_factor: float = 100,
                 display_step: int = 20,
                 epochs: int = 400,
                 fine_tune_epochs: int = 100,
                 block_size: int = 1_000_000,
                 with_early_stopping: bool = True,
                 do_synchronize: bool = False,
                 eps: float = 1e-7,
                 max_final_samples: int = 20,
                 pre_hidden: int = 64, # New
                 **kwargs):
        super().__init__(**kwargs)

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
        self.current_node_search_space: torch.Tensor = None
        self.sample_space: torch.Tensor = None
        self.edges_to_attack_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        self.semi = None

        self.make_pgd_forward_from_victim()

        # --- selector optimizer for online updates at resampling ---
        if getattr(self, "selector", None) is not None:
            self.selector_opt = torch.optim.Adam(
                self.selector.parameters(), lr=1e-3, weight_decay=5e-4
            )
        else:
            self.selector_opt = None

        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.block_size), 1.)

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
        selector_params = kwargs.get("selector_params", {}) or {}

        assert self.block_size > n_perturbations, \
            f'The search space size ({self.block_size}) must be ' \
            + f'greater than the number of permutations ({n_perturbations})'

        # For early stopping (not explicitly covered by pesudo code)
        best_accuracy = float('Inf')
        best_epoch = float('-Inf')

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        #tried_mask for selector exclusion
        self.tried_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)

        # Sample initial search space (Algorithm 1, line 3-4)
        if use_cert in ("accuracy_drop_selector", "accuracy_drop_selector_with_resampling"):
            print(use_cert, "-> sampling with accuracy drop selector")

            self._load_selector_params(selector_params, ads_mode=ads_mode)

            if self.drop_mode == "acc": #TODO: logging für alle ads_modes+drop_modes
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.acc_drop_threshold_k_samples}.pt"
            elif self.drop_mode == "loss":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.loss_drop_threshold_k_samples}.pt"
            elif self.drop_mode == "endpoint":
                if self.training_data_node_cap > 0:
                    cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_one_sample}_drpmd{self.drop_mode}_trnodecap{self.training_data_node_cap}"
                else:
                    cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_one_sample}_drpmd{self.drop_mode}.pt"
            elif self.drop_mode == "endpointPRBCD":
                if self.training_data_node_cap > 0:
                    cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_one_sample}_drpmd{self.drop_mode}_trnodecap{self.training_data_node_cap}"
                else:
                    cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_one_sample}_drpmd{self.drop_mode}.pt"

            if os.path.exists(cache_path):
                print("[CACHE] loading selection:", cache_path)
                (
                    y_out,
                    edge_index_lab,
                    y_label,
                    tried_set,
                    harmful_set,
                    sub_nodes,
                    edge_index_sub,
                    edge_weight_sub,
                    edge_index_lab_local,
                    X_sub,
                    edge_index_struct_local,
                    meta,
                ) = PRBCD.load_selection(
                    cache_path,
                    device=self.device,
                )
            else:
                print("[CACHE] computing selection and saving:", cache_path)

                y_out, edge_index_lab, y_label, tried_set, harmful_set = self.label_edge_flips_prbcd_selfsample_fast(
                    n_perturbations=n_perturbations,
                    mode=ads_mode,
                    drop_mode=self.drop_mode,
                    n_candidates_k_sample=self.n_candidates_k_sample,
                    n_candidates_one_sample=self.n_candidates_one_sample,
                    acc_drop_threshold_k_samples=self.acc_drop_threshold_k_samples,
                    loss_drop_threshold_k_samples=self.loss_drop_threshold_k_samples,
                    k_samples_batch=self.k_samples_batch,
                    training_data_node_cap=self.training_data_node_cap,
                )
                meta = {
                    "ads_mode": self.ads_mode,
                    "n_candidates_k_sample": self.n_candidates_k_sample,
                    "acc_drop_threshold_k_samples": self.acc_drop_threshold_k_samples,
                    "loss_drop_threshold_k_samples": self.loss_drop_threshold_k_samples,
                }
                PRBCD.save_selection(cache_path, y_out, edge_index_lab, y_label, tried_set, harmful_set, meta=meta)

            # ---- selection statistics: tried vs harmful ----
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            PRBCD.record_selection_statistics(
                tried_set=tried_set,
                harmful_set=harmful_set,
                n_nodes=int(self.n),
                device=self.device,
                stats_dir="Plotting_Data/TrainingDataStats",
                csv_prefix=(
                    f"{timestamp}_"
                    f"selection_dataset{self.dataset}_seed{self.seed}_"
                    f"ads_{ads_mode}_drpmd{self.drop_mode}_"
                    f"trnodecap{getattr(self, 'training_data_node_cap', 0)}"
                ),
                top_k=10,
                extra={
                    "timestamp": timestamp,
                    "dataset": self.dataset,
                    "seed": self.seed,
                    "ads_mode": ads_mode,
                    "drop_mode": self.drop_mode,
                    "training_data_node_cap": int(getattr(self, "training_data_node_cap", 0)),
                    "n_candidates_one_sample": int(getattr(self, "n_candidates_one_sample", 0)),
                    "n_candidates_k_sample": int(getattr(self, "n_candidates_k_sample", 0)),
                    "k_samples_batch": int(getattr(self, "k_samples_batch", 0)),
                    "tau": float(getattr(self, "tau", 0.0)),
                },
            )

            X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)
            stats = PRBCD.tried_add_del_proportion(tried_set, edge_index_struct, n=int(self.n))
            print(stats)

            self.lp_model = self.train_link_prediction_gnn(
                x=X,
                edge_index_struct=edge_index_struct,
                edge_index_lab=edge_index_lab,
                y_label=y_label,
                device=self.device,
                num_epochs=200,
                use_tqdm=True,
                verbose=True,
            )
            self.sample_block_from_linkpred_threshold(graph=graph, n_perturbations=n_perturbations, tau=self.tau)
            self.tried_set = tried_set
        elif use_cert in ("accuracy_drop_selector_subgraph_random", "accuracy_drop_selector_subgraph_khop", "accuracy_drop_selector_subgraph_growhop"):
            print(use_cert, "-> sampling with accuracy drop selector subgraph")

            self._load_selector_params(selector_params, ads_mode=ads_mode)

            if use_cert in ("accuracy_drop_selector_subgraph_random",):
                self.ads_mode = "random_subgraph"
            elif use_cert in ("accuracy_drop_selector_subgraph_khop",):
                self.ads_mode = "khop_subgraph"
            else:
                self.ads_mode = "growhop_subgraph"

            if self.ads_mode == "random_subgraph":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_{self.ads_mode}_n{self.n_candidates_one_sample}_m{self.subgraph_size}.pt"
            elif self.ads_mode == "khop_subgraph":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_{self.ads_mode}_n{self.n_candidates_one_sample}_k{self.k_subgraph}.pt"
            elif self.ads_mode == "growhop_subgraph":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_{self.ads_mode}_n{self.n_candidates_one_sample}_m{self.subgraph_size}.pt"
            else:
                raise ValueError("ads_mode Unknown")

            X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)

            if os.path.exists(cache_path):
                print("[CACHE] loading selection:", cache_path)
                (
                    y_out,
                    edge_index_lab,
                    y_label,
                    tried_set,
                    harmful_set,
                    sub_nodes,
                    edge_index_sub,
                    edge_weight_sub,
                    edge_index_lab_local,
                    X_sub,
                    edge_index_struct_local,
                    meta,
                ) = PRBCD.load_selection(
                    cache_path,
                    device=self.device,
                )
            else:
                print("[CACHE] computing selection and saving:", cache_path)
                if self.ads_mode == "random_subgraph":
                    sub_nodes, edge_index_sub, edge_weight_sub, g2l, l2g = PRBCD.extract_random_induced_subgraph(
                        edge_index_struct,
                        self.n,
                        subgraph_size=self.subgraph_size,
                    )
                elif self.ads_mode == "khop_subgraph": #no local values
                    sub_nodes, edge_index_sub, edge_weight_sub, g2l, l2g = PRBCD.extract_k_hop_induced_subgraph_of_random_node(
                        edge_index_struct,
                        self.n,
                        k=self.k_subgraph,
                    )
                elif self.ads_mode == "growhop_subgraph":
                    sub_nodes, edge_index_sub, edge_weight_sub, g2l, l2g, hops_used = PRBCD.extract_growing_hop_subgraph_of_random_node(
                        edge_index_struct,
                        self.n,
                        subgraph_size=self.subgraph_size
                    )
                else:
                    raise ValueError("ads_mode Unknown")

                y_out, edge_index_lab, y_label, edge_index_lab_local, tried_set, harmful_set = self.label_edge_flips_prbcd_subgraph_endpoint_one_sample(
                    n_candidates_one_sample=self.n_candidates_one_sample,
                    sub_nodes=sub_nodes,
                )

                X_sub, edge_index_struct_local = PRBCD.build_lp_struct_for_subgraph(
                    X=X,
                    edge_index_struct=edge_index_struct,
                    sub_nodes=sub_nodes,
                    device=self.device,
                )

                meta = {
                    "n_candidates_one_sample": self.n_candidates_one_sample,
                }
                PRBCD.save_selection(cache_path, y_out, edge_index_lab, y_label, tried_set, harmful_set, sub_nodes, edge_index_sub, edge_weight_sub, edge_index_lab_local, X_sub, edge_index_struct_local, meta=meta)

            stats = PRBCD.tried_add_del_proportion(tried_set, edge_index_struct, n=int(self.n))
            print(stats)

            self.sub_nodes = sub_nodes
            self.X_sub = X_sub
            self.edge_index_struct_local = edge_index_struct_local

            self.lp_model = self.train_link_prediction_gnn(
                x=X_sub,
                edge_index_struct=edge_index_struct_local,
                edge_index_lab=edge_index_lab_local,
                y_label=y_label,
                device=self.device,
                num_epochs=200,
                use_tqdm=True,
                verbose=True,
            )
            #self.sample_block_from_linkpred_threshold_subgraph(n_perturbations=n_perturbations, tau=0.8)
            self.sample_block_from_linkpred_threshold(graph=graph, n_perturbations=n_perturbations)
            # self.init_search_space_from_y_out(y_out=y_out,n_perturbations=n_perturbations)
            self.tried_set = tried_set
        else:
            print(use_cert, "run sampling with no certificate")
            self.sample_random_block(n_perturbations, self.block_size)
        # Accuracy and attack statistics before the attack even started
        with torch.no_grad():

            logits = self._get_logits(self.attr, self.edge_index, self.edge_weight)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

            logging.info(f'\nBefore the attack - Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

            self._append_attack_statistics(loss.item(), accuracy, 0., 0.)

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
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack]) #Todo: Hier wird der loss und gradient für perturbed edge weight erzeugt.
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

                self._append_attack_statistics(loss, accuracy, probability_mass_update, probability_mass_projected)

                # Resampling of search space (Algorithm 1, line 9-14)
                if epoch < self.epochs_resampling - 1:
                    if use_cert in ("accuracy_drop_selector", "accuracy_drop_selector_subgraph", "accuracy_drop_selector_with_resampling"):
                        print(use_cert, "run resampling with no certificate")
                        if use_cert in ("accuracy_drop_selector_with_resampling", ):
                            if epoch % 2 == 0:
                                self.resample_block_from_linkpred_threshold(graph=graph,
                                                                            n_perturbations=n_perturbations, score_batch_size=int(self.block_size/10), tau=0.7-epoch*0.04)
                        else:
                            self.resample_random_block(n_perturbations=n_perturbations, mod_block_size=self.block_size)
                        pass
                    else:
                        print(use_cert, "run resampling with no certificate")
                        self.resample_random_block(n_perturbations, mod_block_size=self.block_size)
                        pass
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

    def sample_block_from_linkpred_model(
            self,
            graph,
            n_perturbations: int = 0,
            pool_size: int = 50_000,  # how many candidates to score each attempt
            pool_multiplier: int = 4,  # pool_size = pool_multiplier * block_size if pool_size is None
            score_batch_size: int = 200_000,  # chunking for scoring pairs (safe default)
    ):
        """
        Like sample_random_block(), but uses self.lp_model to score candidates and
        initializes current_search_space with the top-scoring pairs.

        Requires:
          - self.lp_model already trained and on correct device
          - self.extract_X_and_edge_index_from_sparsegraph(graph)
          - PRBCD.linear_to_triu_idx / PRBCD.linear_to_full_idx
          - self.n_possible_edges, self.block_size, self.eps, self.make_undirected, self.device
        """

        if not hasattr(self, "lp_model") or self.lp_model is None:
            raise RuntimeError("self.lp_model is not set. Train/load the linkpred model before sampling.")

        # determine pool size
        if pool_size is None:
            pool_size = int(pool_multiplier) * int(self.block_size)
        pool_size = int(pool_size)

        # Get structure and features
        X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)
        X = X.to(self.device)
        edge_index_struct = edge_index_struct.to(self.device)

        # Precompute node embeddings ONCE
        self.lp_model = self.lp_model.to(self.device).eval()
        with torch.no_grad():
            h = self.lp_model.encoder(X, edge_index_struct)  # (N, d)

        # helper to score decoded pairs with the edge head
        def _score_pairs(edge_index_lab: torch.Tensor) -> torch.Tensor:
            # returns sigmoid scores shape (M,)
            if edge_index_lab.numel() == 0:
                return torch.empty((0,), device=self.device)
            with torch.no_grad():
                logits = self.lp_model.edge_head(h, edge_index_lab).view(-1)
                return torch.sigmoid(logits)

        # Repeat like sample_random_block does, but now "sample then rank"
        for _ in range(self.max_final_samples):

            # 1) sample candidate linear indices
            cand_lin = torch.randint(
                self.n_possible_edges, (pool_size,), device=self.device
            )
            cand_lin = torch.unique(cand_lin, sorted=True)

            # 2) decode to pairs exactly like PRBCD does
            if self.make_undirected:
                cand_ei = PRBCD.linear_to_triu_idx(self.n, cand_lin)  # (2, M)
            else:
                cand_ei = PRBCD.linear_to_full_idx(self.n, cand_lin)  # (2, M)
                is_not_self_loop = cand_ei[0] != cand_ei[1]
                cand_lin = cand_lin[is_not_self_loop]
                cand_ei = cand_ei[:, is_not_self_loop]

            # If we lost too many due to uniqueness/self-loop removal, resample
            if cand_lin.numel() == 0:
                continue

            # 3) score candidates (chunked)
            M = cand_lin.numel()
            scores = torch.empty((M,), device=self.device, dtype=torch.float32)

            start = 0
            while start < M:
                end = min(start + int(score_batch_size), M)
                scores[start:end] = _score_pairs(cand_ei[:, start:end])
                start = end

            # 4) pick top block_size
            k = min(int(self.block_size), M)
            top_idx = torch.topk(scores, k=k, largest=True).indices

            self.current_search_space = cand_lin[top_idx]
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)

            # if we ended up with fewer than block_size due to uniqueness, you can top up by taking more
            if self.current_search_space.numel() < int(self.block_size) and M > k:
                # take more in descending order until we fill
                sorted_idx = torch.argsort(scores, descending=True)
                fill = []
                taken = set(self.current_search_space.tolist())
                for j in sorted_idx.tolist():
                    lin_j = int(cand_lin[j].item())
                    if lin_j in taken:
                        continue
                    fill.append(lin_j)
                    taken.add(lin_j)
                    if len(taken) >= int(self.block_size):
                        break
                if fill:
                    fill_t = torch.tensor(fill, device=self.device, dtype=self.current_search_space.dtype)
                    self.current_search_space = torch.unique(
                        torch.cat([self.current_search_space, fill_t], dim=0),
                        sorted=True
                    )

            # 5) initialize modified_edge_index EXACTLY like sample_random_block
            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            # 6) initialize perturbed_edge_weight EXACTLY like sample_random_block
            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )

            self.tried_mask[self.current_search_space] = True

            if self.current_search_space.size(0) >= n_perturbations:
                return

        raise RuntimeError(
            "Sampling model-guided block was not successful. "
            "Try increasing pool_size or decreasing n_perturbations / block_size."
        )

    def sample_block_from_linkpred_threshold(
            self,
            graph,
            n_perturbations: int = 0,
            tau: float = 0.8,
            max_sampling_tries: int = 2_000_000,
            score_batch_size: int = 1000,
            rng_seed: int = 0,
            exclude_tried: bool = False,
    ):
        if not hasattr(self, "lp_model") or self.lp_model is None:
            raise RuntimeError("self.lp_model is not set.")

        # optional global tried mask
        if exclude_tried and (not hasattr(self, "tried_mask") or self.tried_mask is None):
            self.tried_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)

        # prep features/structure + embeddings once
        X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)
        X = X.to(self.device)
        edge_index_struct = edge_index_struct.to(self.device)

        self.lp_model = self.lp_model.to(self.device).eval()
        with torch.no_grad():
            h = self.lp_model.encoder(X, edge_index_struct)

        g = torch.Generator(device=self.device)
        g.manual_seed(int(rng_seed))

        accepted = []
        accepted_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)

        tries = 0
        while len(accepted) < int(self.block_size) and tries < int(max_sampling_tries):
            tries += 1

            # sample a batch of candidate linear indices
            cand_lin = torch.randint(
                self.n_possible_edges,
                (int(score_batch_size),),
                device=self.device,
                generator=g,
            )
            cand_lin = torch.unique(cand_lin, sorted=False)

            # exclude duplicates within this block
            cand_lin = cand_lin[~accepted_mask[cand_lin]]

            # optionally exclude globally tried
            if exclude_tried:
                cand_lin = cand_lin[~self.tried_mask[cand_lin]]

            if cand_lin.numel() == 0:
                continue

            # decode to pairs for scoring
            if self.make_undirected:
                cand_ei = PRBCD.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCD.linear_to_full_idx(self.n, cand_lin)
                is_not_self = cand_ei[0] != cand_ei[1]
                cand_lin = cand_lin[is_not_self]
                cand_ei = cand_ei[:, is_not_self]
                if cand_lin.numel() == 0:
                    continue

            # score candidates
            with torch.no_grad():
                logits = self.lp_model.edge_head(h, cand_ei).view(-1)
                scores = torch.sigmoid(logits)

            # accept those above threshold
            keep = scores >= float(tau)
            cand_keep = cand_lin[keep]

            if cand_keep.numel() == 0:
                continue

            # add until block is full
            need = int(self.block_size) - len(accepted)
            cand_keep = cand_keep[:need]

            accepted.extend(cand_keep.tolist())
            accepted_mask[cand_keep] = True

        if len(accepted) < int(self.block_size):
            raise RuntimeError(
                f"Could not fill block_size={self.block_size} with threshold tau={tau}. "
                f"Got {len(accepted)} after {tries} tries. Lower tau or increase max_sampling_tries."
            )

        # finalize current_search_space exactly like PRBCD
        self.current_search_space = torch.tensor(
            accepted,
            device=self.device,
            dtype=torch.long,
        )
        self.current_search_space = torch.unique(self.current_search_space, sorted=True)

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
            is_not_self = self.modified_edge_index[0] != self.modified_edge_index[1]
            self.current_search_space = self.current_search_space[is_not_self]
            self.modified_edge_index = self.modified_edge_index[:, is_not_self]

        PRBCD.print_linear_tensor_node_dominance(
            name="SAMPLED_PRBCD_BLOCK",
            lin_indices=self.current_search_space,
            n_nodes=int(self.n),
            top_k=10,
            device=self.device,
        )

        # ============================================================
        # Optional LP hit-rate detour
        # Does NOT write into attack_statistics.
        # Does NOT change the chosen block.
        # Does NOT stop the PR-BCD attack.
        # ============================================================
        if getattr(self, "lp_hit_rate_detour", True):
            with torch.no_grad():
                detour_scores = torch.empty(
                    self.current_search_space.numel(),
                    device=self.device,
                    dtype=torch.float32,
                )

                start = 0
                while start < self.current_search_space.numel():
                    end = min(
                        start + int(score_batch_size),
                        self.current_search_space.numel(),
                    )

                    batch_ei = self.modified_edge_index[:, start:end]
                    detour_logits = self.lp_model.edge_head(h, batch_ei).view(-1)
                    detour_scores[start:end] = torch.sigmoid(detour_logits)

                    start = end

            self._lp_endpoint_hit_rate_detour(
                cand_lin=self.current_search_space,
                cand_ei=self.modified_edge_index,
                scores=detour_scores,
                top_k=getattr(self, "lp_hit_rate_top_k", 200),
                out_dir=getattr(
                    self,
                    "lp_hit_rate_out_dir",
                    "extendedPlotting/lpEndpointHitRate",
                ),
                use_cert=getattr(self, "use_cert", ""),
                ads_mode=getattr(self, "ads_mode", ""),
                tau=tau,
            )

        self.perturbed_edge_weight = torch.full_like(
            self.current_search_space,
            self.eps,
            dtype=torch.float32,
            requires_grad=True,
        )

        if exclude_tried:
            self.tried_mask[self.current_search_space] = True

        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError("Block has fewer unique edges than n_perturbations. Lower tau or increase sampling.")

    @torch.no_grad()
    def _lp_endpoint_hit_rate_detour(
            self,
            cand_lin: torch.Tensor,
            cand_ei: torch.Tensor,
            scores: torch.Tensor,
            top_k: int = 200,
            out_dir: str = "extendedPlotting/lpEndpointHitRate",
            use_cert: str = "",
            ads_mode: str = "",
            tau: float = None,
    ):
        import os
        import csv
        import time

        os.makedirs(out_dir, exist_ok=True)

        cand_lin = cand_lin.detach().to(self.device).long()
        cand_ei = cand_ei.detach().to(self.device).long()
        scores = scores.detach().to(self.device).float()

        if cand_lin.numel() == 0 or scores.numel() == 0:
            return

        k = min(int(top_k), int(scores.numel()))
        top_idx = torch.topk(scores, k=k, largest=True).indices

        selected_lin = cand_lin[top_idx]
        selected_ei = cand_ei[:, top_idx]
        selected_scores = scores[top_idx]

        clean_edge_index = self.edge_index.to(self.device)
        clean_edge_weight = (
            self.edge_weight.to(self.device).float()
            if getattr(self, "edge_weight", None) is not None
            else torch.ones(clean_edge_index.size(1), device=self.device)
        )

        labels_dev = self.labels.to(self.device)

        logits_clean = self.attacked_model(
            data=self.attr.to(self.device),
            adj=(clean_edge_index, clean_edge_weight),
        )

        pred_clean = logits_clean.argmax(dim=-1)
        clean_correct = pred_clean == labels_dev

        n = int(self.n)

        present = PRBCD._build_uppertri_bitset(clean_edge_index, n).to(self.device)

        E = clean_edge_index.size(1)
        dir_pos = torch.full(
            (n, n),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        dir_pos[clean_edge_index[0], clean_edge_index[1]] = torch.arange(
            E,
            dtype=torch.long,
            device=self.device,
        )

        rows = []

        endpoint_hits = 0
        u_hits = 0
        v_hits = 0
        both_hits = 0

        for lin_t, edge_t, score_t in zip(
                selected_lin,
                selected_ei.t(),
                selected_scores,
        ):
            lin = int(lin_t.item())
            u = int(edge_t[0].item())
            v = int(edge_t[1].item())
            score = float(score_t.item())

            if self.make_undirected:
                exists_clean = bool(present[lin].item())
            else:
                exists_clean = int(dir_pos[u, v].item()) >= 0

            action = "del" if exists_clean else "add"

            edge_index_use = clean_edge_index
            edge_weight_use = clean_edge_weight.clone()

            if action == "del":
                del_pos = []

                pos_uv = int(dir_pos[u, v].item())
                if pos_uv >= 0:
                    del_pos.append(pos_uv)

                if self.make_undirected:
                    pos_vu = int(dir_pos[v, u].item())
                    if pos_vu >= 0:
                        del_pos.append(pos_vu)

                if del_pos:
                    del_pos = torch.tensor(
                        del_pos,
                        dtype=torch.long,
                        device=self.device,
                    )
                    edge_weight_use[del_pos] = 0.0

            else:
                if self.make_undirected:
                    extra_edges = torch.tensor(
                        [[u, v], [v, u]],
                        dtype=torch.long,
                        device=self.device,
                    ).t()
                else:
                    extra_edges = torch.tensor(
                        [[u], [v]],
                        dtype=torch.long,
                        device=self.device,
                    )

                extra_weights = torch.ones(
                    extra_edges.size(1),
                    dtype=torch.float32,
                    device=self.device,
                )

                edge_index_use = torch.cat([edge_index_use, extra_edges], dim=1)
                edge_weight_use = torch.cat([edge_weight_use, extra_weights], dim=0)

            logits_pert = self.attacked_model(
                data=self.attr.to(self.device),
                adj=(edge_index_use, edge_weight_use),
            )

            pred_pert = logits_pert.argmax(dim=-1)

            u_was_correct = bool(clean_correct[u].item())
            v_was_correct = bool(clean_correct[v].item())

            u_now_wrong = int(pred_pert[u].item()) != int(labels_dev[u].item())
            v_now_wrong = int(pred_pert[v].item()) != int(labels_dev[v].item())

            u_hit = u_was_correct and u_now_wrong
            v_hit = v_was_correct and v_now_wrong

            endpoint_hit = u_hit or v_hit
            both_hit = u_hit and v_hit

            endpoint_hits += int(endpoint_hit)
            u_hits += int(u_hit)
            v_hits += int(v_hit)
            both_hits += int(both_hit)

            rows.append({
                "dataset": getattr(self, "dataset", ""),
                "seed": getattr(self, "seed", ""),
                "use_cert": use_cert,
                "ads_mode": ads_mode,
                "tau": tau,

                "linear_idx": lin,
                "u": u,
                "v": v,
                "lp_score": score,
                "exists_clean": exists_clean,
                "action": action,

                "u_label": int(labels_dev[u].item()),
                "v_label": int(labels_dev[v].item()),
                "u_clean_pred": int(pred_clean[u].item()),
                "v_clean_pred": int(pred_clean[v].item()),
                "u_pert_pred": int(pred_pert[u].item()),
                "v_pert_pred": int(pred_pert[v].item()),

                "u_was_correct": u_was_correct,
                "v_was_correct": v_was_correct,
                "u_hit_correct_to_incorrect": u_hit,
                "v_hit_correct_to_incorrect": v_hit,
                "endpoint_hit": endpoint_hit,
            })

        n_selected = len(rows)

        endpoint_hit_rate = endpoint_hits / n_selected if n_selected else 0.0
        u_hit_rate = u_hits / n_selected if n_selected else 0.0
        v_hit_rate = v_hits / n_selected if n_selected else 0.0
        both_hit_rate = both_hits / n_selected if n_selected else 0.0

        summary_row = {
            "dataset": getattr(self, "dataset", ""),
            "seed": getattr(self, "seed", ""),
            "use_cert": use_cert,
            "ads_mode": ads_mode,
            "tau": tau,
            "top_k": k,
            "n_selected": n_selected,

            "endpoint_hits": endpoint_hits,
            "u_hits": u_hits,
            "v_hits": v_hits,
            "both_hits": both_hits,

            "endpoint_hit_rate": endpoint_hit_rate,
            "u_hit_rate": u_hit_rate,
            "v_hit_rate": v_hit_rate,
            "both_hit_rate": both_hit_rate,

            "mean_lp_score": float(selected_scores.mean().item()) if n_selected else "",
            "max_lp_score": float(selected_scores.max().item()) if n_selected else "",
            "min_lp_score": float(selected_scores.min().item()) if n_selected else "",
        }

        summary_path = os.path.join(
            out_dir,
            f"lp_endpoint_hitrate__dataset{getattr(self, 'dataset', '')}"
            f"__seed{getattr(self, 'seed', '')}"
            f"__usecert{use_cert}"
            f"__mode{ads_mode}"
            f"__tau{tau}"
            f"__{time.strftime('%y%m%d-%H%M%S')}.csv",
        )

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)

        details_path = summary_path.replace(".csv", "__details.csv")

        if rows:
            with open(details_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        print("[LP hit-rate detour]")
        print("summary:", summary_path)
        print("details:", details_path)
        print("endpoint_hit_rate:", endpoint_hit_rate)

    def sample_block_from_linkpred_threshold_subgraph(
            self,
            n_perturbations: int = 0,
            tau: float = 0.8,
            max_sampling_tries: int = 2_000_000,
            score_batch_size: int = 1000,
            rng_seed: int = 0,
            exclude_tried: bool = False,
    ):
        """
        Uses an LP model trained on a SUBGRAPH (local ids 0..m-1) to sample a block,
        but stores the resulting search space as GLOBAL linear edge ids (same semantics as before).

        Required attributes available on self (from cache or computation):
          - self.sub_nodes                (m,) global ids; index = local id
          - self.X_sub                    (m,F) features in local row order (optional; if not, build from global X)
          - self.edge_index_struct_local  (2,E_sub) local ids
          - self.lp_model                 trained on subgraph

        Also uses existing globals:
          - self.n (global N)
          - self.make_undirected
          - self.block_size
          - self.eps
          - self.n_possible_edges (global, for tried_mask size)
        """
        import torch

        if not hasattr(self, "lp_model") or self.lp_model is None:
            raise RuntimeError("self.lp_model is not set.")

        if not hasattr(self, "sub_nodes") or self.sub_nodes is None:
            raise RuntimeError("self.sub_nodes missing. Need sub_nodes to map local->global.")

        if not hasattr(self, "edge_index_struct_local") or self.edge_index_struct_local is None:
            raise RuntimeError("self.edge_index_struct_local missing. Need local structure for encoder.")

        # Optional global tried mask (GLOBAL linear ids)
        if exclude_tried and (not hasattr(self, "tried_mask") or self.tried_mask is None):
            self.tried_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)

        sub_nodes = self.sub_nodes.to(self.device)
        edge_index_struct_local = self.edge_index_struct_local.to(self.device)

        m = int(sub_nodes.numel())
        if m < 2:
            raise RuntimeError("Subgraph must have at least 2 nodes.")

        # Prepare X_sub
        if hasattr(self, "X_sub") and self.X_sub is not None:
            X_sub = self.X_sub.to(self.device)
        else:
            raise RuntimeError("self.X_sub missing. Store it from cache or build it before calling this sampler.")

        # Encode once (LOCAL)
        self.lp_model = self.lp_model.to(self.device).eval()
        with torch.no_grad():
            h = self.lp_model.encoder(X_sub, edge_index_struct_local)

        g = torch.Generator(device=self.device)
        g.manual_seed(int(rng_seed))

        # Local candidate space size
        if self.make_undirected:
            n_possible_local = m * (m - 1) // 2
        else:
            n_possible_local = m * m  # includes self-loops; we'll filter them like before

        accepted_global = []  # GLOBAL linear ids (this is what PRBCD expects downstream)
        accepted_local_mask = torch.zeros(n_possible_local, device=self.device, dtype=torch.bool)

        tries = 0
        while len(accepted_global) < int(self.block_size) and tries < int(max_sampling_tries):
            tries += 1

            # sample LOCAL candidate linear ids
            cand_lin_local = torch.randint(
                n_possible_local, (int(score_batch_size),), device=self.device, generator=g
            )
            cand_lin_local = torch.unique(cand_lin_local, sorted=False)

            # exclude duplicates within this block (LOCAL)
            cand_lin_local = cand_lin_local[~accepted_local_mask[cand_lin_local]]
            if cand_lin_local.numel() == 0:
                continue

            # decode LOCAL lin -> LOCAL pairs for scoring
            if self.make_undirected:
                cand_ei_local = PRBCD.linear_to_triu_idx(m, cand_lin_local)  # (2, B) local ids
                # convert to GLOBAL lin to optionally exclude tried
                u_l = cand_ei_local[0]
                v_l = cand_ei_local[1]
            else:
                cand_ei_local = PRBCD.linear_to_full_idx(m, cand_lin_local)
                is_not_self = cand_ei_local[0] != cand_ei_local[1]
                cand_lin_local = cand_lin_local[is_not_self]
                cand_ei_local = cand_ei_local[:, is_not_self]
                if cand_lin_local.numel() == 0:
                    continue
                u_l = cand_ei_local[0]
                v_l = cand_ei_local[1]

            # map LOCAL endpoints -> GLOBAL endpoints
            u_g = sub_nodes[u_l]
            v_g = sub_nodes[v_l]

            # build GLOBAL linear ids for those candidates (same convention as the rest of PRBCD)
            if self.make_undirected:
                # ensure u < v for upper-tri encoding
                uu = torch.minimum(u_g, v_g)
                vv = torch.maximum(u_g, v_g)
                cand_ei_global_uv = torch.stack([uu, vv], dim=0)  # (2,B)
                cand_lin_global = PRBCD.triu_idx_to_linear_idx(int(self.n), cand_ei_global_uv).long()
            else:
                cand_ei_global_uv = torch.stack([u_g, v_g], dim=0)
                cand_lin_global = PRBCD.full_idx_to_linear_idx(int(self.n), cand_ei_global_uv).long()

            # optionally exclude globally tried (GLOBAL)
            if exclude_tried:
                keep_not_tried = ~self.tried_mask[cand_lin_global]
                cand_lin_local = cand_lin_local[keep_not_tried]
                cand_ei_local = cand_ei_local[:, keep_not_tried]
                cand_lin_global = cand_lin_global[keep_not_tried]
                if cand_lin_local.numel() == 0:
                    continue

            # score using LOCAL embeddings + LOCAL pairs
            with torch.no_grad():
                logits = self.lp_model.edge_head(h, cand_ei_local).view(-1)
                scores = torch.sigmoid(logits)

            keep = scores >= float(tau)
            if keep.sum().item() == 0:
                continue

            cand_keep_local = cand_lin_local[keep]
            cand_keep_global = cand_lin_global[keep]

            # add until block is full
            need = int(self.block_size) - len(accepted_global)
            cand_keep_local = cand_keep_local[:need]
            cand_keep_global = cand_keep_global[:need]

            accepted_global.extend(cand_keep_global.tolist())
            accepted_local_mask[cand_keep_local] = True

            if exclude_tried:
                self.tried_mask[cand_keep_global] = True

        if len(accepted_global) < int(self.block_size):
            raise RuntimeError(
                f"Could not fill block_size={self.block_size} with tau={tau}. "
                f"Got {len(accepted_global)} after {tries} tries. "
                f"Lower tau or increase max_sampling_tries."
            )

        # finalize exactly like original PRBCD: GLOBAL linear ids
        self.current_search_space = torch.tensor(accepted_global, device=self.device, dtype=torch.long)
        self.current_search_space = torch.unique(self.current_search_space, sorted=True)

        # GLOBAL modified_edge_index for downstream perturbation code
        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(int(self.n), self.current_search_space)
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(int(self.n), self.current_search_space)
            is_not_self = self.modified_edge_index[0] != self.modified_edge_index[1]
            self.current_search_space = self.current_search_space[is_not_self]
            self.modified_edge_index = self.modified_edge_index[:, is_not_self]

        self.perturbed_edge_weight = torch.full_like(
            self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
        )

        if self.current_search_space.size(0) < int(n_perturbations):
            raise RuntimeError("Block has fewer unique edges than n_perturbations. Lower tau or increase sampling.")

    def sample_direct_attack_from_linkpred_threshold(
            self,
            graph,
            n_perturbations: int = 0,
            tau: float = 0.8,  # acceptance threshold
            max_sampling_tries: int = 2_000_000,
            score_batch_size: int = 1000,
            rng_seed: int = 0,
            exclude_tried: bool = False,  # optional: never reuse edges across resamples
    ):
        if not hasattr(self, "lp_model") or self.lp_model is None:
            raise RuntimeError("self.lp_model is not set.")

        # optional global tried mask
        if exclude_tried and (not hasattr(self, "tried_mask") or self.tried_mask is None):
            self.tried_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)

        # prep features/structure + embeddings once
        X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)
        X = X.to(self.device)
        edge_index_struct = edge_index_struct.to(self.device)

        self.lp_model = self.lp_model.to(self.device).eval()
        with torch.no_grad():
            h = self.lp_model.encoder(X, edge_index_struct)

        g = torch.Generator(device=self.device)
        g.manual_seed(int(rng_seed))

        accepted = []
        accepted_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)

        tries = 0
        while len(accepted) < int(self.block_size) and tries < int(max_sampling_tries):
            tries += 1

            # sample a batch of candidate linear indices
            cand_lin = torch.randint(
                self.n_possible_edges, (int(score_batch_size),), device=self.device, generator=g
            )
            cand_lin = torch.unique(cand_lin, sorted=False)

            # exclude duplicates within this block
            cand_lin = cand_lin[~accepted_mask[cand_lin]]

            # optionally exclude globally tried
            if exclude_tried:
                cand_lin = cand_lin[~self.tried_mask[cand_lin]]

            if cand_lin.numel() == 0:
                continue

            # decode to pairs for scoring
            if self.make_undirected:
                cand_ei = PRBCD.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCD.linear_to_full_idx(self.n, cand_lin)
                is_not_self = cand_ei[0] != cand_ei[1]
                cand_lin = cand_lin[is_not_self]
                cand_ei = cand_ei[:, is_not_self]
                if cand_lin.numel() == 0:
                    continue

            # score
            with torch.no_grad():
                logits = self.lp_model.edge_head(h, cand_ei).view(-1)
                scores = torch.sigmoid(logits)

            # accept those above threshold
            keep = scores >= float(tau)
            cand_keep = cand_lin[keep]

            if cand_keep.numel() == 0:
                continue

            # add until block is full
            need = int(self.block_size) - len(accepted)
            cand_keep = cand_keep[:need]

            accepted.extend(cand_keep.tolist())
            accepted_mask[cand_keep] = True

        if len(accepted) < int(self.block_size):
            raise RuntimeError(
                f"Could not fill block_size={self.block_size} with threshold tau={tau}. "
                f"Got {len(accepted)} after {tries} tries. Lower tau or increase max_sampling_tries."
            )

        # finalize current_search_space exactly like PRBCD
        self.current_search_space = torch.tensor(accepted, device=self.device, dtype=torch.long)
        self.current_search_space = torch.unique(self.current_search_space, sorted=True)

        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)
            is_not_self = self.modified_edge_index[0] != self.modified_edge_index[1]
            self.current_search_space = self.current_search_space[is_not_self]
            self.modified_edge_index = self.modified_edge_index[:, is_not_self]

        self.perturbed_edge_weight = torch.full_like(
            self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
        )

        if exclude_tried:
            self.tried_mask[self.current_search_space] = True

        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError("Block has fewer unique edges than n_perturbations. Lower tau or increase sampling.")

    def init_search_space_from_y_out(self, y_out: torch.Tensor, n_perturbations: int = 0):
        """
        Initialize self.current_search_space, self.modified_edge_index, and
        self.perturbed_edge_weight based on the positions of ones in y_out.

        Args:
            y_out: Tensor of shape (n_possible_edges,), dtype=torch.uint8
                   where 1 marks a harmful edge flip.
            n_perturbations: optional threshold for minimum number of perturbations to initialize.
        """
        device = getattr(self, "device", "cpu")

        # ---- collect indices of harmful flips
        ones_idx = torch.nonzero(y_out, as_tuple=False).flatten()
        if ones_idx.numel() == 0:
            raise ValueError("No harmful edges found in y_out; cannot initialize search space.")

        # ---- unique, sorted
        self.current_search_space = torch.unique(ones_idx.to(device), sorted=True)

        # ---- build modified edge index
        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            self.current_search_space = self.current_search_space[is_not_self_loop]
            self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        # ---- initialize perturbed edge weights
        self.perturbed_edge_weight = torch.full_like(
            self.current_search_space,
            self.eps,
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        # ---- sanity check: at least n_perturbations if requested
        if n_perturbations and self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError(
                f"Insufficient harmful edges ({self.current_search_space.size(0)}) "
                f"for requested n_perturbations={n_perturbations}."
            )

        return

    @staticmethod
    def extract_random_induced_subgraph(
            edge_index: torch.Tensor,  # (2, E) global node ids
            num_nodes: int,  # N
            subgraph_size: int,  # m
            edge_weight: Optional[torch.Tensor] = None,  # (E,) or None
            rng_seed: int = 0,
            device: Optional[torch.device] = None,
    ) -> Tuple[
        torch.Tensor,  # sub_nodes (m,) global ids
        torch.Tensor,  # edge_index_sub (2, E_sub) local ids 0..m-1
        torch.Tensor,  # edge_weight_sub (E_sub,) float
        torch.Tensor,  # g2l mapping (N,) long, -1 if not in subgraph
        Dict[int, int],  # l2g dict (local->global) for convenience
    ]:
        """
        Uniformly samples 'subgraph_size' distinct nodes from the global graph and returns
        the *induced* subgraph on those nodes.

        - sub_nodes are GLOBAL node ids.
        - edge_index_sub uses LOCAL ids 0..m-1 (ready to feed into a GNN on the subgraph).
        - g2l maps GLOBAL->LOCAL (-1 if not selected).
        """
        if num_nodes <= 0:
            raise ValueError("num_nodes must be > 0")
        if subgraph_size < 2:
            raise ValueError("subgraph_size must be >= 2")
        if subgraph_size > num_nodes:
            raise ValueError("subgraph_size cannot exceed num_nodes")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, E)")

        if device is None:
            device = edge_index.device

        edge_index = edge_index.to(device=device, dtype=torch.long).contiguous()
        E = int(edge_index.size(1))

        if edge_weight is None:
            edge_weight = torch.ones(E, device=device, dtype=torch.float32)
        else:
            if edge_weight.ndim != 1 or edge_weight.numel() != E:
                raise ValueError("edge_weight must have shape (E,)")
            edge_weight = edge_weight.to(device=device, dtype=torch.float32).contiguous()

        # ---- sample nodes uniformly without replacement
        g = torch.Generator(device=device)
        g.manual_seed(int(rng_seed))
        perm = torch.randperm(num_nodes, generator=g, device=device)
        sub_nodes = perm[:subgraph_size].contiguous()  # (m,)

        # ---- build global->local mapping
        g2l = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        g2l[sub_nodes] = torch.arange(subgraph_size, device=device, dtype=torch.long)

        # ---- filter edges where both endpoints are in sub_nodes
        src_g = edge_index[0]
        dst_g = edge_index[1]
        src_l = g2l[src_g]
        dst_l = g2l[dst_g]
        in_sub = (src_l >= 0) & (dst_l >= 0)

        edge_index_sub = torch.stack([src_l[in_sub], dst_l[in_sub]], dim=0).contiguous()
        edge_weight_sub = edge_weight[in_sub].contiguous()

        # convenience: local->global mapping as python dict (optional)
        l2g = {int(i): int(sub_nodes[i].item()) for i in range(subgraph_size)}

        return sub_nodes, edge_index_sub, edge_weight_sub, g2l, l2g

    @staticmethod
    def extract_k_hop_induced_subgraph_of_random_node(
            edge_index: torch.Tensor,  # (2, E) global node ids
            num_nodes: int,  # N
            k: int,  # number of hops
            edge_weight: Optional[torch.Tensor] = None,  # (E,) or None
            rng_seed: int = 0,
            device: Optional[torch.device] = None,
            undirected: bool = True,  # treat graph as undirected for neighborhood
            max_nodes: Optional[int] = None,  # optional cap (subsample neighborhood if huge)
    ) -> Tuple[
        torch.Tensor,  # sub_nodes (m,) global ids
        torch.Tensor,  # edge_index_sub (2, E_sub) local ids 0..m-1
        torch.Tensor,  # edge_weight_sub (E_sub,) float
        torch.Tensor,  # g2l mapping (N,) long, -1 if not in subgraph
        Dict[int, int],  # l2g dict (local->global)
    ]:
        """
        Samples a random seed node uniformly from {0..N-1}, takes its k-hop neighborhood,
        and returns the induced subgraph on that node set.

        - sub_nodes are GLOBAL node ids.
        - edge_index_sub uses LOCAL ids 0..m-1.
        - g2l maps GLOBAL->LOCAL (-1 if not selected).
        - If undirected=True, neighborhood expansion uses edges in both directions.
        - If max_nodes is set and neighborhood is larger, it is uniformly subsampled down.
        """
        if num_nodes <= 0:
            raise ValueError("num_nodes must be > 0")
        if k < 0:
            raise ValueError("k must be >= 0")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, E)")

        if device is None:
            device = edge_index.device

        edge_index = edge_index.to(device=device, dtype=torch.long).contiguous()
        E = int(edge_index.size(1))

        if edge_weight is None:
            edge_weight = torch.ones(E, device=device, dtype=torch.float32)
        else:
            if edge_weight.ndim != 1 or edge_weight.numel() != E:
                raise ValueError("edge_weight must have shape (E,)")
            edge_weight = edge_weight.to(device=device, dtype=torch.float32).contiguous()

        # ---- sample seed node uniformly
        g = torch.Generator(device=device)
        g.manual_seed(int(rng_seed))
        seed = int(torch.randint(0, num_nodes, (1,), generator=g, device=device).item())

        # ---- build adjacency lists (CSR-like via sorting) for fast hop expansion
        # We create an "edge list" used for neighbor queries:
        src = edge_index[0]
        dst = edge_index[1]
        if undirected:
            src_all = torch.cat([src, dst], dim=0)
            dst_all = torch.cat([dst, src], dim=0)
        else:
            src_all = src
            dst_all = dst

        # Sort by src_all so we can slice neighbors quickly
        perm = torch.argsort(src_all)
        src_sorted = src_all[perm]
        dst_sorted = dst_all[perm]

        # Build pointer array: rowptr[i]..rowptr[i+1] are positions of neighbors of node i
        # counts per node
        deg = torch.bincount(src_sorted, minlength=num_nodes)
        rowptr = torch.zeros(num_nodes + 1, device=device, dtype=torch.long)
        rowptr[1:] = torch.cumsum(deg, dim=0)

        def neighbors(nodes: torch.Tensor) -> torch.Tensor:
            """Return unique 1-hop neighbors of a set of nodes."""
            # gather all neighbor slices
            # This loop is fine for moderate |nodes|; for huge neighborhoods consider batching.
            chunks = []
            for u in nodes.tolist():
                start = int(rowptr[u].item())
                end = int(rowptr[u + 1].item())
                if end > start:
                    chunks.append(dst_sorted[start:end])
            if not chunks:
                return torch.empty(0, device=device, dtype=torch.long)
            return torch.unique(torch.cat(chunks, dim=0), sorted=False)

        # ---- k-hop BFS expansion
        visited = torch.zeros(num_nodes, device=device, dtype=torch.bool)
        frontier = torch.tensor([seed], device=device, dtype=torch.long)
        visited[seed] = True

        for _ in range(k):
            nbrs = neighbors(frontier)
            if nbrs.numel() == 0:
                break
            new = nbrs[~visited[nbrs]]
            if new.numel() == 0:
                break
            visited[new] = True
            frontier = new

        sub_nodes = visited.nonzero(as_tuple=False).view(-1)  # global ids, (m,)
        if sub_nodes.numel() < 2:
            # still return something consistent (just seed node)
            sub_nodes = torch.tensor([seed], device=device, dtype=torch.long)

        # ---- optional cap (uniform subsample)
        if max_nodes is not None and sub_nodes.numel() > int(max_nodes):
            perm_nodes = torch.randperm(sub_nodes.numel(), generator=g, device=device)[: int(max_nodes)]
            sub_nodes = sub_nodes[perm_nodes].contiguous()

            # rebuild visited mask to match subsample
            visited = torch.zeros(num_nodes, device=device, dtype=torch.bool)
            visited[sub_nodes] = True

        # ---- build global->local mapping
        m = int(sub_nodes.numel())
        g2l = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        g2l[sub_nodes] = torch.arange(m, device=device, dtype=torch.long)

        # ---- induced edges: both endpoints in sub_nodes
        src_g = edge_index[0]
        dst_g = edge_index[1]
        src_l = g2l[src_g]
        dst_l = g2l[dst_g]
        in_sub = (src_l >= 0) & (dst_l >= 0)

        edge_index_sub = torch.stack([src_l[in_sub], dst_l[in_sub]], dim=0).contiguous()
        edge_weight_sub = edge_weight[in_sub].contiguous()

        # convenience local->global dict
        l2g = {int(i): int(sub_nodes[i].item()) for i in range(m)}

        return sub_nodes, edge_index_sub, edge_weight_sub, g2l, l2g

    @staticmethod
    def extract_growing_hop_subgraph_of_random_node(
            edge_index: torch.Tensor,  # (2, E) global node ids
            num_nodes: int,  # N
            subgraph_size: int,  # desired #nodes in subgraph
            edge_weight: Optional[torch.Tensor] = None,  # (E,) or None
            rng_seed: int = 0,
            device: Optional[torch.device] = None,
            undirected: bool = True,  # treat graph as undirected for expansion
            max_hops: Optional[int] = None,  # optional safety cap
    ) -> Tuple[
        torch.Tensor,  # sub_nodes (m,) global ids
        torch.Tensor,  # edge_index_sub (2, E_sub) local ids 0..m-1
        torch.Tensor,  # edge_weight_sub (E_sub,) float
        torch.Tensor,  # g2l mapping (N,) long, -1 if not in subgraph
        Dict[int, int],  # l2g dict (local->global)
        int,  # hops_used
    ]:
        """
        Samples a random seed node uniformly from {0..N-1}, starts with its 1-hop neighborhood,
        then keeps expanding by another 1-hop layer (i.e., increases hop radius 1,2,3,...) until
        at least `subgraph_size` nodes are collected (or expansion saturates / max_hops reached).
        Returns the induced subgraph on the collected node set.

        - Expansion is BFS-like, layer by layer.
        - If the boundary expansion overshoots the target, it uniformly subsamples the LAST added layer
          so the final node count is exactly `subgraph_size` (when possible).
        - If the reachable component is smaller than `subgraph_size`, returns the full component.

        Returns:
            sub_nodes (global ids), edge_index_sub (local ids), edge_weight_sub,
            g2l (global->local), l2g (local->global), hops_used
        """
        if num_nodes <= 0:
            raise ValueError("num_nodes must be > 0")
        if subgraph_size <= 0:
            raise ValueError("subgraph_size must be > 0")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, E)")

        if device is None:
            device = edge_index.device

        edge_index = edge_index.to(device=device, dtype=torch.long).contiguous()
        E = int(edge_index.size(1))

        if edge_weight is None:
            edge_weight = torch.ones(E, device=device, dtype=torch.float32)
        else:
            if edge_weight.ndim != 1 or edge_weight.numel() != E:
                raise ValueError("edge_weight must have shape (E,)")
            edge_weight = edge_weight.to(device=device, dtype=torch.float32).contiguous()

        # ---- RNG + sample seed node
        g = torch.Generator(device=device)
        g.manual_seed(int(rng_seed))
        seed = int(torch.randint(0, num_nodes, (1,), generator=g, device=device).item())

        # ---- build CSR-like neighbor access (sorted by src)
        src = edge_index[0]
        dst = edge_index[1]
        if undirected:
            src_all = torch.cat([src, dst], dim=0)
            dst_all = torch.cat([dst, src], dim=0)
        else:
            src_all = src
            dst_all = dst

        perm = torch.argsort(src_all)
        src_sorted = src_all[perm]
        dst_sorted = dst_all[perm]

        deg = torch.bincount(src_sorted, minlength=num_nodes)
        rowptr = torch.zeros(num_nodes + 1, device=device, dtype=torch.long)
        rowptr[1:] = torch.cumsum(deg, dim=0)

        def neighbors_1hop(nodes: torch.Tensor) -> torch.Tensor:
            """Return unique 1-hop neighbors of a set of nodes."""
            chunks = []
            for u in nodes.tolist():
                start = int(rowptr[u].item())
                end = int(rowptr[u + 1].item())
                if end > start:
                    chunks.append(dst_sorted[start:end])
            if not chunks:
                return torch.empty(0, device=device, dtype=torch.long)
            return torch.unique(torch.cat(chunks, dim=0), sorted=False)

        # ---- BFS expansion (radius grows by 1-hop layers)
        visited = torch.zeros(num_nodes, device=device, dtype=torch.bool)
        visited[seed] = True

        # start with frontier = seed, but we want "k=1 neighborhood" as first expansion
        frontier = torch.tensor([seed], device=device, dtype=torch.long)

        hops_used = 0
        # we will keep track of the last added layer (for controlled subsampling)
        while visited.sum().item() < subgraph_size:
            if max_hops is not None and hops_used >= int(max_hops):
                break

            nbrs = neighbors_1hop(frontier)
            if nbrs.numel() == 0:
                break

            new = nbrs[~visited[nbrs]]
            if new.numel() == 0:
                break

            hops_used += 1

            # If adding the whole layer would exceed target, subsample from this layer
            need = int(subgraph_size - visited.sum().item())
            if new.numel() > need:
                pick = torch.randperm(new.numel(), generator=g, device=device)[:need]
                new = new[pick]

            visited[new] = True
            frontier = new

        sub_nodes = visited.nonzero(as_tuple=False).view(-1)  # (m,) global ids

        if sub_nodes.numel() == 0:
            sub_nodes = torch.tensor([seed], device=device, dtype=torch.long)

        # ---- build global->local mapping
        m = int(sub_nodes.numel())
        g2l = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        g2l[sub_nodes] = torch.arange(m, device=device, dtype=torch.long)

        # ---- induced edges: both endpoints in sub_nodes
        src_g = edge_index[0]
        dst_g = edge_index[1]
        src_l = g2l[src_g]
        dst_l = g2l[dst_g]
        in_sub = (src_l >= 0) & (dst_l >= 0)

        edge_index_sub = torch.stack([src_l[in_sub], dst_l[in_sub]], dim=0).contiguous()
        edge_weight_sub = edge_weight[in_sub].contiguous()

        l2g = {int(i): int(sub_nodes[i].item()) for i in range(m)}

        return sub_nodes, edge_index_sub, edge_weight_sub, g2l, l2g, hops_used

    @staticmethod
    def build_lp_struct_for_subgraph(
            X: torch.Tensor,  # (N, F) global node features
            edge_index_struct: torch.Tensor,  # (2, E) global ids
            sub_nodes: torch.Tensor,  # (m,) global ids; index = local id
            device: Optional[torch.device] = None,
            undirected: bool = False,  # set True if you want to symmetrize structure
    ) -> Tuple[
        torch.Tensor,  # X_sub (m, F)
        torch.Tensor,  # edge_index_struct_local (2, E_sub) local ids 0..m-1
    ]:
        """
        Builds subgraph-only LP training inputs:
          - X_sub: features restricted to sub_nodes (row i corresponds to local node i)
          - edge_index_struct_local: induced subgraph edges, relabeled to local ids 0..m-1

        NOTE: local node i corresponds to global node sub_nodes[i].
        """
        if device is None:
            device = X.device

        X = X.to(device=device)
        edge_index_struct = edge_index_struct.to(device=device, dtype=torch.long).contiguous()
        sub_nodes = sub_nodes.to(device=device, dtype=torch.long).contiguous()

        N = int(X.size(0))
        m = int(sub_nodes.numel())

        # --- local feature matrix ---
        X_sub = X[sub_nodes].contiguous()  # (m, F)

        # --- global -> local map ---
        g2l = torch.full((N,), -1, device=device, dtype=torch.long)
        g2l[sub_nodes] = torch.arange(m, device=device, dtype=torch.long)

        # --- filter edges inside the node set ---
        src_g = edge_index_struct[0]
        dst_g = edge_index_struct[1]
        src_l = g2l[src_g]
        dst_l = g2l[dst_g]
        in_sub = (src_l >= 0) & (dst_l >= 0)

        edge_index_struct_local = torch.stack([src_l[in_sub], dst_l[in_sub]], dim=0).contiguous()

        # optional: symmetrize structure if desired
        if undirected and edge_index_struct_local.numel() > 0:
            rev = edge_index_struct_local.flip(0)
            edge_index_struct_local = torch.cat([edge_index_struct_local, rev], dim=1)

        return X_sub, edge_index_struct_local

    def append_search_space_with_y_out(self, y_out: torch.Tensor, n_perturbations: int = 0):
        """
        Drop half of the current block according to keep_heuristic='WeightOnly',
        then append harmful edges from y_out, and finally refill the block up to
        self.block_size with random edges (preserving existing weights).

        Args:
            y_out: Tensor of shape (n_possible_edges,), dtype=torch.uint8
                   where 1 marks a harmful edge flip.
            n_perturbations: optional lower bound used as a stopping condition
                             for the refill loop (like in resample_random_block).
        """
        import torch

        device = getattr(self, "device", "cpu")

        # -----------------------------
        # 1) KEEP PHASE: drop low weights
        # -----------------------------
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)  # ascending
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]

        kept_lin = self.current_search_space[sorted_idx].to(device)
        kept_w = self.perturbed_edge_weight[sorted_idx].to(device)

        self.current_search_space = kept_lin
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx].to(device)
        self.perturbed_edge_weight = kept_w

        # -----------------------------
        # 2) APPEND PHASE: add y_out harmful indices
        # -----------------------------
        ones_idx = torch.nonzero(y_out, as_tuple=False).flatten()
        if ones_idx.numel() == 0:
            raise ValueError("No harmful edges found in y_out; cannot append to search space.")

        ones_idx = ones_idx.to(device)

        # concatenate kept edges + new harmful edges, then unique+sorted
        concat_lin = torch.cat([self.current_search_space, ones_idx], dim=0)

        new_search_space, inv = torch.unique(
            concat_lin,
            sorted=True,
            return_inverse=True
        )

        num_kept = kept_lin.size(0)
        pos_kept = inv[:num_kept]  # positions of old kept edges in the new search space

        # rebuild perturbed_edge_weight: old kept weights preserved, new edges get eps
        new_w = torch.full(
            (new_search_space.size(0),),
            self.eps,
            dtype=torch.float32,
            device=device,
        )
        new_w[pos_kept] = kept_w

        self.current_search_space = new_search_space
        self.perturbed_edge_weight = new_w

        # build modified_edge_index for this augmented block
        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            self.current_search_space = self.current_search_space[is_not_self_loop]
            self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
            self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

    def _node_saliency_from_victim(self,
                                   subset: str = "attack",
                                   loss_type: str = "ce") -> torch.Tensor:
        """
        Compute per-node saliency from the victim's classification loss via a
        node-level gate. Returns a length-N tensor of non-negative saliencies.

        subset: "attack" (= self.idx_attack), "all", or a sequence/tensor of node indices.
        """
        # Put victim in eval mode; this does NOT disable grads.
        self.attacked_model.eval()

        # ---- helpers ---------------------------------------------------------
        def _as_long_idx(idx_like):
            if torch.is_tensor(idx_like):
                return idx_like.to(self.device, dtype=torch.long)
            return torch.as_tensor(idx_like, device=self.device, dtype=torch.long)

        def _as_labels(y_like):
            if torch.is_tensor(y_like):
                return y_like.to(self.device, dtype=torch.long)
            return torch.as_tensor(y_like, device=self.device, dtype=torch.long)

        # Choose which nodes contribute to the loss
        if subset == "attack":
            idx = _as_long_idx(self.idx_attack)
        elif subset == "all":
            idx = torch.arange(self.n, device=self.device, dtype=torch.long)
        else:
            idx = _as_long_idx(subset)

        # Handle empty selection gracefully
        if idx.numel() == 0:
            return torch.zeros(self.n, device=self.device)

        # ---- data on the correct device/dtypes -------------------------------
        x = self.attr.to(self.device).float()  # (N, F)

        # IMPORTANT: use the *current* (possibly perturbed) adjacency if available
        if hasattr(self, "get_modified_adj"):
            ei, ew = self.get_modified_adj()
            ei = ei.to(self.device)
            ew = ew.to(self.device).float()
        else:
            ei = self.edge_index.to(self.device)
            ew = (self.edge_weight.to(self.device).float()
                  if getattr(self, "edge_weight", None) is not None
                  else torch.ones(ei.size(1), device=self.device))

        y = _as_labels(self.labels)

        # ---- compute logits with grad enabled -------------------------------
        with torch.enable_grad():
            # node gates (one scalar per node) to measure sensitivity
            g = torch.zeros(self.n, device=self.device).requires_grad_(True)  # (N,)
            x_gated = x * (1.0 + g.unsqueeze(1))  # broadcast to (N, F)

            logits = self.attacked_model(data=x_gated, adj=(ei, ew))  # (N, C)

            if loss_type == "tanhMargin":
                # margin = logit_true - max_{k != true} logit_k ; we *minimize* margin
                logits_sel = logits[idx]  # (M, C)
                y_sel = y[idx]  # (M,)
                true_logit = logits_sel.gather(1, y_sel.view(-1, 1)).squeeze(1)  # (M,)
                masked = logits_sel.clone()
                masked[torch.arange(masked.size(0), device=masked.device), y_sel] = -1e30
                max_other = masked.max(dim=1).values
                margin = true_logit - max_other
                loss = torch.tanh(margin).mean().neg()  # -mean(tanh(margin))
            else:
                # default: cross-entropy over the chosen subset
                loss = F.cross_entropy(logits[idx], y[idx])

            # gradient of loss wrt node gates
            (grad_g,) = torch.autograd.grad(loss, g, retain_graph=False, create_graph=False)

        saliency = grad_g.abs().detach()  # (N,)
        return saliency

    def _gumbel_topk(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Gumbel-top-k without replacement on a 1D logits tensor.
        Returns indices of the k sampled items (k clipped to len).
        """
        k = int(min(k, logits.numel()))
        if k <= 0 or logits.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=logits.device)
        # Add i.i.d. Gumbel noise: g = -log(-log(U))
        g = -torch.log(-torch.log(torch.clamp(torch.rand_like(logits), 1e-12, 1. - 1e-12)))
        return torch.topk(logits + g, k=k, largest=True).indices

    def _dense_from_edge_index(self):
        N = int(self.n)
        A = torch.zeros((N, N), dtype=torch.float32, device=self.device)
        ei = self.edge_index.to(self.device)
        A[ei[0].long(), ei[1].long()] = 1.0
        A[ei[1].long(), ei[0].long()] = 1.0
        A.fill_diagonal_(0.0)
        return A

    def _normalize_dense(self, A):
        N = A.size(0)
        A_tilde = A + torch.eye(N, dtype=A.dtype, device=A.device)
        deg = A_tilde.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
        D = torch.diag(deg_inv_sqrt)
        return D @ A_tilde @ D

    def make_pgd_forward_from_victim(self):
        # attach a function that maps (X, Abar_dense) -> logits via attacked_model
        def fwd(X, Abar):
            # convert dense normalized adjacency to sparse edge_index/edge_weight
            idx = Abar.nonzero(as_tuple=False).t()  # (2, E)
            w = Abar[idx[0], idx[1]]  # (E,)
            return self.attacked_model(data=X, adj=(idx, w))

        self.pgd_forward = fwd

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

    def resample_block_from_linkpred_model(
            self,
            graph,
            n_perturbations: int,
            pool_size: int = 200_000,  # larger pool helps once tried_mask grows
            score_batch_size: int = 200_000,
            rng_seed: int = 0,
            strict: bool = True,  # if True: never reuse tried edges; if False: fallback allowed
    ):
        if not hasattr(self, "lp_model") or self.lp_model is None:
            raise RuntimeError("self.lp_model is not set.")

        # ----- ensure tried_mask exists -----
        if not hasattr(self, "tried_mask") or self.tried_mask is None:
            self.tried_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)
            # mark whatever is currently in the block as tried (safe)
            if hasattr(self, "current_search_space") and self.current_search_space.numel() > 0:
                self.tried_mask[self.current_search_space] = True

        # ----- KEEP STEP (same as your code) -----
        if self.keep_heuristic == "WeightOnly":
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError("Only keep_heuristic=`WeightOnly` supported")

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # ----- PREP: compute embeddings once (you can cache this outside for speed) -----
        X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)
        X = X.to(self.device)
        edge_index_struct = edge_index_struct.to(self.device)

        self.lp_model = self.lp_model.to(self.device).eval()
        with torch.no_grad():
            h = self.lp_model.encoder(X, edge_index_struct)  # (N,d)

        def score_pairs(edge_index_lab: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                logits = self.lp_model.edge_head(h, edge_index_lab).view(-1)
                return torch.sigmoid(logits)

        g = torch.Generator(device=self.device)
        g.manual_seed(int(rng_seed))

        # helper: decode lin -> edge_index for scoring
        def decode_lin(cand_lin: torch.Tensor) -> torch.Tensor:
            if self.make_undirected:
                return PRBCD.linear_to_triu_idx(self.n, cand_lin)
            ei = PRBCD.linear_to_full_idx(self.n, cand_lin)
            is_not_self = ei[0] != ei[1]
            return ei[:, is_not_self], cand_lin[is_not_self]

        # ----- REFILL UNTIL block_size -----
        for _ in range(self.max_final_samples):
            n_needed = int(self.block_size - self.current_search_space.numel())
            if n_needed <= 0:
                # restore weights like PRBCD expects and exit
                # (weights already restored below after merge)
                if self.current_search_space.size(0) > n_perturbations:
                    return
                # If weird corner case, continue
                continue

            # Build masks for "currently in block"
            block_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)
            block_mask[self.current_search_space] = True

            # 1) sample candidate pool (linear indices)
            cand_lin = torch.randint(self.n_possible_edges, (int(pool_size),), device=self.device, generator=g)
            cand_lin = torch.unique(cand_lin, sorted=False)

            # 2) filter: unseen AND not currently in block
            unseen_ok = (~self.tried_mask[cand_lin]) & (~block_mask[cand_lin])
            cand_lin = cand_lin[unseen_ok]

            # If strict, and we got nothing, we can't fill
            if cand_lin.numel() == 0:
                if strict:
                    raise RuntimeError(
                        "No unseen candidates available to fill block. Increase pool_size or stop earlier.")
                # fallback: allow tried but not in block
                cand_lin = torch.randint(self.n_possible_edges, (int(pool_size),), device=self.device, generator=g)
                cand_lin = torch.unique(cand_lin, sorted=False)
                cand_lin = cand_lin[~block_mask[cand_lin]]
                if cand_lin.numel() == 0:
                    continue

            # 3) decode and (optionally) filter self-loops
            if self.make_undirected:
                cand_ei = PRBCD.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCD.linear_to_full_idx(self.n, cand_lin)
                is_not_self = cand_ei[0] != cand_ei[1]
                cand_lin = cand_lin[is_not_self]
                cand_ei = cand_ei[:, is_not_self]
                if cand_lin.numel() == 0:
                    continue

            # 4) score candidates in chunks
            M = cand_lin.numel()
            scores = torch.empty((M,), device=self.device, dtype=torch.float32)
            start = 0
            while start < M:
                end = min(start + int(score_batch_size), M)
                scores[start:end] = score_pairs(cand_ei[:, start:end])
                start = end

            # 5) take top n_needed
            k_fill = min(n_needed, M)
            top_idx = torch.topk(scores, k=k_fill, largest=True).indices
            fill_lin = cand_lin[top_idx]

            # 6) merge like PRBCD: keep old weights, init new at eps
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, fill_lin)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32
            )
            self.perturbed_edge_weight[unique_idx[:perturbed_edge_weight_old.numel()]] = perturbed_edge_weight_old

            if not self.make_undirected:
                is_not_self = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self]

            # 7) mark as tried (THIS is what ensures “never evaluated twice”)
            self.tried_mask[self.current_search_space] = True

            # success
            if self.current_search_space.size(0) >= self.block_size and self.current_search_space.size(
                    0) > n_perturbations:
                return

        raise RuntimeError(
            "Model-guided unseen resampling failed to fill block_size. "
            "Increase pool_size or use strict=False fallback."
        )

    def resample_block_from_linkpred_threshold(
            self,
            graph,
            n_perturbations: int,
            tau: float = 0.7,
            max_sampling_tries: int = 2_000_000,
            score_batch_size: int = 10000,
            rng_seed: int = 0,
            exclude_tried: bool = True,
    ):
        """
        Resampling function, that encodes the link-pred GNN on the *current perturbed graph*
        (via self.get_modified_adj()) instead of the unperturbed input `graph`.

        Notes:
          - We binarize the current attacked adjacency for the LP encoder by default:
                keep edges with edge_weight > 0.5
            (Adjust the threshold below if your LP model expects something else.)
          - Candidate sampling is done from a restricted "allowed pool" derived from a single
            combined block mask (block + tried), rather than sampling from the full space
            and then cutting.
        """

        import torch

        if not hasattr(self, "lp_model") or self.lp_model is None:
            raise RuntimeError("self.lp_model is not set.")

        # -----------------------------
        # tried_mask init
        # -----------------------------
        if exclude_tried and (not hasattr(self, "tried_mask") or self.tried_mask is None):
            self.tried_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)
            if hasattr(self, "current_search_space") and self.current_search_space.numel() > 0:
                self.tried_mask[self.current_search_space] = True

        # -----------------------------
        # keep step (same as PRBCD)
        # -----------------------------
        if self.keep_heuristic == "WeightOnly":
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError("Only keep_heuristic=`WeightOnly` supported")

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # -----------------------------
        # Build *perturbed* structure for LP encoder
        # -----------------------------
        # Features still come from `graph`, but structure comes from get_modified_adj()
        X, _ = self.extract_X_and_edge_index_from_sparsegraph(graph)
        X = X.to(self.device)

        with torch.no_grad():
            edge_index_mod, edge_weight_mod = self.get_modified_adj()
            edge_index_mod = edge_index_mod.to(self.device)
            edge_weight_mod = edge_weight_mod.to(self.device).float()

            edge_index_struct = edge_index_mod

            '''# Binarize attacked adjacency for message passing
            present = edge_weight_mod > 0.5
            edge_index_struct = edge_index_mod[:, present]'''

        # encode once
        self.lp_model = self.lp_model.to(self.device).eval()
        with torch.no_grad():
            h = self.lp_model.encoder(X, edge_index_struct)

        g = torch.Generator(device=self.device)
        g.manual_seed(int(rng_seed))

        # -----------------------------
        # masks for uniqueness within the block
        # -----------------------------
        block_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)
        if self.current_search_space.numel() > 0:
            block_mask[self.current_search_space] = True

        # Combine masks: "blocked" = in block OR (optionally) already tried
        blocked_mask = block_mask.clone()
        if exclude_tried:
            blocked_mask |= self.tried_mask

        # Allowed pool: indices we can sample from (NOT blocked)
        allowed_pool = torch.nonzero(~blocked_mask, as_tuple=False).view(-1)

        accepted = []
        tries = 0
        n_needed = int(self.block_size - self.current_search_space.numel())

        # If we already have enough, skip refill loop
        if n_needed < 0:
            n_needed = 0

        while len(accepted) < n_needed and tries < int(max_sampling_tries):
            tries += 1

            # Nothing left to sample from -> cannot refill
            if allowed_pool.numel() == 0:
                break

            # sample directly from allowed pool (restricted sampling)
            k = min(int(score_batch_size), int(allowed_pool.numel()))
            idx = torch.randint(allowed_pool.numel(), (k,), device=self.device, generator=g)
            cand_lin = allowed_pool[idx]
            cand_lin = torch.unique(cand_lin, sorted=False)

            # Pool can be slightly stale because we accept edges and update block_mask.
            # Filter out edges that have become blocked since pool creation.
            cand_lin = cand_lin[~block_mask[cand_lin]]
            if cand_lin.numel() == 0:
                continue

            # decode to pairs
            if self.make_undirected:
                cand_ei = PRBCD.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCD.linear_to_full_idx(self.n, cand_lin)
                is_not_self = cand_ei[0] != cand_ei[1]
                cand_lin = cand_lin[is_not_self]
                cand_ei = cand_ei[:, is_not_self]
                if cand_lin.numel() == 0:
                    continue

            # score candidates
            with torch.no_grad():
                logits = self.lp_model.edge_head(h, cand_ei).view(-1)
                scores = torch.sigmoid(logits)

            keep = scores >= float(tau)
            cand_keep = cand_lin[keep]
            if cand_keep.numel() == 0:
                continue

            need = n_needed - len(accepted)
            cand_keep = cand_keep[:need]

            accepted.extend(cand_keep.tolist())

            # update block_mask for uniqueness
            block_mask[cand_keep] = True

            # also update blocked_mask + allowed_pool if you want to keep pool tight
            # (this avoids repeatedly drawing now-blocked edges)
            if cand_keep.numel() > 0:
                blocked_mask[cand_keep] = True
                # remove newly blocked edges from pool
                # (cheap: just filter pool by blocked_mask)
                allowed_pool = allowed_pool[~blocked_mask[allowed_pool]]

        if len(accepted) < n_needed:
            raise RuntimeError(
                f"Could not refill to block_size with tau={tau}. "
                f"Needed {n_needed}, got {len(accepted)} after {tries} tries. Lower tau / increase tries."
            )

        fill_lin = torch.tensor(accepted, device=self.device, dtype=torch.long)

        # -----------------------------
        # merge like PRBCD (but correct mapping of old weights)
        # -----------------------------
        old_space = self.current_search_space
        old_w = self.perturbed_edge_weight.clone()

        concat = torch.cat((old_space, fill_lin), dim=0)

        # unique edges in the new block
        new_space, inv = torch.unique(concat, sorted=True, return_inverse=True)

        # initialize all new weights to eps
        new_w = torch.full((new_space.numel(),), float(self.eps), device=self.device, dtype=torch.float32)

        # map old weights into their positions in new_space
        old_len = old_space.numel()
        new_w[inv[:old_len]] = old_w

        self.current_search_space = new_space

        # rebuild modified_edge_index from new_space
        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)

        self.perturbed_edge_weight = new_w

        # directed: remove self-loops if any (keep tensors aligned)
        if not self.make_undirected:
            is_not_self = self.modified_edge_index[0] != self.modified_edge_index[1]
            self.current_search_space = self.current_search_space[is_not_self]
            self.modified_edge_index = self.modified_edge_index[:, is_not_self]
            self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self]

        if exclude_tried:
            self.tried_mask[self.current_search_space] = True

        if self.current_search_space.size(0) <= n_perturbations:
            raise RuntimeError("Block ended up too small. Lower tau.")

    def label_edge_flips_prbcd_selfsample(
            self,
            n_candidates: int = 10,
            p_add: float = 0.5,
            p_del: float = 0.5,
            drop_threshold: float = 0.001,
            rng_seed: int = 0,
            max_sampling_tries: int = 10000
    ) -> tuple[Tensor, set[tuple[int, float]], set[tuple[int, float]]]:
        """
        Samples flip candidates using self.acc_sampler_sample_one_edge(), evaluates each flip,
        and returns:
          - y_out: vector of shape (self.n_possible_edges,) with 1s at indices of harmful flips
          - harmful_set: a set of (lin_edge_idx, drop) pairs for each harmful flip

        Indexing:
          - If self.make_undirected: uses upper-tri linear indexing via
            PRBCD.triu_idx_to_linear_idx(n, [[u],[v]]) with u<v.
          - Else (directed): uses full indexing u*n + v via PRBCD.full_to_linear_idx.
        """
        import numpy as np
        import torch

        device = getattr(self, "device", "cpu")

        # --- clean graph edge arrays (CPU lists for quick edits) ---
        edge_index_clean = self.edge_index.cpu().clone()
        if getattr(self, "edge_weight", None) is None:
            edge_weight_clean = torch.ones(edge_index_clean.size(1), dtype=torch.float32)
        else:
            edge_weight_clean = self.edge_weight.cpu().clone().float()

        u_arr = edge_index_clean[0].numpy().astype(int).tolist()
        v_arr = edge_index_clean[1].numpy().astype(int).tolist()
        w_arr = edge_weight_clean.numpy().tolist()

        # existing undirected set (min,max) for quick membership checks
        existing_set = set()
        for a, b in zip(u_arr, v_arr):
            if a == b:
                continue
            existing_set.add((a if a < b else b, b if a < b else a))

        # --- baseline accuracy once on clean graph ---
        with torch.no_grad():
            a = self.edge_index.to(device)
            clean_ew = (self.edge_weight.to(device) if getattr(self, "edge_weight", None) is not None
                        else torch.ones(self.edge_index.size(1), device=device))
            logits_clean = self.attacked_model(data=self.attr.to(device),
                                               adj=(self.edge_index.to(device), clean_ew))
            acc_clean = utils.accuracy(logits_clean, self.labels.to(device), self.idx_attack)

        # --- prepare outputs ---
        y_out = torch.zeros(int(self.n_possible_edges), dtype=torch.uint8, device=device)
        tried_set: Set[Tuple[int, float]] = set()  # (linear_edge_index, drop)
        harmful_set: Set[Tuple[int, float]] = set()  # (linear_edge_index, drop)

        # helper: build coalesced sparse tensors
        def make_sparse_edge_tensors(u_list, v_list, w_list):
            idx = torch.tensor([u_list, v_list], dtype=torch.long)
            w = torch.tensor(w_list, dtype=torch.float32)
            try:
                import torch_sparse
                idx_coal, w_coal = torch_sparse.coalesce(idx, w, m=self.n, n=self.n, op='sum')
                return idx_coal.to(device), w_coal.to(device)
            except Exception:
                coo = torch.sparse_coo_tensor(idx, w, (self.n, self.n)).coalesce()
                return coo.indices().to(device), coo.values().to(device)

        # mapping (u,v) -> linear index over "all possible edges"
        def lin_idx_for_pair(u: int, v: int) -> int:
            if self.make_undirected:
                uu, vv = (u, v) if u < v else (v, u)
                full = torch.tensor([[uu], [vv]], dtype=torch.long)
                return int(PRBCD.triu_idx_to_linear_idx(self.n, full).item())
            else:
                full = torch.tensor([[u], [v]], dtype=torch.long)
                return int(PRBCD.full_to_linear_idx(self.n, full).item())

        # --- sample candidates and evaluate ---
        rng = np.random.default_rng(rng_seed)
        seen = set()
        tries = 0
        flips_done = 0  # count of flips we actually perform/evaluate

        # Ensure we do at least one flip; then continue until n_candidates flips (or tries exhausted)
        while (flips_done == 0 or flips_done < n_candidates) and tries < max_sampling_tries:
            cand = self.acc_sampler_sample_one_edge(
                p_add=p_add, p_del=p_del,
                rng_seed=int(rng.integers(0, 2 ** 31 - 1))
            )
            tries += 1
            if not cand:
                continue  # resample until we get a candidate

            u, v, action = cand[0]
            key = (int(u), int(v), str(action))
            if key in seen:
                continue  # avoid evaluating the same flip twice
            seen.add(key)

            # start from clean lists
            u_list = u_arr[:]
            v_list = v_arr[:]
            w_list = w_arr[:]

            if action == 'add':
                if self.make_undirected:
                    if (min(u, v), max(u, v)) not in existing_set:
                        u_list += [int(u), int(v)]
                        v_list += [int(v), int(u)]
                        w_list += [1.0, 1.0]
                else:
                    u_list.append(int(u));
                    v_list.append(int(v));
                    w_list.append(1.0)
            else:  # 'del'
                new_u, new_v, new_w = [], [], []
                if self.make_undirected:
                    for a_, b_, w_ in zip(u_list, v_list, w_list):
                        if (a_ == u and b_ == v) or (a_ == v and b_ == u):
                            continue
                        new_u.append(a_);
                        new_v.append(b_);
                        new_w.append(w_)
                else:
                    for a_, b_, w_ in zip(u_list, v_list, w_list):
                        if (a_ == u and b_ == v):
                            continue
                        new_u.append(a_);
                        new_v.append(b_);
                        new_w.append(w_)
                u_list, v_list, w_list = new_u, new_v, new_w

            # build perturbed adjacency
            try:
                pert_ei, pert_ew = make_sparse_edge_tensors(u_list, v_list, w_list)
            except Exception:
                continue  # skip this flip if assembly failed

            # evaluate drop
            with torch.no_grad():
                logits_pert = self.attacked_model(data=self.attr.to(device), adj=(pert_ei, pert_ew))
                acc_pert = utils.accuracy(logits_pert, self.labels.to(device), self.idx_attack)
            drop = float(acc_clean - acc_pert)
            lin = lin_idx_for_pair(int(u), int(v))
            tried_set.add((lin, drop))

            if drop > drop_threshold:
                harmful_set.add((lin, drop))
                y_out[lin] = 1
                flips_done += 1  # count this evaluated flip

        return y_out, tried_set, harmful_set

    def label_edge_flips_prbcd_selfsample_fast(
            self,
            n_perturbations,
            n_candidates_one_sample: int = 2000,
            n_candidates_k_sample: int = 2000,
            p_add: float = 0.06,  # unused in uniform flip
            p_del: float = 0.1,  # unused in uniform flip
            acc_drop_threshold_one_sample: float = 3e-3,
            acc_drop_threshold_k_samples: float = -1,
            acc_drop_threshold_k_hop: float = 1e-2,
            drop_mode: str = "acc",  # "acc" | "loss" | "endpoint" | "endpointPRBCD"
            loss_drop_threshold_one_sample: float = 1e-3,
            loss_drop_threshold_k_samples: float = 1e-3,
            loss_drop_threshold_k_hop: float = 1e-3,
            rng_seed: int = 0,
            max_sampling_tries: int = 100_000,
            mode: str = "k_hop",  # one_sample | node_matching | k_action | k_action_individual | k_hop
            k_samples_batch: int = 10,  # only used in k_action
            n_candidates_k_hop_sample: int = 2000,  # target # of harmful flips in k_hop mode
            k_hop: int = 2,  # hop radius for k_hop mode
            prev_tried_set: "Set[tuple[int, float]] | None" = None,
            training_data_node_cap: int = 0
    ) -> tuple[torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Set[tuple[int, float]],
    Set[tuple[int, float]]]:

        import torch
        from typing import Set, Tuple, Optional

        device = getattr(self, "device", "cpu")

        n: int = int(self.n)
        if n <= 1:
            raise ValueError("Graph must have at least 2 nodes.")

        training_data_node_cap = int(training_data_node_cap)
        if training_data_node_cap < 0:
            raise ValueError("training_data_node_cap must be >= 0. Use 0 to disable the cap.")

        if drop_mode == "endpointPRBCD" and mode not in ("one_sample", "node_matching"):
            raise ValueError(
                "drop_mode='endpointPRBCD' only supports mode='one_sample' "
                "or mode='node_matching'."
            )

        # ---- base adjacency
        ei_base = self.edge_index.to(device=device, dtype=torch.long).contiguous()  # (2,E)
        ew_base = (
            self.edge_weight.to(device).float()
            if getattr(self, "edge_weight", None) is not None
            else torch.ones(ei_base.size(1), device=device, dtype=torch.float32)
        ).contiguous()

        # ---- build undirected membership bitset
        present = PRBCD._build_uppertri_bitset(ei_base, n)  # (num_pairs,) bool

        E = ei_base.size(1)
        dir_pos = torch.full((n, n), -1, dtype=torch.int32, device=device)
        dir_pos[ei_base[0], ei_base[1]] = torch.arange(E, device=device, dtype=torch.int32)

        # ---- clean forward once
        logits_clean = self.attacked_model(data=self.attr.to(device), adj=(ei_base, ew_base))
        acc_clean = utils.accuracy(logits_clean, self.labels.to(device), self.idx_attack)

        # ---- clean loss once
        idx_attack = self.idx_attack
        labels_dev = self.labels.to(device)

        loss_clean = F.cross_entropy(
            logits_clean[idx_attack],
            labels_dev[idx_attack],
            reduction="mean",
        )

        # ---- clean pred once
        pred_clean = logits_clean.argmax(dim=-1)
        clean_correct = (pred_clean == labels_dev.to(device))

        if drop_mode == "acc":
            thr_one = acc_drop_threshold_one_sample
            thr_k = acc_drop_threshold_k_samples
            thr_khop = acc_drop_threshold_k_hop
        elif drop_mode == "loss":
            thr_one = loss_drop_threshold_one_sample
            thr_k = loss_drop_threshold_k_samples
            thr_khop = loss_drop_threshold_k_hop
        elif drop_mode in ("endpoint", "endpointPRBCD"):
            thr_one = None
            thr_k = None
            thr_khop = None
        else:
            raise ValueError(
                f"Unknown drop_mode='{drop_mode}'. "
                "Use 'acc', 'loss', 'endpoint', or 'endpointPRBCD'."
            )

        # ---- outputs over *all* upper-tri pairs
        num_pairs = n * (n - 1) // 2
        y_out = torch.zeros(num_pairs, dtype=torch.uint8, device=device)

        # Base sampling pool for normal endpoint mode: all possible upper-triangle pairs.
        # If training_data_node_cap == 0, this pool is never restricted by node cap.
        endpoint_all_pool_idx = torch.arange(num_pairs, dtype=torch.long, device=device)

        # ---- lists to collect labeled pairs (only tried pairs)
        lab_u: list[int] = []
        lab_v: list[int] = []
        lab_k: list[int] = []  # linear indices k_lin for those pairs

        # start tried_set from previous run if given
        tried_set: Set[Tuple[int, float | None]] = (
            set(prev_tried_set) if prev_tried_set is not None else set()
        )
        harmful_set: Set[Tuple[int, float | None]] = set()

        # Only used for endpoint-style training-data construction.
        # cap=0 means disabled. If cap>0, the sampling pool itself is
        # restricted after a node reaches `training_data_node_cap`, so newly
        # sampled/evaluated endpoint candidates cannot touch capped nodes.
        endpoint_node_harmful_count = [0 for _ in range(n)]

        # Tracks which capped nodes were already printed, so each node is reported only once.
        endpoint_capped_nodes_reported: set[int] = set()

        # Fast duplicate check for harmful edge ids.
        harmful_k_seen: set[int] = set()

        # ---- seen mask to avoid duplicate candidates (device-side)
        seen = torch.zeros(num_pairs, dtype=torch.bool, device=device)

        # ---- helper: apply previous tried_set to seen ----
        if prev_tried_set:
            prev_indices = [idx for (idx, _drop) in prev_tried_set if 0 <= idx < num_pairs]
            if prev_indices:
                prev_tensor = torch.tensor(prev_indices, device=device, dtype=torch.long)
                seen[prev_tensor] = True

        # ---- torch RNG
        g = torch.Generator(device=device)
        g.manual_seed(int(rng_seed))

        flips_done, tries = 0, 0

        # -------------------------------------------------------------------------
        # endpointPRBCD candidate block
        # -------------------------------------------------------------------------
        prbcd_block_idx: Optional[torch.Tensor] = None

        if drop_mode == "endpointPRBCD":
            prbcd_block_idx = self._run_prbcd_for_endpoint_candidate_block(
                rng_seed=int(rng_seed),
                n_perturbations=n_perturbations,
            )

            prbcd_block_idx = prbcd_block_idx.to(device=device, dtype=torch.long).flatten()

            # keep only valid upper-tri pair indices
            prbcd_block_idx = prbcd_block_idx[
                (prbcd_block_idx >= 0) & (prbcd_block_idx < num_pairs)
                ]

            # unique block indices
            prbcd_block_idx = torch.unique(prbcd_block_idx, sorted=False)

            if prbcd_block_idx.numel() == 0:
                raise RuntimeError(
                    "endpointPRBCD produced an empty candidate block. "
                    "The PRBCD helper must return at least one valid upper-tri edge index."
                )

            # remove pairs already tried in prev_tried_set
            prbcd_block_idx = prbcd_block_idx[~seen[prbcd_block_idx]]

            if prbcd_block_idx.numel() == 0:
                raise RuntimeError(
                    "endpointPRBCD candidate block contains only previously tried edges."
                )

        # ----- helper: endpoint criterion -----
        def _endpoint_flipped_correct_to_incorrect(
                u: int,
                v: int,
                logits_pert: torch.Tensor,
        ) -> bool:
            pred_pert = logits_pert.argmax(dim=-1)

            u = int(u)
            v = int(v)

            u_flip = bool(clean_correct[u] and (pred_pert[u] != labels_dev[u].to(device)))
            v_flip = bool(clean_correct[v] and (pred_pert[v] != labels_dev[v].to(device)))

            # optional: only count flips if endpoint is in idx_attack
            # u_flip = u_flip and bool(attack_mask[u])
            # v_flip = v_flip and bool(attack_mask[v])

            return u_flip or v_flip

        # ----- helper: endpoint harmful-set node cap via sampling-pool restriction -----
        def _endpoint_node_cap_active() -> bool:
            return (
                    drop_mode in ("endpoint", "endpointPRBCD")
                    and training_data_node_cap > 0
            )

        def _restrict_endpoint_pool_by_node_cap(pool_idx: torch.Tensor) -> torch.Tensor:
            """
            Restrict a linear upper-triangle edge pool by removing all edges
            touching nodes that have already reached training_data_node_cap.

            Important: if training_data_node_cap == 0, this returns the pool
            unchanged, apart from normal tensor/device cleanup.
            """
            if pool_idx is None:
                return torch.empty(0, dtype=torch.long, device=device)

            pool_idx = pool_idx.to(device=device, dtype=torch.long).flatten()

            if pool_idx.numel() == 0:
                return pool_idx

            # Always keep the pool within the valid linear index range.
            pool_idx = pool_idx[(pool_idx >= 0) & (pool_idx < num_pairs)]

            # No node-cap restriction when cap is disabled.
            if not _endpoint_node_cap_active():
                return pool_idx

            counts = torch.tensor(
                endpoint_node_harmful_count,
                dtype=torch.long,
                device=device,
            )
            allowed_nodes = counts < int(training_data_node_cap)

            if bool(allowed_nodes.all().item()):
                return pool_idx

            uv = PRBCD.linear_to_triu_idx(n, pool_idx)
            u = uv[0]
            v = uv[1]

            keep = allowed_nodes[u] & allowed_nodes[v]
            return pool_idx[keep]

        def _available_from_endpoint_pool(pool_idx: torch.Tensor) -> torch.Tensor:
            """
            Apply the current endpoint sampling-pool rules:
              1. use the mode-specific base pool
                 - endpoint: all possible edges
                 - endpointPRBCD: PRBCD block
              2. remove already-seen/evaluated edges
              3. if nodecap > 0, remove edges touching capped nodes
            """
            pool_idx = _restrict_endpoint_pool_by_node_cap(pool_idx)

            if pool_idx.numel() == 0:
                return pool_idx

            return pool_idx[~seen[pool_idx]]

        def _refresh_endpoint_sampling_pool_after_harmful() -> None:
            """
            After a new harmful edge is accepted, node counts may have changed.
            If a node just became capped, restrict the active sampling pool so
            future samples cannot touch capped nodes.
            """
            nonlocal endpoint_all_pool_idx, prbcd_block_idx

            if not _endpoint_node_cap_active():
                return

            if drop_mode == "endpointPRBCD":
                if prbcd_block_idx is not None:
                    prbcd_block_idx = _restrict_endpoint_pool_by_node_cap(prbcd_block_idx)
            else:
                endpoint_all_pool_idx = _restrict_endpoint_pool_by_node_cap(endpoint_all_pool_idx)

        def _report_newly_capped_nodes(edge_u: int, edge_v: int, edge_k: int) -> None:
            """
            Print a message exactly once for each node when it first reaches
            training_data_node_cap.
            """
            if not _endpoint_node_cap_active():
                return

            for node in (int(edge_u), int(edge_v)):
                if (
                        endpoint_node_harmful_count[node] >= int(training_data_node_cap)
                        and node not in endpoint_capped_nodes_reported
                ):
                    endpoint_capped_nodes_reported.add(node)
                    print(
                        f"[NODE CAP REACHED] "
                        f"node={node} | "
                        f"count={endpoint_node_harmful_count[node]}/"
                        f"{int(training_data_node_cap)} | "
                        f"trigger_edge=({int(edge_u)}, {int(edge_v)}) | "
                        f"k={int(edge_k)} | "
                        f"mode={drop_mode}"
                    )

        def _record_endpoint_harmful(u: int, v: int, k_lin: int) -> bool:
            """
            Records an endpoint-harmful edge as positive training data.

            The node cap is enforced by restricting the sampling pool, not by
            throwing out already-sampled harmful edges after evaluation.
            If a node reaches the cap because of this edge, it is printed once,
            and the active endpoint sampling pool is refreshed immediately.
            """
            k_lin = int(k_lin)

            if k_lin in harmful_k_seen:
                return False

            harmful_k_seen.add(k_lin)
            y_out[k_lin] = 1
            harmful_set.add((k_lin, None))

            if _endpoint_node_cap_active():
                endpoint_node_harmful_count[int(u)] += 1
                endpoint_node_harmful_count[int(v)] += 1
                _report_newly_capped_nodes(int(u), int(v), k_lin)
                _refresh_endpoint_sampling_pool_after_harmful()

            return True

        # ----- helper: compute drop for perturbation selection -----
        def _compute_drop(logits_pert: torch.Tensor) -> float | None:
            if drop_mode == "acc":
                acc_pert = utils.accuracy(logits_pert, labels_dev, idx_attack)
                return float(acc_clean - acc_pert)

            elif drop_mode == "loss":
                loss_pert = F.cross_entropy(
                    logits_pert[idx_attack],
                    labels_dev[idx_attack],
                    reduction="mean",
                )
                return float((loss_pert - loss_clean).item())

            elif drop_mode in ("endpoint", "endpointPRBCD"):
                return None

            else:
                raise ValueError(
                    f"Unknown drop_mode='{drop_mode}'. "
                    "Use 'acc', 'loss', 'endpoint', or 'endpointPRBCD'."
                )

        # ----- helper: build perturbed adjacency fresh from a batch of flips -----
        def _build_perturbed_adj(flips):
            """
            Given a list of flips [(u, v, action, k_lin), ...],
            build (ei_use, ew_use) from scratch based on ei_base, ew_base.
            - 'del' -> set weights of (u,v) and (v,u) to 0 in the base part
            - 'add' -> append directed edges (u,v) and (v,u)
            """
            ew_use = ew_base.clone()
            del_indices = []
            add_edges = []

            for (u, v, action, _k_lin) in flips:
                if action == "del":
                    idx = int(dir_pos[u, v].item())
                    if idx >= 0:
                        del_indices.append(idx)

                    idx2 = int(dir_pos[v, u].item())
                    if idx2 >= 0:
                        del_indices.append(idx2)

                else:  # "add"
                    add_edges.append((u, v))
                    add_edges.append((v, u))

            if del_indices:
                del_idx_tensor = torch.tensor(del_indices, device=device, dtype=torch.long)
                ew_use[del_idx_tensor] = 0.0

            if add_edges:
                extra_ei = torch.tensor(add_edges, device=device, dtype=torch.long).t()  # (2, M)
                extra_ew = torch.ones(extra_ei.size(1), device=device, dtype=torch.float32)
                ei_use = torch.cat([ei_base, extra_ei], dim=1)
                ew_use = torch.cat([ew_use, extra_ew], dim=0)
            else:
                ei_use = ei_base

            return ei_use, ew_use

        # ----- helper: convert upper-tri linear index to flip tuple -----
        def _k_lin_to_flip(k_lin: int):
            k_tensor = torch.tensor([int(k_lin)], device=device, dtype=torch.long)
            uv = PRBCD.linear_to_triu_idx(n, k_tensor)

            u = int(uv[0, 0].item())
            v = int(uv[1, 0].item())

            action = "del" if bool(present[int(k_lin)].item()) else "add"

            return u, v, action, int(k_lin)

        # ----- helper: sample one flip from a mode-specific endpoint pool -----
        def _sample_one_flip_from_endpoint_pool(pool_idx: torch.Tensor):
            """
            Samples one flip from the current restricted endpoint pool.

            endpoint:      pool_idx = endpoint_all_pool_idx
            endpointPRBCD: pool_idx = prbcd_block_idx

            If training_data_node_cap == 0, no node-cap restriction is applied.
            """
            available = _available_from_endpoint_pool(pool_idx)

            if available.numel() == 0:
                return None

            pos = torch.randint(
                available.numel(),
                (1,),
                generator=g,
                device=device,
            ).item()

            k_lin = int(available[pos].item())
            return _k_lin_to_flip(k_lin)

        # ----- helper: builds n/2 matching-style flips from full endpoint pool -----
        def _sample_node_matching_batch():
            """
            Returns a matching-style batch from the endpoint sampling pool.

            For endpoint mode, the base pool is all possible upper-triangle
            edges. The current pool is then restricted by seen edges and, only
            when training_data_node_cap > 0, by capped nodes.
            """
            target = n // 2
            available = _available_from_endpoint_pool(endpoint_all_pool_idx)

            if available.numel() == 0:
                return []

            perm = torch.randperm(available.numel(), generator=g, device=device)
            available = available[perm]

            used_nodes: set[int] = set()
            batch = []

            for k_tensor in available:
                k_lin = int(k_tensor.item())
                u, v, action, k_lin = _k_lin_to_flip(k_lin)

                if u in used_nodes or v in used_nodes:
                    continue

                used_nodes.add(u)
                used_nodes.add(v)
                batch.append((u, v, action, k_lin))

                if len(batch) >= target:
                    break

            return batch

        # ----- helper: builds n/2 matching-style flips from PRBCD block -----
        def _sample_node_matching_batch_from_prbcd_block():
            """
            Returns a matching-style batch from the endpointPRBCD sampling pool.

            The base pool is the PRBCD candidate block. The current pool is
            then restricted by seen edges and, only when
            training_data_node_cap > 0, by capped nodes.
            """
            assert prbcd_block_idx is not None

            target = n // 2
            available = _available_from_endpoint_pool(prbcd_block_idx)

            if available.numel() == 0:
                return []

            perm = torch.randperm(available.numel(), generator=g, device=device)
            available = available[perm]

            used_nodes: set[int] = set()
            batch = []

            for k_tensor in available:
                k_lin = int(k_tensor.item())
                u, v, action, k_lin = _k_lin_to_flip(k_lin)

                if u in used_nodes or v in used_nodes:
                    continue

                used_nodes.add(u)
                used_nodes.add(v)
                batch.append((u, v, action, k_lin))

                if len(batch) >= target:
                    break

            return batch

        # ---- adjacency list for k_hop mode (CPU side) ----
        adj_list = None
        if mode == "k_hop":
            ei_cpu = ei_base.cpu()
            adj_list = [[] for _ in range(n)]
            src = ei_cpu[0].tolist()
            dst = ei_cpu[1].tolist()

            for u, v in zip(src, dst):
                adj_list[u].append(v)

        # -------------------------------------------------------------------------
        # switch: sampling mode
        # -------------------------------------------------------------------------
        if mode == "one_sample":
            while (flips_done == 0 or flips_done < n_candidates_one_sample) and tries < max_sampling_tries:

                if drop_mode in ("endpoint", "endpointPRBCD"):
                    if drop_mode == "endpointPRBCD":
                        sampled = _sample_one_flip_from_endpoint_pool(prbcd_block_idx)
                    else:
                        sampled = _sample_one_flip_from_endpoint_pool(endpoint_all_pool_idx)

                    tries += 1

                    if sampled is None:
                        break

                    u, v, action, k_lin = sampled

                else:
                    u, v, action, k_lin = PRBCD._sample_one_flip(n, present, g, device)
                    tries += 1

                    if seen[k_lin]:
                        continue

                seen[k_lin] = True

                batch = [(u, v, action, k_lin)]
                ei_use, ew_use = _build_perturbed_adj(batch)

                with torch.no_grad():
                    logits_pert = self.attacked_model(data=self.attr.to(device), adj=(ei_use, ew_use))

                drop = _compute_drop(logits_pert)
                tried_set.add((int(k_lin), drop))

                # record this pair as labeled (harmful or not)
                lab_u.append(int(u))
                lab_v.append(int(v))
                lab_k.append(int(k_lin))

                if drop_mode in ("loss", "acc"):
                    if drop > thr_one:
                        y_out[k_lin] = 1
                        harmful_set.add((int(k_lin), drop))
                        flips_done += 1

                        print(
                            f"[HARMFUL EDGE FOUND] "
                            f"k={int(k_lin)} | edge=({int(u)}, {int(v)}) | "
                            f"action={action} | drop={drop:.6f} | "
                            f"found={flips_done}/{n_candidates_one_sample}"
                        )

                else:
                    # endpoint and endpointPRBCD
                    if _endpoint_flipped_correct_to_incorrect(u, v, logits_pert):
                        if _record_endpoint_harmful(u, v, k_lin):
                            flips_done += 1

                            print(
                                f"[HARMFUL EDGE FOUND] "
                                f"k={int(k_lin)} | edge=({int(u)}, {int(v)}) | "
                                f"action={action} | mode={drop_mode} | "
                                f"found={flips_done}/{n_candidates_one_sample}"
                            )

        elif mode == "node_matching":
            if drop_mode not in ("endpoint", "endpointPRBCD"):
                raise ValueError(
                    "mode='node_matching' is intended for drop_mode='endpoint' "
                    "or drop_mode='endpointPRBCD' (batched endpoint check)."
                )

            # collect harmful edges until target reached
            while (flips_done == 0 or flips_done < n_candidates_k_hop_sample) and tries < max_sampling_tries:

                if drop_mode == "endpointPRBCD":
                    batch = _sample_node_matching_batch_from_prbcd_block()
                else:
                    batch = _sample_node_matching_batch()

                if not batch:
                    break

                # mark seen + count tries
                for (u, v, _action, k_lin) in batch:
                    seen[k_lin] = True
                    tries += 1

                # build one perturbed graph for the whole batch
                ei_use, ew_use = _build_perturbed_adj(batch)

                with torch.no_grad():
                    logits_pert = self.attacked_model(data=self.attr.to(device), adj=(ei_use, ew_use))

                # endpoint labeling per edge in the batch
                for (u, v, _action, k_lin) in batch:
                    tried_set.add((int(k_lin), None))

                    lab_u.append(int(u))
                    lab_v.append(int(v))
                    lab_k.append(int(k_lin))

                    if _endpoint_flipped_correct_to_incorrect(u, v, logits_pert):
                        if _record_endpoint_harmful(u, v, k_lin):
                            flips_done += 1

                # keep original stopping behavior, but use the node_matching target consistently
                if flips_done >= n_candidates_k_hop_sample:
                    break

        elif mode == "k_action":
            while (flips_done == 0 or flips_done < n_candidates_k_sample) and tries < max_sampling_tries:
                batch = PRBCD._sample_k_flips(
                    n=n,
                    present=present,
                    g=g,
                    device=device,
                    k=int(k_samples_batch),
                    seen=seen,
                )

                if not batch:
                    break

                for (u, v, _action, k_lin) in batch:
                    seen[k_lin] = True
                    tries += 1

                ei_use, ew_use = _build_perturbed_adj(batch)

                with torch.no_grad():
                    logits_pert = self.attacked_model(data=self.attr.to(device), adj=(ei_use, ew_use))

                drop = _compute_drop(logits_pert)

                for (u, v, _action, k_lin) in batch:
                    tried_set.add((int(k_lin), drop))

                    # record as labeled
                    lab_u.append(int(u))
                    lab_v.append(int(v))
                    lab_k.append(int(k_lin))

                if drop > thr_k:
                    for (_u, _v, _action, k_lin) in batch:
                        y_out[k_lin] = 1
                        harmful_set.add((int(k_lin), drop))
                    flips_done += len(batch)

        elif mode == "k_action_individual":
            while (flips_done == 0 or flips_done < n_candidates_k_sample) and tries < max_sampling_tries:
                batch = PRBCD._sample_k_flips(
                    n=n,
                    present=present,
                    g=g,
                    device=device,
                    k=int(k_samples_batch),
                    seen=seen,
                )

                if not batch:
                    break

                # mark seen + count tries (same as k_action)
                for (u, v, _action, k_lin) in batch:
                    seen[k_lin] = True
                    tries += 1

                # --- evaluate batch ---
                ei_use, ew_use = _build_perturbed_adj(batch)

                with torch.no_grad():
                    logits_pert = self.attacked_model(data=self.attr.to(device), adj=(ei_use, ew_use))

                drop_batch = _compute_drop(logits_pert)

                # --- if batch is NOT harmful enough, record batch drop (weak labels) ---
                if drop_batch <= thr_k:
                    for (u, v, _action, k_lin) in batch:
                        tried_set.add((int(k_lin), drop_batch))
                        lab_u.append(int(u))
                        lab_v.append(int(v))
                        lab_k.append(int(k_lin))

                    continue

                # --- if batch IS harmful, re-evaluate each edge individually ---
                for (u, v, action, k_lin) in batch:
                    single = [(u, v, action, k_lin)]
                    ei_s, ew_s = _build_perturbed_adj(single)

                    with torch.no_grad():
                        logits_s = self.attacked_model(data=self.attr.to(device), adj=(ei_s, ew_s))

                    drop_ind = _compute_drop(logits_s)

                    tried_set.add((int(k_lin), drop_ind))
                    lab_u.append(int(u))
                    lab_v.append(int(v))
                    lab_k.append(int(k_lin))

                    # label harmful edges using INDIVIDUAL drop
                    if drop_ind > thr_k / k_samples_batch:  # TODO: hier individual drop festlegen
                        y_out[k_lin] = 1
                        harmful_set.add((int(k_lin), drop_ind))
                        flips_done += 1

        elif mode == "k_hop":
            if adj_list is None:
                raise RuntimeError("adj_list must be built for k_hop mode.")

            max_root_tries = 100_000

            while (flips_done == 0 or flips_done < n_candidates_k_hop_sample) and tries < max_sampling_tries:
                batch = PRBCD._sample_khop_flips(
                    n=n,
                    present=present,
                    g=g,
                    device=device,
                    k_hop=k_hop,
                    seen=seen,
                    adj_list=adj_list,
                    max_root_tries=max_root_tries,
                )

                if not batch:
                    break

                for (u, v, _action, k_lin) in batch:
                    seen[k_lin] = True
                    tries += 1

                ei_use, ew_use = _build_perturbed_adj(batch)

                with torch.no_grad():
                    logits_pert = self.attacked_model(data=self.attr.to(device), adj=(ei_use, ew_use))

                drop = _compute_drop(logits_pert)

                for (u, v, _action, k_lin) in batch:
                    tried_set.add((int(k_lin), drop))

                    # record as labeled
                    lab_u.append(int(u))
                    lab_v.append(int(v))
                    lab_k.append(int(k_lin))

                if drop > thr_khop:
                    for (_u, _v, _action, k_lin) in batch:
                        y_out[k_lin] = 1
                        harmful_set.add((int(k_lin), drop))

                    flips_done += len(batch)

            if flips_done > n_candidates_k_hop_sample:
                harmful_idx = torch.nonzero(y_out, as_tuple=False).flatten()
                keep = harmful_idx[:n_candidates_k_hop_sample]
                remove = harmful_idx[n_candidates_k_hop_sample:]

                y_out[remove] = 0
                keep_set = set(int(i.item()) for i in keep)
                harmful_set = {
                    (idx, drop)
                    for (idx, drop) in harmful_set
                    if idx in keep_set
                }
                flips_done = n_candidates_k_hop_sample

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                "Use 'one_sample', 'node_matching', 'k_action', "
                "'k_action_individual', or 'k_hop'."
            )

        # ---- build edge_index_lab and y_label ----
        # create mapping from k_lin -> (u, v) for all actually evaluated pairs
        k_to_uv = {k: (u, v) for u, v, k in zip(lab_u, lab_v, lab_k)}

        if not tried_set or len(k_to_uv) == 0:
            edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
            y_label = torch.empty((0,), device=device, dtype=torch.uint8)
            return y_out, edge_index_lab, y_label, tried_set, harmful_set

        # ---------------- endpoint / endpointPRBCD mode: balanced labels ----------------
        if drop_mode in ("endpoint", "endpointPRBCD"):
            # Extract k_lin indices from sets of (k_lin, drop)
            harmful_k = {int(k) for (k, _d) in harmful_set}
            tried_k = {int(k) for (k, _d) in tried_set}

            # only keep those we can map to (u,v)
            harmful_k = [k for k in harmful_k if k in k_to_uv]

            if len(harmful_k) == 0:
                edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
                y_label = torch.empty((0,), device=device, dtype=torch.uint8)
                return y_out, edge_index_lab, y_label, tried_set, harmful_set

            harmful_k_set = set(harmful_k)

            # candidate negatives = tried but not harmful
            neg_candidates = [
                k for k in tried_k
                if (k not in harmful_k_set)
                   and (k in k_to_uv)
            ]

            # sample same number of negatives as positives (or as many as available)
            n_pos = len(harmful_k)
            n_neg = min(len(neg_candidates), n_pos)

            # reproducible RNG
            import random
            rnd = random.Random(int(rng_seed))
            rnd.shuffle(neg_candidates)
            neg_k = neg_candidates[:n_neg]

            sel_u, sel_v, sel_y = [], [], []

            # positives -> 1
            for k_lin in harmful_k:
                u, v = k_to_uv[k_lin]
                sel_u.append(u)
                sel_v.append(v)
                sel_y.append(1)

            # negatives -> 0
            for k_lin in neg_k:
                u, v = k_to_uv[k_lin]
                sel_u.append(u)
                sel_v.append(v)
                sel_y.append(0)

            if len(sel_u) == 0:
                edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
                y_label = torch.empty((0,), device=device, dtype=torch.uint8)
                return y_out, edge_index_lab, y_label, tried_set, harmful_set

            u_tensor = torch.tensor(sel_u, device=device, dtype=torch.long)
            v_tensor = torch.tensor(sel_v, device=device, dtype=torch.long)
            edge_index_lab = torch.stack([u_tensor, v_tensor], dim=0)

            y_label = torch.tensor(sel_y, device=device, dtype=torch.uint8)

            return y_out, edge_index_lab, y_label, tried_set, harmful_set

        # ---------------- acc/loss mode: keep your old drop-based selection ----------------

        # sort tried_set by drop value (ascending: smallest drop first)
        sorted_tried = sorted(tried_set, key=lambda t: t[1])  # (k_lin, drop)

        M_target = int(n_candidates_k_sample)
        K = min(len(sorted_tried), M_target)

        if K == 0:
            edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
            y_label = torch.empty((0,), device=device, dtype=torch.uint8)
            return y_out, edge_index_lab, y_label, tried_set, harmful_set

        half_high = K  # TODO: your current setting
        half_low = K

        low_part = sorted_tried[:half_low]
        high_part = sorted_tried[-half_high:] if half_high > 0 else []

        sel_u, sel_v, sel_y = [], [], []

        for k_lin, _drop in low_part:
            if k_lin not in k_to_uv:
                continue

            u, v = k_to_uv[k_lin]
            sel_u.append(u)
            sel_v.append(v)
            sel_y.append(0)

        for k_lin, _drop in high_part:
            if k_lin not in k_to_uv:
                continue

            u, v = k_to_uv[k_lin]
            sel_u.append(u)
            sel_v.append(v)
            sel_y.append(1)

        if len(sel_u) == 0:
            edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
            y_label = torch.empty((0,), device=device, dtype=torch.uint8)
            return y_out, edge_index_lab, y_label, tried_set, harmful_set

        u_tensor = torch.tensor(sel_u, device=device, dtype=torch.long)
        v_tensor = torch.tensor(sel_v, device=device, dtype=torch.long)
        edge_index_lab = torch.stack([u_tensor, v_tensor], dim=0)

        y_label = torch.tensor(sel_y, device=device, dtype=torch.uint8)

        return y_out, edge_index_lab, y_label, tried_set, harmful_set

    def label_edge_flips_prbcd_subgraph_endpoint_one_sample(
            self,
            sub_nodes: torch.Tensor,  # 1D long tensor of GLOBAL node ids, shape (m,)
            n_candidates_one_sample: int = 2000,
            rng_seed: int = 0,
            max_sampling_tries: int = 100_000,
            prev_tried_set: "set[tuple[int, float | None]] | None" = None,
    ) -> tuple[
        torch.Tensor,  # y_out over ALL global upper-tri pairs
        torch.Tensor,  # edge_index_lab (2, M) in GLOBAL node ids
        torch.Tensor,  # y_label (M,) uint8
        torch.Tensor,  # edge_index_lab_local (LOCAL)
        "set[tuple[int, float | None]]",  # tried_set with GLOBAL k_lin
        "set[tuple[int, float | None]]",  # harmful_set with GLOBAL k_lin
    ]:
        import torch
        from typing import Set, Tuple

        device = getattr(self, "device", "cpu")

        n_global = int(self.n)
        if n_global <= 1:
            raise ValueError("Graph must have at least 2 nodes.")
        if sub_nodes.numel() < 2:
            raise ValueError("sub_nodes must contain at least 2 nodes.")

        sub_nodes = sub_nodes.to(device=device, dtype=torch.long).contiguous()
        m = int(sub_nodes.numel())

        # ---------------- helpers: (u,v) <-> lin index in upper triangle ----------------
        # k_lin for 0 <= u < v < n, mapping used by PRBCD._build_uppertri_bitset
        # k(u,v) = u*(n-1) - u*(u+1)//2 + (v-u-1)
        def _pair_to_lin(u: int, v: int, n: int) -> int:
            if u > v:
                u, v = v, u
            # assumes u < v
            return u * (n - 1) - (u * (u + 1)) // 2 + (v - u - 1)

        # ---------------- base adjacency (global) ----------------
        ei_base_g = self.edge_index.to(device=device, dtype=torch.long).contiguous()  # (2,E)
        ew_base_g = (
            self.edge_weight.to(device).float().contiguous()
            if getattr(self, "edge_weight", None) is not None
            else torch.ones(ei_base_g.size(1), device=device, dtype=torch.float32)
        )

        # ---------------- build induced subgraph adjacency ----------------
        # mask edges where both endpoints are in sub_nodes
        # build global->local mapping
        g2l = torch.full((n_global,), -1, device=device, dtype=torch.long)
        g2l[sub_nodes] = torch.arange(m, device=device, dtype=torch.long)

        src_g = ei_base_g[0]
        dst_g = ei_base_g[1]
        src_l = g2l[src_g]
        dst_l = g2l[dst_g]
        in_sub = (src_l >= 0) & (dst_l >= 0)

        ei_sub = torch.stack([src_l[in_sub], dst_l[in_sub]], dim=0).contiguous()  # (2,E_sub) LOCAL ids
        ew_sub = ew_base_g[in_sub].contiguous()

        # local dir_pos for deletes (local adjacency)
        E_sub = int(ei_sub.size(1))
        dir_pos = torch.full((m, m), -1, dtype=torch.int32, device=device)
        if E_sub > 0:
            dir_pos[ei_sub[0], ei_sub[1]] = torch.arange(E_sub, device=device, dtype=torch.int32)

        # membership bitset for local undirected presence
        present = PRBCD._build_uppertri_bitset(ei_sub, m)  # (m*(m-1)//2,) bool

        # ---------------- clean forward on SUBGRAPH ONLY ----------------
        # IMPORTANT: This assumes your attacked_model can run on (attr_sub, adj_sub)
        # If your model needs full-feature tensor with local indexing, we slice attr accordingly.
        attr_sub = self.attr.to(device)[sub_nodes]
        labels_sub = self.labels.to(device)[sub_nodes]

        logits_clean = self.attacked_model(data=attr_sub, adj=(ei_sub, ew_sub))
        pred_clean = logits_clean.argmax(dim=-1)
        clean_correct = (pred_clean == labels_sub)

        # ---------------- outputs ----------------
        num_pairs_global = n_global * (n_global - 1) // 2
        y_out = torch.zeros(num_pairs_global, dtype=torch.uint8, device=device)

        lab_u: list[int] = []
        lab_v: list[int] = []
        lab_k: list[int] = []  # GLOBAL k_lin

        tried_set: Set[Tuple[int, float | None]] = set(prev_tried_set) if prev_tried_set is not None else set()
        harmful_set: Set[Tuple[int, float | None]] = set()

        # seen should prevent duplicates globally (optional but matches your previous semantics)
        seen_global = torch.zeros(num_pairs_global, dtype=torch.bool, device=device)
        if prev_tried_set:
            prev_indices = [idx for (idx, _d) in prev_tried_set if 0 <= idx < num_pairs_global]
            if prev_indices:
                seen_global[torch.tensor(prev_indices, device=device, dtype=torch.long)] = True

        # torch RNG
        g = torch.Generator(device=device)
        g.manual_seed(int(rng_seed))

        flips_done, tries = 0, 0

        def _endpoint_flipped_correct_to_incorrect(u_l: int, v_l: int, logits_pert: torch.Tensor) -> bool:
            pred_pert = logits_pert.argmax(dim=-1)
            u_l = int(u_l)
            v_l = int(v_l)

            u_flip = bool(clean_correct[u_l] and (pred_pert[u_l] != labels_sub[u_l]))
            v_flip = bool(clean_correct[v_l] and (pred_pert[v_l] != labels_sub[v_l]))
            return u_flip or v_flip

        def _build_perturbed_adj_one(u_l: int, v_l: int, action: str):
            """
            Build perturbed (ei_use, ew_use) on the SUBGRAPH ONLY.
            - del: set weights of directed edges (u,v) and (v,u) to 0 if present
            - add: append directed edges (u,v) and (v,u) (weight 1)
            """
            ew_use = ew_sub.clone()
            add_edges = []
            del_indices = []

            if action == "del":
                idx = int(dir_pos[u_l, v_l].item())
                if idx >= 0:
                    del_indices.append(idx)
                idx2 = int(dir_pos[v_l, u_l].item())
                if idx2 >= 0:
                    del_indices.append(idx2)
            else:  # "add"
                add_edges.append((u_l, v_l))
                add_edges.append((v_l, u_l))

            if del_indices:
                del_idx_tensor = torch.tensor(del_indices, device=device, dtype=torch.long)
                ew_use[del_idx_tensor] = 0.0

            if add_edges:
                extra_ei = torch.tensor(add_edges, device=device, dtype=torch.long).t()
                extra_ew = torch.ones(extra_ei.size(1), device=device, dtype=torch.float32)
                ei_use = torch.cat([ei_sub, extra_ei], dim=1)
                ew_use = torch.cat([ew_use, extra_ew], dim=0)
            else:
                ei_use = ei_sub

            return ei_use, ew_use

        # ---------------- main loop: one_sample on SUBGRAPH, endpoint labels on SUBGRAPH ----------------
        while (flips_done == 0 or flips_done < n_candidates_one_sample) and tries < max_sampling_tries:
            # sample one flip in LOCAL subgraph index space
            u_l, v_l, action, k_lin_local = PRBCD._sample_one_flip(m, present, g, device)
            tries += 1

            # map LOCAL endpoints -> GLOBAL endpoints
            u_g = int(sub_nodes[int(u_l)].item())
            v_g = int(sub_nodes[int(v_l)].item())

            # global pair index
            k_lin_global = _pair_to_lin(u_g, v_g, n_global)

            if seen_global[k_lin_global]:
                continue
            seen_global[k_lin_global] = True

            # build perturbed subgraph adjacency and evaluate on SUBGRAPH ONLY
            ei_use, ew_use = _build_perturbed_adj_one(int(u_l), int(v_l), action)

            with torch.no_grad():
                logits_pert = self.attacked_model(data=attr_sub, adj=(ei_use, ew_use))

            drop = None  # endpoint-mode doesn't need a scalar drop; keep None for compatibility
            tried_set.add((int(k_lin_global), drop))

            lab_u.append(u_g)
            lab_v.append(v_g)
            lab_k.append(int(k_lin_global))

            if _endpoint_flipped_correct_to_incorrect(int(u_l), int(v_l), logits_pert):
                y_out[k_lin_global] = 1
                harmful_set.add((int(k_lin_global), drop))
                flips_done += 1

        # ---------------- build edge_index_lab & y_label from tried/harmful (balanced like your endpoint block) ----------------
        if len(lab_k) == 0:
            edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
            y_label = torch.empty((0,), device=device, dtype=torch.uint8)
            return y_out, edge_index_lab, y_label, tried_set, harmful_set

        k_to_uv = {k: (u, v) for (u, v, k) in zip(lab_u, lab_v, lab_k)}

        harmful_k = [int(k) for (k, _d) in harmful_set if int(k) in k_to_uv]
        if len(harmful_k) == 0:
            edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
            y_label = torch.empty((0,), device=device, dtype=torch.uint8)
            return y_out, edge_index_lab, y_label, tried_set, harmful_set

        tried_k = [int(k) for (k, _d) in tried_set if int(k) in k_to_uv]
        harmful_k_set = set(harmful_k)
        neg_candidates = [k for k in tried_k if k not in harmful_k_set]

        import random
        rnd = random.Random(int(rng_seed))
        rnd.shuffle(neg_candidates)
        n_pos = len(harmful_k)
        n_neg = min(len(neg_candidates), n_pos)
        neg_k = neg_candidates[:n_neg]

        sel_u, sel_v, sel_y = [], [], []

        for k in harmful_k:
            u, v = k_to_uv[k]
            sel_u.append(u);
            sel_v.append(v);
            sel_y.append(1)

        for k in neg_k:
            u, v = k_to_uv[k]
            sel_u.append(u);
            sel_v.append(v);
            sel_y.append(0)

        if len(sel_u) == 0:
            edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
            y_label = torch.empty((0,), device=device, dtype=torch.uint8)
            edge_index_lab_local = torch.empty((2, 0), device=device, dtype=torch.long)
            return y_out, edge_index_lab, y_label, edge_index_lab_local, tried_set, harmful_set

        edge_index_lab = torch.stack(
            [torch.tensor(sel_u, device=device, dtype=torch.long),
             torch.tensor(sel_v, device=device, dtype=torch.long)],
            dim=0
        )

        # ---- build LOCAL versions of labeled edges ----
        u_global = edge_index_lab[0]
        v_global = edge_index_lab[1]

        u_local = g2l[u_global]
        v_local = g2l[v_global]

        edge_index_lab_local = torch.stack([u_local, v_local], dim=0)

        y_label = torch.tensor(sel_y, device=device, dtype=torch.uint8)
        return y_out, edge_index_lab, y_label, edge_index_lab_local, tried_set, harmful_set

    def _run_prbcd_for_endpoint_candidate_block(
            self,
            n_perturbations,
            n_candidates: int = 2_000_000,
            rng_seed: int = 0,
            epochs: int = 5,
            **kwargs,
    ) -> torch.Tensor:
        """
        Placeholder for endpointPRBCD mode.

        This function should run a PRBCD attack or PRBCD-style candidate selection
        and return a 1D tensor of upper-triangular linear edge indices.

        Expected return:
            block_lin: torch.Tensor of shape (B,), dtype=torch.long
                       containing candidate edge indices in PRBCD upper-tri format.

        Important:
            The returned indices must refer to the same linear indexing scheme used by
            PRBCD.triu_idx_to_linear_idx(n, ...).
        """

        # TODO:
        # 1. Run PRBCD attack / construct PRBCD search block.
        # 2. Extract the resulting block, e.g. self.current_search_space.
        # 3. Return it as a 1D long tensor on self.device.

        self.sample_random_block(n_perturbations, n_candidates)
        # Accuracy and attack statistics before the attack even started
        with torch.no_grad():

            logits = self._get_logits(self.attr, self.edge_index, self.edge_weight)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

            logging.info(f'\nBefore the attack - Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

            self._append_attack_statistics(loss.item(), accuracy, 0., 0.)

            del logits, loss

        # Loop over the epochs (Algorithm 1, line 5)
        for epoch in tqdm(range(epochs)):
            self.perturbed_edge_weight.requires_grad = True

            # Retreive sparse perturbed adjacency matrix `A \oplus p_{t-1}` (Algorithm 1, line 6)
            edge_index, edge_weight = self.get_modified_adj()

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Calculate logits for each node (Algorithm 1, line 6)
            logits = self._get_logits(self.attr, edge_index, edge_weight)
            # Calculate loss combining all each node (Algorithm 1, line 7)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[
                self.idx_attack])  # Todo: Hier wird der loss und gradient für perturbed edge weight erzeugt.
            # Retreive gradient towards the current block (Algorithm 1, line 7)
            gradient = utils.grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
            self.gradient = gradient

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                # Gradient update step (Algorithm 1, line 7)
                edge_weight = self.update_edge_weights(n_candidates, epoch, gradient)[1]
                # For monitoring
                probability_mass_update = self.perturbed_edge_weight.sum().item()
                # Projection to stay within relaxed `L_0` budget (Algorithm 1, line 8)
                self.perturbed_edge_weight = Attack.project(
                    n_candidates, self.perturbed_edge_weight, self.eps)
                # For monitoring
                probability_mass_projected = self.perturbed_edge_weight.sum().item()

                # Calculate accuracy after the current epoch (overhead for monitoring and early stopping)
                edge_index, edge_weight = self.get_modified_adj()
                logits = self.attacked_model(data=self.attr.to(self.device), adj=(edge_index, edge_weight))
                accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

                del edge_index, edge_weight, logits

                if epoch % self.display_step == 0:
                    logging.info(f'\nEpoch: {epoch} Loss: {loss} Accuracy: {100 * accuracy:.3f} %\n')

                # Resampling of search space (Algorithm 1, line 9-14)
                self.resample_random_block(n_perturbations, n_candidates)

        # Sample final discrete graph (Algorithm 1, line 16)
        edge_index = self.sample_final_edges(50000)[0]

        edge_index_flat = PRBCD.triu_idx_to_linear_idx(int(self.n),edge_index)

        return edge_index_flat

    def label_edge_flips_prbcd_k_action(
            self,
            n_candidates_k_sample: int = 2000,
            drop_threshold_k_samples: float = -1,
            rng_seed: int = 0,
            max_sampling_tries: int = 100_000,
            k_samples_batch: int = 20,
            prev_tried_set: "Set[tuple[int, float]] | None" = None,
    ):
        """
        Reine k_action-Version von label_edge_flips_prbcd_selfsample_fast.
        Führt Sampling von k-Flips gleichzeitig durch und evaluiert sie in einem Schritt.
        """

        import torch
        from typing import Set, Tuple

        device = getattr(self, "device", "cpu")

        n: int = int(self.n)
        if n <= 1:
            raise ValueError("Graph must have at least 2 nodes.")

        # ---- base adjacency
        ei_base = self.edge_index.to(device=device, dtype=torch.long).contiguous()
        ew_base = (self.edge_weight.to(device).float()
                   if getattr(self, "edge_weight", None) is not None
                   else torch.ones(ei_base.size(1), device=device, dtype=torch.float32)).contiguous()

        # ---- undirected membership bitset
        present = PRBCD._build_uppertri_bitset(ei_base, n)

        E = ei_base.size(1)
        dir_pos = torch.full((n, n), -1, dtype=torch.int32, device=device)
        dir_pos[ei_base[0], ei_base[1]] = torch.arange(E, device=device, dtype=torch.int32)

        # ---- clean accuracy
        logits_clean = self.attacked_model(data=self.attr.to(device), adj=(ei_base, ew_base))
        acc_clean = utils.accuracy(logits_clean, self.labels.to(device), self.idx_attack)

        # ---- outputs
        num_pairs = n * (n - 1) // 2
        y_out = torch.zeros(num_pairs, dtype=torch.uint8, device=device)

        tried_set: Set[Tuple[int, float]] = set(prev_tried_set) if prev_tried_set else set()
        harmful_set: Set[Tuple[int, float]] = set()

        # ---- seen mask
        seen = torch.zeros(num_pairs, dtype=torch.bool, device=device)

        if prev_tried_set:
            prev_indices = [idx for (idx, _) in prev_tried_set if 0 <= idx < num_pairs]
            if prev_indices:
                seen[torch.tensor(prev_indices, device=device, dtype=torch.long)] = True

        # RNG
        g = torch.Generator(device=device)
        g.manual_seed(int(rng_seed))

        flips_done = 0
        tries = 0

        # --- helper
        def _build_perturbed_adj(flips):
            ew_use = ew_base.clone()
            del_indices = []
            add_edges = []

            for (u, v, action, _) in flips:
                if action == "del":
                    idx1 = int(dir_pos[u, v])
                    if idx1 >= 0:
                        del_indices.append(idx1)
                    idx2 = int(dir_pos[v, u])
                    if idx2 >= 0:
                        del_indices.append(idx2)
                else:  # add
                    add_edges.append((u, v))
                    add_edges.append((v, u))

            if del_indices:
                ew_use[torch.tensor(del_indices, device=device)] = 0.0

            if add_edges:
                extra_ei = torch.tensor(add_edges, device=device).t()
                extra_ew = torch.ones(extra_ei.size(1), device=device)
                ei_use = torch.cat([ei_base, extra_ei], dim=1)
                ew_use = torch.cat([ew_use, extra_ew], dim=0)
            else:
                ei_use = ei_base

            return ei_use, ew_use

        # ---- k_action loop ----
        while (flips_done == 0 or flips_done < n_candidates_k_sample) and tries < max_sampling_tries:

            batch = PRBCD._sample_k_flips(
                n=n,
                present=present,
                g=g,
                device=device,
                k=int(k_samples_batch),
                seen=seen,
            )
            if not batch:
                break

            for (_, _, _, k_lin) in batch:
                seen[k_lin] = True
                tries += 1

            ei_use, ew_use = _build_perturbed_adj(batch)

            with torch.no_grad():
                logits = self.attacked_model(
                    data=self.attr.to(device),
                    adj=(ei_use, ew_use)
                )
                acc_pert = utils.accuracy(logits, self.labels.to(device), self.idx_attack)

            drop = float(acc_clean - acc_pert)

            # record in tried_set
            for (_, _, _, k_lin) in batch:
                tried_set.add((int(k_lin), drop))

            if drop > drop_threshold_k_samples:
                for (_, _, _, k_lin) in batch:
                    y_out[k_lin] = 1
                    harmful_set.add((int(k_lin), drop))
                flips_done += len(batch)

        return y_out, tried_set, harmful_set

    # ---------- fast membership over upper triangle ----------
    def _build_uppertri_bitset(edge_index: torch.Tensor, n: int) -> torch.Tensor:
        """
        Returns a boolean vector 'present' of length num_pairs = n*(n-1)//2.
        present[k] = True iff the undirected pair for linear index k exists (u<v).
        """
        num_pairs = n * (n - 1) // 2
        if num_pairs == 0:
            return torch.zeros(0, dtype=torch.bool, device=edge_index.device)
        lin = PRBCD.pairs_to_linear_uppertri(edge_index, n)  # (E',)
        if lin.numel() > 0:
            lin = torch.unique(lin, sorted=False)
        present = torch.zeros(num_pairs, dtype=torch.bool, device=edge_index.device)
        if lin.numel() > 0:
            present[lin] = True
        return present

    # ---------- tried_set to seen converter (for resampling) ----------
    @staticmethod
    def apply_tried_set_to_seen(seen: torch.Tensor, tried_set: set):
        """
        Mark all entries in `seen` that appear in `tried_set`.

        Parameters
        ----------
        seen : torch.Tensor
            A boolean tensor of shape (num_pairs,) tracking which linearized pairs
            have already been sampled.
        tried_set : set[tuple[int, float]]
            Python set of (k_lin, drop) entries for every evaluated candidate.

        Returns
        -------
        torch.Tensor
            The updated boolean `seen` tensor.
        """
        if not tried_set:
            return seen

        # Extract all linear indices that were tried
        tried_indices = [idx for (idx, _drop) in tried_set]

        # Convert to tensor on same device as `seen`
        tried_tensor = torch.tensor(
            tried_indices, device=seen.device, dtype=torch.long
        )

        # Mark them as seen
        seen[tried_tensor] = True
        return seen

    # ---------- single-sample sampler (no batching, torch-only) ----------
    def _sample_one_flip(n: int, present: torch.Tensor, g: torch.Generator, device: torch.device) -> Tuple[
        int, int, str, int]:
        """
        Sample exactly one unordered pair (u<v) uniformly from all C(n,2) pairs.
        Decide 'del' if present[k] else 'add'. Returns (u, v, action, k).
        """
        num_pairs = n * (n - 1) // 2
        k = int(torch.randint(num_pairs, (1,), generator=g, device=device).item())
        uv = PRBCD.linear_to_triu_idx(n, torch.tensor([k], device=device))  # (2,1)
        u = int(uv[0, 0].item())
        v = int(uv[1, 0].item())
        action = "del" if present[k].item() else "add"
        return u, v, action, k

    @staticmethod
    def _sample_k_flips(n: int,
                        present: torch.Tensor,
                        g: torch.Generator,
                        device: torch.device,
                        *,
                        k: int,
                        seen: torch.Tensor) -> list[tuple[int, int, str, int]]:
        """
        Sample up to k DISTINCT unseen undirected pairs (by linear index over the upper triangle).
        For each pair, action = 'del' if present, else 'add'.
        Returns list of (u, v, action, k_lin).
        """
        num_pairs = int(n * (n - 1) // 2)
        if num_pairs <= 0:
            return []

        # candidates mask: not seen
        mask = (~seen).nonzero(as_tuple=False).flatten()
        if mask.numel() == 0:
            return []

        # pick up to k random indices from remaining
        if mask.numel() <= k:
            pick = mask
        else:
            perm = torch.randperm(mask.numel(), device=device, generator=g)
            pick = mask[perm[:k]]

        # map linear -> (u,v) with u<v
        uv = PRBCD.linear_to_triu_idx(n, pick.long())  # shape (2, m)
        u = uv[0].to(torch.int64)
        v = uv[1].to(torch.int64)

        # decide actions from 'present' bitset
        chosen_present = present[pick]  # bool
        actions = ["del" if bool(x) else "add" for x in chosen_present.tolist()]

        # build output list
        out: list[tuple[int, int, str, int]] = []
        for i in range(pick.numel()):
            out.append((int(u[i].item()), int(v[i].item()), actions[i], int(pick[i].item())))
        return out

    @staticmethod
    def _sample_khop_flips(
            n: int,
            present: torch.Tensor,
            g: torch.Generator,
            device: torch.device,
            *,
            k_hop: int,
            seen: torch.Tensor,
            adj_list: list[list[int]],
            max_root_tries: int = 10,
    ) -> list[tuple[int, int, str, int]]:
        """
        Sample ALL DISTINCT unseen undirected pairs inside the k-hop neighborhood of
        a randomly chosen center node.

        Returns list[(u, v, action, k_lin)].

        - Center node is chosen uniformly at random from {0, ..., n-1}, up to
          `max_root_tries` attempts to find one whose neighborhood yields at least
          one unseen pair.
        - k_hop >= 1.
        """
        import torch
        from collections import deque

        num_pairs_total = int(n * (n - 1) // 2)
        if num_pairs_total <= 0 or k_hop <= 0:
            return []

        for _ in range(max_root_tries):
            # 1) pick random center node
            center = int(torch.randint(n, (1,), generator=g, device=device).item())

            # 2) BFS up to depth k_hop to get node set N_k(center)
            visited = set([center])
            q = deque([(center, 0)])
            while q:
                node, dist = q.popleft()
                if dist >= k_hop:
                    continue
                for nb in adj_list[node]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append((nb, dist + 1))

            if len(visited) <= 1:
                # no pairs here, continue
                continue

            nodes = sorted(visited)

            # 3) enumerate all unordered node pairs in this neighborhood (u < v)
            us = []
            vs = []
            for i in range(len(nodes)):
                u = nodes[i]
                for j in range(i + 1, len(nodes)):
                    v = nodes[j]
                    us.append(u)
                    vs.append(v)

            if not us:
                continue

            # 4) map (u, v) -> linear indices over upper triangle
            pair_ei = torch.stack(
                [
                    torch.tensor(us, dtype=torch.long, device=device),
                    torch.tensor(vs, dtype=torch.long, device=device),
                ],
                dim=0,
            )  # shape (2, m)

            k_lin_all = PRBCD.pairs_to_linear_uppertri(pair_ei, n).long()  # (m,)

            # 5) keep only unseen pairs
            unseen_mask = (~seen[k_lin_all]).nonzero(as_tuple=False).flatten()
            if unseen_mask.numel() == 0:
                continue

            k_lin = k_lin_all[unseen_mask]
            u_sel = pair_ei[0, unseen_mask]
            v_sel = pair_ei[1, unseen_mask]

            # 6) decide actions from 'present' bitset
            chosen_present = present[k_lin]  # bool
            actions = ["del" if bool(x) else "add" for x in chosen_present.tolist()]

            # 7) build output list
            out: list[tuple[int, int, str, int]] = []
            for i in range(k_lin.numel()):
                out.append(
                    (
                        int(u_sel[i].item()),
                        int(v_sel[i].item()),
                        actions[i],
                        int(k_lin[i].item()),
                    )
                )

            return out

        # if we get here, no suitable center found within max_root_tries
        return []

    def acc_sampler_sample_one_edge(
            self,
            p_add: float = 0.1,
            p_del: float = 0.06,
            rng_seed: Optional[int] = None,
            *,
            existing_set: Set[Tuple[int, int]],
            N: int,
    ) -> List[Tuple[int, int, str]]:
        """
        Keep sampling one random unordered node pair (u,v) until a flip is accepted.
        Uses the precomputed undirected membership 'existing_set' and node count N.
        Returns exactly one candidate: [(u,v,'add'|'del')].
        """
        import numpy as np

        if N is None or N <= 1:
            raise ValueError("acc_sampler_sample_one_edge: graph must have at least 2 nodes.")

        # feasibility once (no rebuilding)
        num_pairs = N * (N - 1) // 2
        num_edges = len(existing_set)
        num_non_edges = num_pairs - num_edges
        if num_edges == 0 and p_add <= 0.0:
            raise ValueError("No existing edges and p_add=0 — cannot sample a flip.")
        if num_non_edges == 0 and p_del <= 0.0:
            raise ValueError("Graph is complete and p_del=0 — cannot sample a flip.")

        rng = np.random.default_rng(rng_seed)

        while True:
            u = int(rng.integers(0, N))
            v = int(rng.integers(0, N - 1))
            if v >= u:
                v += 1
            if u > v:
                u, v = v, u

            is_edge = (u, v) in existing_set
            prob = p_del if is_edge else p_add
            if rng.random() < prob:
                action = "del" if is_edge else "add"
                return [(u, v, action)]

    def acc_sampler_flip_one_edge_simple(
            self,
            *,
            existing_set: Set[Tuple[int, int]],
            N: int,
            rng_seed: Optional[int] = None,
    ) -> Tuple[int, int, str]:
        """
        Pick exactly one random unordered node pair (u,v) uniformly (u < v) and flip its state.
        Returns (u, v, 'add' | 'del').
        """
        import numpy as np
        import math

        if N is None or N <= 1:
            raise ValueError("flip_one_edge: graph must have at least 2 nodes.")

        rng = np.random.default_rng(rng_seed)

        # Sample a single index in [0, C(N,2))
        num_pairs = N * (N - 1) // 2
        t = int(rng.integers(0, num_pairs))

        disc = (2 * N - 1) ** 2 - 8 * t
        u = int((2 * N - 1 - math.isqrt(disc)) // 2)
        off_u = u * (2 * N - u - 1) // 2
        v = u + 1 + (t - off_u)

        action = "del" if (u, v) in existing_set else "add"
        return (u, v, action)

    def _make_margin_labels(self):
        with torch.no_grad():
            x = self.attr.to(self.device)
            ei = self.edge_index.to(self.device)

            ew = self.edge_weight
            if ew is None:
                ew = torch.ones(ei.size(1), device=self.device)
            else:
                ew = ew.to(self.device)

            logits = self.attacked_model(data=x, adj=(ei, ew))

            top2_vals = torch.topk(logits, k=2, dim=1).values
            margins = (top2_vals[:, 0] - top2_vals[:, 1]).clamp_min(0.0)

            mean, std = margins.mean(), margins.std().clamp_min(1e-6)
            return (margins - mean) / std
    '''
    def train_link_prediction_gnn(
            self,
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
            # ---- early stopping / best checkpoint ----
            early_stop: bool = True,
            early_stop_metric: str = "auc",  # "auc" | "ap" | "val_loss" | "val_acc"
            early_stop_patience: int = 15,
            early_stop_min_delta: float = 1e-4,
            restore_best: bool = True,
    ):
        x = x.to(device)
        edge_index_struct = edge_index_struct.to(device)
        edge_index_lab = edge_index_lab.to(device)
        y_label = y_label.float().to(device)

        M = edge_index_lab.size(1)
        if M == 0:
            print("[LP-GNN] No labeled pairs. Returning untrained model.")
            model = LinkPredictionGNN(
                in_dim=x.size(1),
                hidden_dim=hidden_dim,
                out_dim=out_dim,
            ).to(device)
            return model

        assert M == y_label.numel(), "edge_index_lab and y_label must have the same number of examples"

        y_int = y_label.long()
        if verbose:
            print(
                f"[LP-GNN] Labeled pairs: M={M} | "
                f"pos={(y_int == 1).sum().item()} | neg={(y_int == 0).sum().item()}"
            )

        # ---- split ----
        assert M % 2 == 0, "Expected equal number of negatives/positives (M must be even)."
        half = M // 2

        neg_idx_all = torch.arange(0, half, device=device)
        pos_idx_all = torch.arange(half, M, device=device)

        neg_perm = neg_idx_all[torch.randperm(half, device=device)]
        pos_perm = pos_idx_all[torch.randperm(half, device=device)]

        train_size_per_class = int(0.8 * half)

        train_idx = torch.cat([neg_perm[:train_size_per_class], pos_perm[:train_size_per_class]])
        val_idx = torch.cat([neg_perm[train_size_per_class:], pos_perm[train_size_per_class:]])

        train_idx = train_idx[torch.randperm(train_idx.numel(), device=device)]
        val_idx = val_idx[torch.randperm(val_idx.numel(), device=device)]

        # ---- model ---

        model = LinkPredictionGNN(
            in_dim=x.size(1),
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        pos_weight = (len(neg_idx_all) / len(pos_idx_all)) if len(pos_idx_all) > 0 else 1.0
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

        # ---- CSV logger ----
        csv_fields = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "acc_pos",
            "acc_neg",
            "auc",
            "ap",
            "tp",
            "fp",
            "tn",
            "fn",
            "p_min",
            "p_mean",
            "p_max",
            "grad_norm",
            "best_metric",
            "is_best",
        ]

        if csv_path is None:
            csv_path = PRBCD.make_selector_gnn_log_path(
                ads_mode=self.ads_mode,
                drop_mode=self.drop_mode,
                k_samples_batch=self.k_samples_batch,
                n_candidates_k_sample=self.n_candidates_k_sample,
                acc_drop_threshold_k_samples=self.acc_drop_threshold_k_samples,
                loss_drop_threshold_k_samples=self.loss_drop_threshold_k_samples,
                dataset=self.dataset,
            )

        logger = CSVMetricLogger(
            path=csv_path,
            fieldnames=csv_fields,
            append=csv_append,
        )

        # ---- helpers for early stopping ----
        metric_mode = {
            "auc": "max",
            "ap": "max",
            "val_acc": "max",
            "val_loss": "min",
        }
        if early_stop_metric not in metric_mode:
            raise ValueError(f"early_stop_metric must be one of {list(metric_mode.keys())}")

        want = metric_mode[early_stop_metric]
        best_metric = -float("inf") if want == "max" else float("inf")
        best_epoch = -1
        best_state = None

        def _is_improvement(curr: float, best: float) -> bool:
            if want == "max":
                return curr > best + early_stop_min_delta
            else:
                return curr < best - early_stop_min_delta

        patience_left = int(early_stop_patience)

        epoch_iter = tqdm(range(num_epochs), desc="[LP-GNN] Training") if use_tqdm else range(num_epochs)

        for epoch in epoch_iter:
            # ---- train ----
            model.train()
            optimizer.zero_grad()

            logits_train = model(x, edge_index_struct, edge_index_lab[:, train_idx]).view(-1)

            if torch.isnan(logits_train).any() or torch.isinf(logits_train).any():
                print(f"[LP-GNN][ERROR] NaN/Inf in train logits at epoch {epoch + 1}")
                break

            loss = loss_fn(logits_train, y_label[train_idx])
            loss.backward()

            grad_norm_val = None
            if log_grad_norm:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm_val = total_norm ** 0.5

            optimizer.step()

            # ---- validation ----
            model.eval()
            with torch.no_grad():
                logits_val = model(x, edge_index_struct, edge_index_lab[:, val_idx]).view(-1)

                if torch.isnan(logits_val).any() or torch.isinf(logits_val).any():
                    print(f"[LP-GNN][ERROR] NaN/Inf in val logits at epoch {epoch + 1}")
                    break

                val_loss = loss_fn(logits_val, y_label[val_idx])

                probs_val = torch.sigmoid(logits_val)
                preds_val = (probs_val >= 0.5).long()
                yv = y_int[val_idx]

                acc_val = float((preds_val == yv).float().mean().item())
                acc_pos = float((preds_val[yv == 1] == 1).float().mean().item()) if (yv == 1).any() else float("nan")
                acc_neg = float((preds_val[yv == 0] == 0).float().mean().item()) if (yv == 0).any() else float("nan")

                tp, fp, tn, fn = PRBCD._confusion_counts(preds_val, yv)
                auc, ap = PRBCD._safe_auc_ap_sklearn(probs_val, yv)

                pmin = float(probs_val.min().item())
                pmean = float(probs_val.mean().item())
                pmax = float(probs_val.max().item())

            metric_val = None
            if early_stop_metric == "auc":
                metric_val = auc
            elif early_stop_metric == "ap":
                metric_val = ap
            elif early_stop_metric == "val_acc":
                metric_val = acc_val
            elif early_stop_metric == "val_loss":
                metric_val = float(val_loss.item())

            is_best = False
            if metric_val is not None:
                if _is_improvement(float(metric_val), float(best_metric)):
                    best_metric = float(metric_val)
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    patience_left = int(early_stop_patience)
                    is_best = True
                else:
                    patience_left -= 1

            # ---- CSV log ----
            logger.log({
                "epoch": epoch + 1,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss.item()),
                "val_acc": acc_val,
                "acc_pos": acc_pos,
                "acc_neg": acc_neg,
                "auc": auc,
                "ap": ap,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "p_min": pmin,
                "p_mean": pmean,
                "p_max": pmax,
                "grad_norm": grad_norm_val,
                "best_metric": float(best_metric) if best_epoch != -1 else float("nan"),
                "is_best": 1 if is_best else 0,
            })

            # ---- tqdm ----
            if use_tqdm:
                postfix = {
                    "tr_loss": f"{loss.item():.4f}",
                    "va_loss": f"{val_loss.item():.4f}",
                    "va_acc": f"{acc_val:.3f}",
                }
                if auc is not None:
                    postfix["auc"] = f"{auc:.3f}"
                if ap is not None:
                    postfix["ap"] = f"{ap:.3f}"
                postfix["pat"] = patience_left if metric_val is not None else "n/a"
                epoch_iter.set_postfix(postfix)

            # ---- print ----
            if verbose and ((epoch + 1) % log_every == 0 or epoch == 0 or (epoch + 1) == num_epochs):
                msg = (
                    f"[LP-GNN] Epoch {epoch + 1:03d}/{num_epochs} | "
                    f"train_loss={loss.item():.4f} | val_loss={val_loss.item():.4f} | "
                    f"val_acc={acc_val:.4f} | acc_pos={acc_pos:.4f} | acc_neg={acc_neg:.4f} | "
                    f"TP/FP/TN/FN={tp}/{fp}/{tn}/{fn} | "
                    f"p(min/mean/max)={pmin:.3f}/{pmean:.3f}/{pmax:.3f}"
                )
                if auc is not None:
                    msg += f" | AUC={auc:.4f}"
                if ap is not None:
                    msg += f" | AP={ap:.4f}"
                if metric_val is not None:
                    msg += f" | best_{early_stop_metric}={best_metric:.4f} (epoch {best_epoch})"
                if grad_norm_val is not None:
                    msg += f" | grad_norm={grad_norm_val:.3e}"
                print(msg)

            # ---- early stop ----
            if early_stop and metric_val is not None and patience_left <= 0:
                if verbose:
                    print(
                        f"[LP-GNN] Early stopping at epoch {epoch + 1}. "
                        f"Best {early_stop_metric}={best_metric:.4f} at epoch {best_epoch}."
                    )
                break

        logger.close()

        # ---- restore best ----
        if restore_best and best_state is not None:
            model.load_state_dict(best_state)
            if verbose:
                print(f"[LP-GNN] Restored best model from epoch {best_epoch} ({early_stop_metric}={best_metric:.4f}).")

        if verbose:
            print(f"[LP-GNN] Training complete. Metrics saved to '{csv_path}'")

        return model
    '''

    def train_link_prediction_gnn(
            self,
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
            early_stop_metric: str = "val_f1",
            # allowed:
            # "val_auc" | "val_ap" | "val_loss" | "val_acc" | "val_f1" | "val_recall"
            early_stop_patience: int = 10,
            early_stop_min_delta: float = 1e-4,
            restore_best: bool = True,
            # ---- split ratios ----
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            test_ratio: float = 0.15,
            threshold: float = 0.5,
    ):
        x = x.to(device)
        edge_index_struct = edge_index_struct.to(device)
        edge_index_lab = edge_index_lab.to(device)
        y_label = y_label.float().to(device)

        if y_src_label is not None:
            y_src_label = y_src_label.float().to(device)
        if y_dst_label is not None:
            y_dst_label = y_dst_label.float().to(device)

        M = edge_index_lab.size(1)

        if M == 0:
            print("[LP-GNN] No labeled pairs. Returning untrained model.")
            model = LinkPredictionGNN(
                in_dim=x.size(1),
                hidden_dim=hidden_dim,
                out_dim=out_dim,
            ).to(device)
            return model

        assert M == y_label.numel(), "edge_index_lab and y_label must have the same number of examples"

        if y_src_label is not None:
            assert M == y_src_label.numel(), "edge_index_lab and y_src_label must have the same number of examples"

        if y_dst_label is not None:
            assert M == y_dst_label.numel(), "edge_index_lab and y_dst_label must have the same number of examples"

        ratio_sum = train_ratio + val_ratio + test_ratio
        assert abs(ratio_sum - 1.0) < 1e-6, "train_ratio + val_ratio + test_ratio must equal 1.0"

        use_aux = (y_src_label is not None) and (y_dst_label is not None)

        y_int = y_label.long()

        if verbose:
            msg = (
                f"[LP-GNN] Labeled pairs: M={M} | "
                f"pos={(y_int == 1).sum().item()} | neg={(y_int == 0).sum().item()}"
            )

            if use_aux:
                msg += (
                    f" | src_pos={(y_src_label.long() == 1).sum().item()} "
                    f"| dst_pos={(y_dst_label.long() == 1).sum().item()} "
                    f"| aux_loss_weight={aux_loss_weight}"
                )

            print(msg)

        # ======================================================
        # Stratified train / validation / test split
        # ======================================================

        assert M % 2 == 0, "Expected equal number of negatives/positives. M must be even."

        half = M // 2

        neg_idx_all = torch.arange(0, half, device=device)
        pos_idx_all = torch.arange(half, M, device=device)

        neg_perm = neg_idx_all[torch.randperm(half, device=device)]
        pos_perm = pos_idx_all[torch.randperm(half, device=device)]

        train_size_per_class = int(train_ratio * half)
        val_size_per_class = int(val_ratio * half)

        # Everything left goes to test so that no samples are lost
        test_size_per_class = half - train_size_per_class - val_size_per_class

        if train_size_per_class <= 0 or val_size_per_class <= 0 or test_size_per_class <= 0:
            raise ValueError(
                "[LP-GNN] Split too small. Need at least one positive and one negative sample "
                "in train, validation, and test."
            )

        neg_train = neg_perm[:train_size_per_class]
        neg_val = neg_perm[train_size_per_class:train_size_per_class + val_size_per_class]
        neg_test = neg_perm[train_size_per_class + val_size_per_class:]

        pos_train = pos_perm[:train_size_per_class]
        pos_val = pos_perm[train_size_per_class:train_size_per_class + val_size_per_class]
        pos_test = pos_perm[train_size_per_class + val_size_per_class:]

        train_idx = torch.cat([neg_train, pos_train])
        val_idx = torch.cat([neg_val, pos_val])
        test_idx = torch.cat([neg_test, pos_test])

        train_idx = train_idx[torch.randperm(train_idx.numel(), device=device)]
        val_idx = val_idx[torch.randperm(val_idx.numel(), device=device)]
        test_idx = test_idx[torch.randperm(test_idx.numel(), device=device)]

        if verbose:
            print(
                f"[LP-GNN] Split: "
                f"train={train_idx.numel()} | "
                f"val={val_idx.numel()} | "
                f"test={test_idx.numel()}"
            )

        # ======================================================
        # Model
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

        pos_weight = (len(neg_idx_all) / len(pos_idx_all)) if len(pos_idx_all) > 0 else 1.0

        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device)
        )

        if use_aux:
            src_pos = float((y_src_label == 1).sum().item())
            src_neg = float((y_src_label == 0).sum().item())
            dst_pos = float((y_dst_label == 1).sum().item())
            dst_neg = float((y_dst_label == 0).sum().item())

            src_pos_weight = (src_neg / src_pos) if src_pos > 0 else 1.0
            dst_pos_weight = (dst_neg / dst_pos) if dst_pos > 0 else 1.0

            loss_fn_src = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(src_pos_weight, device=device)
            )

            loss_fn_dst = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(dst_pos_weight, device=device)
            )

        # ======================================================
        # Evaluation helper
        # ======================================================

        def _safe_div(num: float, den: float) -> float:
            return float(num / den) if den > 0 else 0.0

        def _compute_metrics_from_logits(logits: torch.Tensor, labels_int: torch.Tensor):
            logits = logits.view(-1)
            labels_int = labels_int.long().view(-1)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            loss_val = loss_fn(logits, labels_int.float())

            tp, fp, tn, fn = PRBCD._confusion_counts(preds, labels_int)

            total = tp + fp + tn + fn

            acc = _safe_div(tp + tn, total)
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            specificity = _safe_div(tn, tn + fp)

            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            auc, ap = PRBCD._safe_auc_ap_sklearn(probs, labels_int)

            pmin = float(probs.min().item()) if probs.numel() > 0 else float("nan")
            pmean = float(probs.mean().item()) if probs.numel() > 0 else float("nan")
            pmax = float(probs.max().item()) if probs.numel() > 0 else float("nan")

            return {
                "loss": float(loss_val.item()),
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
                "p_min": pmin,
                "p_mean": pmean,
                "p_max": pmax,
            }

        def _evaluate_split(split_idx: torch.Tensor):
            model.eval()

            with torch.no_grad():
                if use_aux:
                    out = model(
                        x,
                        edge_index_struct,
                        edge_index_lab[:, split_idx],
                        return_aux=True,
                    )
                    logits = out["edge_logits"].view(-1)
                else:
                    logits = model(
                        x,
                        edge_index_struct,
                        edge_index_lab[:, split_idx],
                    ).view(-1)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    return None

                labels = y_int[split_idx]

                return _compute_metrics_from_logits(logits, labels)

        # ======================================================
        # CSV logger
        # ======================================================

        csv_fields = [
            "epoch",

            "train_loss",
            "val_loss",
            "test_loss",

            "train_acc",
            "val_acc",
            "test_acc",

            "train_precision",
            "val_precision",
            "test_precision",

            "train_recall",
            "val_recall",
            "test_recall",

            "train_f1",
            "val_f1",
            "test_f1",

            "train_specificity",
            "val_specificity",
            "test_specificity",

            "train_auc",
            "val_auc",
            "test_auc",

            "train_ap",
            "val_ap",
            "test_ap",

            "train_tp",
            "train_fp",
            "train_tn",
            "train_fn",

            "val_tp",
            "val_fp",
            "val_tn",
            "val_fn",

            "test_tp",
            "test_fp",
            "test_tn",
            "test_fn",

            "train_p_min",
            "train_p_mean",
            "train_p_max",

            "val_p_min",
            "val_p_mean",
            "val_p_max",

            "test_p_min",
            "test_p_mean",
            "test_p_max",

            "grad_norm",
            "best_metric",
            "is_best",
        ]

        csv_path_was_auto = csv_path is None

        if csv_path is None:
            csv_path = PRBCD.make_selector_gnn_log_path(
                ads_mode=self.ads_mode,
                drop_mode=self.drop_mode,
                k_samples_batch=self.k_samples_batch,
                n_candidates_k_sample=self.n_candidates_k_sample,
                acc_drop_threshold_k_samples=self.acc_drop_threshold_k_samples,
                loss_drop_threshold_k_samples=self.loss_drop_threshold_k_samples,
                dataset=self.dataset,
            )

        logger = CSVMetricLogger(
            path=csv_path,
            fieldnames=csv_fields,
            append=csv_append,
        )

        # ======================================================
        # Early stopping
        # ======================================================

        metric_mode = {
            "val_auc": "max",
            "val_ap": "max",
            "val_acc": "max",
            "val_f1": "max",
            "val_recall": "max",
            "val_loss": "min",
        }

        # Backward compatibility with your old names
        if early_stop_metric == "auc":
            early_stop_metric = "val_auc"
        elif early_stop_metric == "ap":
            early_stop_metric = "val_ap"

        if early_stop_metric not in metric_mode:
            raise ValueError(
                f"early_stop_metric must be one of {list(metric_mode.keys())}, "
                f"or old aliases 'auc' / 'ap'."
            )

        want = metric_mode[early_stop_metric]

        best_metric = -float("inf") if want == "max" else float("inf")
        best_epoch = -1
        best_state = None

        def _is_improvement(curr: float, best: float) -> bool:
            if want == "max":
                return curr > best + early_stop_min_delta
            else:
                return curr < best - early_stop_min_delta

        patience_left = int(early_stop_patience)

        stopped_epoch = 0
        stopped_metric_value = None
        early_stopped = False

        epoch_iter = tqdm(range(num_epochs), desc="[LP-GNN] Training") if use_tqdm else range(num_epochs)

        # ======================================================
        # Training loop
        # ======================================================

        for epoch in epoch_iter:
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

                if torch.isnan(logits_train).any() or torch.isinf(logits_train).any():
                    print(f"[LP-GNN][ERROR] NaN/Inf in train edge logits at epoch {epoch + 1}")
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

                loss = loss_edge + aux_loss_weight * (loss_src + loss_dst)

            else:
                logits_train = model(
                    x,
                    edge_index_struct,
                    edge_index_lab[:, train_idx],
                ).view(-1)

                if torch.isnan(logits_train).any() or torch.isinf(logits_train).any():
                    print(f"[LP-GNN][ERROR] NaN/Inf in train logits at epoch {epoch + 1}")
                    break

                label_smoothing = 0.1

                y_train_soft = (
                        y_label[train_idx] * (1.0 - label_smoothing)
                        + 0.5 * label_smoothing
                )

                loss = loss_fn(logits_train, y_train_soft)

            loss.backward()

            grad_norm_val = None

            if log_grad_norm:
                total_norm = 0.0

                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2

                grad_norm_val = total_norm ** 0.5

            optimizer.step()

            # ==================================================
            # Evaluate train / validation / test
            # ==================================================

            train_metrics = _evaluate_split(train_idx)
            val_metrics = _evaluate_split(val_idx)
            test_metrics = _evaluate_split(test_idx)

            if train_metrics is None:
                print(f"[LP-GNN][ERROR] NaN/Inf during train evaluation at epoch {epoch + 1}")
                break

            if val_metrics is None:
                print(f"[LP-GNN][ERROR] NaN/Inf during validation evaluation at epoch {epoch + 1}")
                break

            if test_metrics is None:
                print(f"[LP-GNN][ERROR] NaN/Inf during test evaluation at epoch {epoch + 1}")
                break

            # Keep your old behavior:
            # train_loss logs the actual training loss, including label smoothing / aux loss.
            train_metrics["loss"] = float(loss.item())

            metric_val = val_metrics[early_stop_metric.replace("val_", "")]
            stopped_epoch = epoch + 1
            stopped_metric_value = float(metric_val) if metric_val is not None else None

            is_best = False

            if metric_val is not None:
                if _is_improvement(float(metric_val), float(best_metric)):
                    best_metric = float(metric_val)
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    patience_left = int(early_stop_patience)
                    is_best = True
                else:
                    patience_left -= 1

            # ==================================================
            # CSV log
            # ==================================================

            logger.log({
                "epoch": epoch + 1,

                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "test_loss": test_metrics["loss"],

                "train_acc": train_metrics["acc"],
                "val_acc": val_metrics["acc"],
                "test_acc": test_metrics["acc"],

                "train_precision": train_metrics["precision"],
                "val_precision": val_metrics["precision"],
                "test_precision": test_metrics["precision"],

                "train_recall": train_metrics["recall"],
                "val_recall": val_metrics["recall"],
                "test_recall": test_metrics["recall"],

                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "test_f1": test_metrics["f1"],

                "train_specificity": train_metrics["specificity"],
                "val_specificity": val_metrics["specificity"],
                "test_specificity": test_metrics["specificity"],

                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "test_auc": test_metrics["auc"],

                "train_ap": train_metrics["ap"],
                "val_ap": val_metrics["ap"],
                "test_ap": test_metrics["ap"],

                "train_tp": train_metrics["tp"],
                "train_fp": train_metrics["fp"],
                "train_tn": train_metrics["tn"],
                "train_fn": train_metrics["fn"],

                "val_tp": val_metrics["tp"],
                "val_fp": val_metrics["fp"],
                "val_tn": val_metrics["tn"],
                "val_fn": val_metrics["fn"],

                "test_tp": test_metrics["tp"],
                "test_fp": test_metrics["fp"],
                "test_tn": test_metrics["tn"],
                "test_fn": test_metrics["fn"],

                "train_p_min": train_metrics["p_min"],
                "train_p_mean": train_metrics["p_mean"],
                "train_p_max": train_metrics["p_max"],

                "val_p_min": val_metrics["p_min"],
                "val_p_mean": val_metrics["p_mean"],
                "val_p_max": val_metrics["p_max"],

                "test_p_min": test_metrics["p_min"],
                "test_p_mean": test_metrics["p_mean"],
                "test_p_max": test_metrics["p_max"],

                "grad_norm": grad_norm_val,
                "best_metric": float(best_metric) if best_epoch != -1 else float("nan"),
                "is_best": 1 if is_best else 0,
            })

            # ==================================================
            # tqdm
            # ==================================================

            if use_tqdm:
                postfix = {
                    "tr_loss": f"{train_metrics['loss']:.4f}",
                    "va_loss": f"{val_metrics['loss']:.4f}",
                    "te_loss": f"{test_metrics['loss']:.4f}",
                    "va_acc": f"{val_metrics['acc']:.3f}",
                    "va_f1": f"{val_metrics['f1']:.3f}",
                    "va_rec": f"{val_metrics['recall']:.3f}",
                    "pat": patience_left if metric_val is not None else "n/a",
                }

                if val_metrics["auc"] is not None:
                    postfix["va_auc"] = f"{val_metrics['auc']:.3f}"

                if val_metrics["ap"] is not None:
                    postfix["va_ap"] = f"{val_metrics['ap']:.3f}"

                epoch_iter.set_postfix(postfix)

            # ==================================================
            # Print
            # ==================================================

            if verbose and (
                    (epoch + 1) % log_every == 0
                    or epoch == 0
                    or (epoch + 1) == num_epochs
            ):
                msg = (
                    f"[LP-GNN] Epoch {epoch + 1:03d}/{num_epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"test_loss={test_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | "
                    f"val_acc={val_metrics['acc']:.4f} | "
                    f"test_acc={test_metrics['acc']:.4f} | "
                    f"val_precision={val_metrics['precision']:.4f} | "
                    f"val_recall={val_metrics['recall']:.4f} | "
                    f"val_f1={val_metrics['f1']:.4f} | "
                    f"val_TP/FP/TN/FN="
                    f"{val_metrics['tp']}/{val_metrics['fp']}/"
                    f"{val_metrics['tn']}/{val_metrics['fn']} | "
                    f"val_p(min/mean/max)="
                    f"{val_metrics['p_min']:.3f}/"
                    f"{val_metrics['p_mean']:.3f}/"
                    f"{val_metrics['p_max']:.3f}"
                )

                if val_metrics["auc"] is not None:
                    msg += f" | val_AUC={val_metrics['auc']:.4f}"

                if val_metrics["ap"] is not None:
                    msg += f" | val_AP={val_metrics['ap']:.4f}"

                if metric_val is not None:
                    msg += (
                        f" | best_{early_stop_metric}="
                        f"{best_metric:.4f} "
                        f"(epoch {best_epoch})"
                    )

                if grad_norm_val is not None:
                    msg += f" | grad_norm={grad_norm_val:.3e}"

                if use_aux:
                    msg += " | multitask_aux=on"

                print(msg)

            # ==================================================
            # Early stopping
            # ==================================================

            if (
                    early_stop
                    and metric_val is not None
                    and (epoch + 1) >= min_epochs_before_early_stop
                    and patience_left <= 0
            ):
                early_stopped = True
                if verbose:
                    print(
                        f"[LP-GNN] Early stopping at epoch {epoch + 1}. "
                        f"Best {early_stop_metric}={best_metric:.4f} "
                        f"at epoch {best_epoch}."
                    )
                break

        logger.close()

        if csv_path_was_auto and not csv_append:
            import os

            def _filename_safe(value) -> str:
                text = str(value)
                return (
                    text.replace("/", "-")
                    .replace("\\", "-")
                    .replace(" ", "")
                    .replace(":", "-")
                    .replace(".", "p")
                )

            metric_name_safe = _filename_safe(early_stop_metric)
            metric_value_safe = (
                _filename_safe(f"{stopped_metric_value:.6g}")
                if stopped_metric_value is not None
                else "nan"
            )
            stop_state = "earlystop" if early_stopped else "completed"
            stop_suffix = (
                f"_{stop_state}_esm-{metric_name_safe}"
                f"_stop-e{int(stopped_epoch):03d}"
                f"_stopval-{metric_value_safe}"
            )

            root, ext = os.path.splitext(csv_path)
            csv_path_with_stop_info = f"{root}{stop_suffix}{ext}"

            if csv_path_with_stop_info != csv_path:
                os.replace(csv_path, csv_path_with_stop_info)
                csv_path = csv_path_with_stop_info

        # ======================================================
        # Restore best validation model
        # ======================================================

        if restore_best and best_state is not None:
            model.load_state_dict(best_state)

            if verbose:
                print(
                    f"[LP-GNN] Restored best model from epoch {best_epoch} "
                    f"({early_stop_metric}={best_metric:.4f})."
                )

        if verbose:
            print(f"[LP-GNN] Training complete. Metrics saved to '{csv_path}'")

        return model

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

    def build_full_idx_matrix(self, reset: bool, sample_size: int):
        if reset:
            # empty edges_to_attack_index for resample or other cases
            self.edges_to_attack_index = torch.empty((2, 0), dtype=torch.long)
        nodes_1 = torch.from_numpy(np.random.choice(self.current_node_search_space, size=sample_size, replace=True))
        # We could change this to only draw nodes_1 from unrobust nodes and nodes_2 from all possible nodes
        # Done: see next method
        nodes_2 = torch.from_numpy(np.random.choice(self.current_node_search_space, size=sample_size, replace=True))
        edges_idx = torch.cat([nodes_1.unsqueeze(0), nodes_2.unsqueeze(0)], dim=0)
        self.edges_to_attack_index = torch.cat([self.edges_to_attack_index, edges_idx], dim=1)
        return edges_idx

    def build_full_idx_matrix_semi(self, reset: bool, sample_size: int):
        if reset:
            # empty edges_to_attack_index for resample or other cases
            self.edges_to_attack_index = torch.empty((2, 0), dtype=torch.long)
        nodes_1 = torch.from_numpy(np.random.choice(self.current_node_search_space, size=sample_size, replace=True))
        nodes_2 = torch.randint(self.n, (sample_size,), device=self.device)
        edges_idx = torch.cat([nodes_1.unsqueeze(0), nodes_2.unsqueeze(0)], dim=0)
        self.edges_to_attack_index = torch.cat([self.edges_to_attack_index, edges_idx], dim=1)
        return edges_idx

    def setup_search_space_undirected(self, n: int):
        # matrix: random drawn block matrix this matrix is drawn from two node sets which are
        # not robust according to certificates
        # from here we want to follow same steps as sample_random_block
        self.current_search_space = self.edges_to_current_search_space(n)

        #while self.current_search_space.size(0) < self.block_size:
        #    # here we want to fill up current_search_space so that there are block_size many entries
        #    if not self.semi:
        #        self.build_full_idx_matrix(False, self.block_size)
        #    else:
        #        self.build_full_idx_matrix_semi(False, self.block_size)
        #    self.current_search_space = self.edges_to_current_search_space(n)

        # now we follow same steps as sample_random_block
        #self.current_search_space = self.current_search_space[
        #                                torch.randperm(self.current_search_space.size(0))[:self.block_size]
        #                                ]
        self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
        # if the new logic is usable this method is redundant
        return

    def edges_to_current_search_space(self, n: int):
        # first we cut edges so that index build a triu matrix
        # (function triu_idx_to_linear only support triu matrix idx)
        self.edges_to_attack_index = PRBCD.flip_matrix_idx_to_triu_idx(self.edges_to_attack_index)
        # we then build linear idx which is the current_search_space
        lin_idx = PRBCD.triu_idx_to_linear_idx(n, self.edges_to_attack_index)
        lin_idx = torch.unique(lin_idx, sorted=True)
        return lin_idx

    def sample_current_node_search_space_det(self, grid_binary_class):
        if self.use_cert in ("sampling_grid_binary_class_alt_11",):
            print("running alternative sampling with grid binary class (1,1)")
            unrobust_nodes_lin_index = np.where(grid_binary_class[:, 1, 1] < 0.3)  # shape: (2810,)
        else:
            print("running alternative sampling with grid binary class (2,2)")
            unrobust_nodes_lin_index = np.where(grid_binary_class[:, 2, 2] < 0.2)
        unrobust_nodes_lin_index = unrobust_nodes_lin_index[0]
        self.current_node_search_space = torch.unique(torch.from_numpy(unrobust_nodes_lin_index), sorted=False)
        return

    def sample_current_node_search_space_random_cert(self, grid_binary_class, sample_size):
        search_space = np.array([], dtype=np.int64)
        print("running random certificate samples")
        # Todo: Try different setups like different sample_size, while loop only to ensure no empty search_space, etc.
        while search_space.shape[0] <= 0:
            unrobust_nodes_lin_idx = np.where(grid_binary_class[:,
                                              np.random.randint(0, 3),
                                              np.random.randint(0, 5)] < np.random.uniform(0.1, 0.4))
            unrobust_nodes_lin_idx = unrobust_nodes_lin_idx[0]
            search_space = np.concatenate([search_space, unrobust_nodes_lin_idx], axis=0)

        self.current_node_search_space = torch.unique(torch.from_numpy(search_space), sorted=False)
        return

    def _append_attack_statistics(self, loss: float, accuracy: float,
                                  probability_mass_update: float, probability_mass_projected: float):
        self.attack_statistics['loss'].append(loss)
        self.attack_statistics['accuracy'].append(accuracy)
        self.attack_statistics['nonzero_weights'].append((self.perturbed_edge_weight > self.eps).sum().item())
        self.attack_statistics['probability_mass_update'].append(probability_mass_update)
        self.attack_statistics['probability_mass_projected'].append(probability_mass_projected)

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
    def save_selection(
            path,
            y_out,
            edge_index_lab,
            y_label,
            tried_set,
            harmful_set,
            sub_nodes=None,
            edge_index_sub=None,
            edge_weight_sub=None,
            edge_index_lab_local=None,
            X_sub=None,
            edge_index_struct_local=None,
            meta=None,
    ):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "y_out": y_out.detach().cpu() if torch.is_tensor(y_out) else y_out,
            "edge_index_lab": edge_index_lab.detach().cpu(),
            "y_label": y_label.detach().cpu(),
            "tried_set": tried_set,
            "harmful_set": harmful_set,
            "meta": meta or {},
        }

        if sub_nodes is not None:
            payload.update({
                "sub_nodes": sub_nodes.detach().cpu() if torch.is_tensor(sub_nodes) else sub_nodes,
                "edge_index_sub": edge_index_sub.detach().cpu() if torch.is_tensor(edge_index_sub) else edge_index_sub,
                "edge_weight_sub": edge_weight_sub.detach().cpu() if torch.is_tensor(
                    edge_weight_sub) else edge_weight_sub,
                "edge_index_lab_local": edge_index_lab_local.detach().cpu() if torch.is_tensor(edge_index_lab_local) else edge_index_lab_local,
                "X_sub": X_sub.detach().cpu() if torch.is_tensor(X_sub) else X_sub,
                "edge_index_struct_local": edge_index_struct_local.detach().cpu() if torch.is_tensor(edge_index_struct_local) else edge_index_struct_local,
            })

        torch.save(payload, path)

    @staticmethod
    def load_selection(path, device="cpu"):
        payload = torch.load(path, map_location="cpu")

        # ---- mandatory fields ----
        y_out = payload["y_out"]
        if torch.is_tensor(y_out):
            y_out = y_out.to(device)

        edge_index_lab = payload["edge_index_lab"]
        if torch.is_tensor(edge_index_lab):
            edge_index_lab = edge_index_lab.to(device)

        y_label = payload["y_label"]
        if torch.is_tensor(y_label):
            y_label = y_label.to(device)

        tried_set = payload["tried_set"]
        harmful_set = payload["harmful_set"]

        # ---- optional subgraph fields ----
        sub_nodes = payload.get("sub_nodes", None)
        if torch.is_tensor(sub_nodes):
            sub_nodes = sub_nodes.to(device)

        edge_index_sub = payload.get("edge_index_sub", None)
        if torch.is_tensor(edge_index_sub):
            edge_index_sub = edge_index_sub.to(device)

        edge_weight_sub = payload.get("edge_weight_sub", None)
        if torch.is_tensor(edge_weight_sub):
            edge_weight_sub = edge_weight_sub.to(device)

        edge_index_lab_local = payload.get("edge_index_lab_local", None)
        if torch.is_tensor(edge_index_lab_local):
            edge_index_lab_local = edge_index_lab_local.to(device)

        X_sub = payload.get("X_sub", None)
        if torch.is_tensor(X_sub):
            X_sub = X_sub.to(device)

        edge_index_struct_local = payload.get("edge_index_struct_local", None)
        if torch.is_tensor(edge_index_struct_local):
            edge_index_struct_local = edge_index_struct_local.to(device)

        meta = payload.get("meta", {})

        return (
            y_out,
            edge_index_lab,
            y_label,
            tried_set,
            harmful_set,
            sub_nodes,
            edge_index_sub,
            edge_weight_sub,
            edge_index_lab_local,
            X_sub,
            edge_index_struct_local,
            meta,
        )

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

    @staticmethod
    def _confusion_counts(preds: torch.Tensor, y: torch.Tensor):
        tp = int(((preds == 1) & (y == 1)).sum().item())
        tn = int(((preds == 0) & (y == 0)).sum().item())
        fp = int(((preds == 1) & (y == 0)).sum().item())
        fn = int(((preds == 0) & (y == 1)).sum().item())
        return tp, fp, tn, fn

    @staticmethod
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

    @staticmethod
    def make_selector_gnn_log_path(
            base_dir: str = "Plotting_Data/SelectorGNNLogs",
            ads_mode: str | None = None,
            drop_mode: str | None = None,
            n_candidates_k_sample: int | None = None,
            k_samples_batch: int | None = None,
            acc_drop_threshold_k_samples: float | None = None,
            loss_drop_threshold_k_samples: float | None = None,
            dataset: str | None = None,
            ext: str = ".csv",
    ):
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        parts = ["selector_gnn"]

        if ads_mode:
            parts.append(f"dataset-{dataset}")
        if ads_mode:
            parts.append(f"ads-{ads_mode}")
        if n_candidates_k_sample is not None:
            parts.append(f"k{n_candidates_k_sample}")
        if k_samples_batch is not None:
            parts.append(f"kb{k_samples_batch}")
        if drop_mode is not None:
            parts.append(f"drpmd{drop_mode}")
        if drop_mode == "acc":
            if acc_drop_threshold_k_samples is not None:
                parts.append(f"drp{acc_drop_threshold_k_samples}")
        if drop_mode == "loss":
            if loss_drop_threshold_k_samples is not None:
                parts.append(f"drp{loss_drop_threshold_k_samples}")

        filename = "_".join(parts) + f"_{timestamp}{ext}"
        return os.path.join(base_dir, filename)

    @staticmethod
    def _build_undirected_edge_set(edge_index_struct: torch.Tensor):
        src = edge_index_struct[0].tolist()
        dst = edge_index_struct[1].tolist()
        undirected = set()
        for u, v in zip(src, dst):
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            undirected.add((a, b))
        return undirected

    @staticmethod
    def _uppertri_decode(k: int, n: int):
        # Decodes k in [0, n*(n-1)/2) into (u,v) with u < v
        # Using inverse triangular numbers.
        # count of pairs starting at u: (n-u-1)
        # prefix(u) = sum_{i=0}^{u-1} (n-i-1) = u*(2n-u-1)/2
        # find u such that prefix(u) <= k < prefix(u+1)
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            prefix = mid * (2 * n - mid - 1) // 2
            prefix_next = (mid + 1) * (2 * n - (mid + 1) - 1) // 2
            if k < prefix:
                hi = mid
            elif k >= prefix_next:
                lo = mid + 1
            else:
                lo = mid
                break
        u = lo
        prefix_u = u * (2 * n - u - 1) // 2
        offset = k - prefix_u
        v = u + 1 + offset
        return int(u), int(v)

    @staticmethod
    def tried_add_del_proportion(tried_set, edge_index_struct, n):
        undirected_edges = PRBCD._build_undirected_edge_set(edge_index_struct)

        n_del_like = 0  # pair exists in graph -> would be a deletion candidate
        n_add_like = 0  # pair not in graph -> would be an addition candidate
        missing_decode = 0

        for (k_lin, _drop) in tried_set:
            k_lin = int(k_lin)
            if k_lin < 0 or k_lin >= n * (n - 1) // 2:
                missing_decode += 1
                continue
            u, v = PRBCD._uppertri_decode(k_lin, n)
            if (u, v) in undirected_edges:
                n_del_like += 1
            else:
                n_add_like += 1

        total = n_del_like + n_add_like
        prop_del = n_del_like / total if total else 0.0
        prop_add = n_add_like / total if total else 0.0

        return {
            "total_tried": total,
            "del_like": n_del_like,
            "add_like": n_add_like,
            "prop_del": prop_del,
            "prop_add": prop_add,
            "skipped": missing_decode,
        }

    def _load_selector_params(self, selector_params: dict, ads_mode=None):
        selector_params = selector_params or {}

        self.ads_mode = selector_params.get(
            "accuracy_drop_selector_mode",
            ads_mode,
        )

        self.n_candidates_k_sample = selector_params.get(
            "n_candidates_k_sample",
            2000,
        )

        self.n_candidates_one_sample = selector_params.get(
            "n_candidates_one_sample",
            5000,
        )

        self.drop_mode = selector_params.get(
            "drop_mode",
            "endpoint",
        )

        self.acc_drop_threshold_k_samples = selector_params.get(
            "acc_drop_threshold_k_samples",
            1e-3,
        )

        self.loss_drop_threshold_k_samples = selector_params.get(
            "loss_drop_threshold_k_samples",
            1e-3,
        )

        self.k_samples_batch = selector_params.get(
            "k_samples_batch",
            10,
        )

        self.tau = selector_params.get(
            "tau",
            0.8,
        )

        self.score_batch_size = selector_params.get(
            "score_batch_size",
            1000,
        )

        self.max_sampling_tries = selector_params.get(
            "max_sampling_tries",
            2_000_000,
        )

        self.exclude_tried = selector_params.get(
            "exclude_tried",
            True,
        )

        self.lp_hit_rate_detour = selector_params.get("lp_hit_rate_detour", False)
        self.lp_hit_rate_top_k = selector_params.get("lp_hit_rate_top_k", 200)
        self.lp_hit_rate_out_dir = selector_params.get(
            "lp_hit_rate_out_dir",
            "extendedPlotting/lpEndpointHitRate",
        )

        self.training_data_node_cap = selector_params.get("training_data_node_cap", 0)

    @staticmethod
    def print_linear_edge_set_node_dominance(
            name: str,
            edge_set,
            n_nodes: int,
            top_k: int = 10,
            device=None,
    ):
        """
        Prints node dominance statistics for an edge set containing entries like:
            (linear_edge_index, drop)

        The linear edge index is interpreted as an upper-triangle PRBCD edge index.
        """
        if edge_set is None or len(edge_set) == 0:
            print(f"[{name} NODE STATS] empty set; no node dominance statistics available.")
            return

        if device is None:
            device = torch.device("cpu")

        lin_indices = torch.tensor(
            [int(item[0]) for item in edge_set],
            dtype=torch.long,
            device=device,
        )

        # Convert linear upper-triangle indices back to endpoint pairs.
        edge_index = PRBCD.linear_to_triu_idx(int(n_nodes), lin_indices)

        endpoints = edge_index.reshape(-1).detach().cpu()

        node_counts = torch.bincount(endpoints, minlength=int(n_nodes))
        active_counts = node_counts[node_counts > 0]

        n_edges = len(edge_set)
        n_endpoint_occurrences = int(endpoints.numel())
        n_active_nodes = int(active_counts.numel())

        top_k = min(top_k, n_active_nodes)
        top_counts, top_nodes = torch.topk(node_counts, k=top_k)

        top1_share = float(top_counts[:1].sum().item() / n_endpoint_occurrences)
        top5_share = float(top_counts[:min(5, top_k)].sum().item() / n_endpoint_occurrences)
        top10_share = float(top_counts[:min(10, top_k)].sum().item() / n_endpoint_occurrences)

        # Effective number of nodes:
        # lower = stronger domination by few nodes
        probs = active_counts.float() / active_counts.sum().float()
        effective_nodes = float(1.0 / torch.sum(probs ** 2).item())

        # Gini coefficient:
        # 0 = even distribution, close to 1 = concentrated on few nodes
        sorted_counts = torch.sort(active_counts.float()).values
        m = sorted_counts.numel()

        if m > 1:
            idx = torch.arange(1, m + 1, dtype=torch.float32)
            gini = float(
                (2.0 * torch.sum(idx * sorted_counts) / (m * torch.sum(sorted_counts)))
                - (m + 1.0) / m
            )
        else:
            gini = 0.0

        print(
            f"[{name} NODE STATS] edges={n_edges} | "
            f"endpoint_occurrences={n_endpoint_occurrences} | "
            f"active_nodes={n_active_nodes}/{int(n_nodes)} | "
            f"effective_nodes={effective_nodes:.2f} | "
            f"gini={gini:.4f} | "
            f"top1_endpoint_share={100.0 * top1_share:.2f}% | "
            f"top5_endpoint_share={100.0 * top5_share:.2f}% | "
            f"top10_endpoint_share={100.0 * top10_share:.2f}%"
        )

        print(
            f"[{name} NODE TOP{top_k}] "
            + ", ".join(
                f"node={int(node)}:count={int(count)}"
                for node, count in zip(top_nodes.tolist(), top_counts.tolist())
                if count > 0
            )
        )

    @staticmethod
    def print_linear_tensor_node_dominance(
            name: str,
            lin_indices: torch.Tensor,
            n_nodes: int,
            top_k: int = 10,
            device=None,
    ):
        """
        Prints node dominance statistics for a tensor of PRBCD upper-tri
        linear edge indices, e.g. self.current_search_space.
        """
        if lin_indices is None or lin_indices.numel() == 0:
            print(f"[{name} NODE STATS] empty tensor; no node dominance statistics available.")
            return

        if device is None:
            device = torch.device("cpu")

        lin_indices = lin_indices.detach().to(device=device, dtype=torch.long)

        # Convert PRBCD upper-triangle linear indices back to endpoint pairs.
        edge_index = PRBCD.linear_to_triu_idx(int(n_nodes), lin_indices)

        endpoints = edge_index.reshape(-1).detach().cpu()
        node_counts = torch.bincount(endpoints, minlength=int(n_nodes))
        active_counts = node_counts[node_counts > 0]

        n_edges = int(lin_indices.numel())
        n_endpoint_occurrences = int(endpoints.numel())
        n_active_nodes = int(active_counts.numel())

        if n_active_nodes == 0:
            print(f"[{name} NODE STATS] no active nodes.")
            return

        top_k = min(int(top_k), n_active_nodes)
        top_counts, top_nodes = torch.topk(node_counts, k=top_k)

        top1_share = float(top_counts[:1].sum().item() / n_endpoint_occurrences)
        top5_share = float(top_counts[:min(5, top_k)].sum().item() / n_endpoint_occurrences)
        top10_share = float(top_counts[:min(10, top_k)].sum().item() / n_endpoint_occurrences)

        probs = active_counts.float() / active_counts.sum().float()
        effective_nodes = float(1.0 / torch.sum(probs ** 2).item())

        sorted_counts = torch.sort(active_counts.float()).values
        m = sorted_counts.numel()

        if m > 1:
            idx = torch.arange(1, m + 1, dtype=torch.float32)
            gini = float(
                (2.0 * torch.sum(idx * sorted_counts) / (m * torch.sum(sorted_counts)))
                - (m + 1.0) / m
            )
        else:
            gini = 0.0

        print(
            f"[{name} NODE STATS] edges={n_edges} | "
            f"endpoint_occurrences={n_endpoint_occurrences} | "
            f"active_nodes={n_active_nodes}/{int(n_nodes)} | "
            f"effective_nodes={effective_nodes:.2f} | "
            f"gini={gini:.4f} | "
            f"top1_endpoint_share={100.0 * top1_share:.2f}% | "
            f"top5_endpoint_share={100.0 * top5_share:.2f}% | "
            f"top10_endpoint_share={100.0 * top10_share:.2f}%"
        )

        print(f"[{name} TOP NODES]")
        for node, count in zip(top_nodes.tolist(), top_counts.tolist()):
            share = 100.0 * float(count) / float(n_endpoint_occurrences)
            print(f"  node={node} | endpoint_count={count} | endpoint_share={share:.2f}%")

    @staticmethod
    def _append_dict_to_csv(csv_path: str, row: dict):
        import csv
        import os

        if csv_path is None:
            return

        directory = os.path.dirname(csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        file_exists = os.path.exists(csv_path)

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _linear_edge_set_node_dominance_stats(
            name: str,
            edge_set,
            n_nodes: int,
            top_k: int = 10,
            device=None,
            extra: dict | None = None,
    ) -> dict:
        import torch

        if extra is None:
            extra = {}

        if device is None:
            device = torch.device("cpu")

        base = {
            **extra,
            "name": str(name),
            "edges": 0,
            "endpoint_occurrences": 0,
            "active_nodes": 0,
            "n_nodes": int(n_nodes),
            "effective_nodes": 0.0,
            "gini": 0.0,
            "top1_endpoint_share": 0.0,
            "top5_endpoint_share": 0.0,
            "top10_endpoint_share": 0.0,
            "top_nodes": "",
            "top_counts": "",
        }

        if edge_set is None or len(edge_set) == 0:
            return base

        lin_indices = torch.tensor(
            [int(item[0]) for item in edge_set],
            dtype=torch.long,
            device=device,
        )

        edge_index = PRBCD.linear_to_triu_idx(int(n_nodes), lin_indices)
        endpoints = edge_index.reshape(-1).detach().cpu()

        node_counts = torch.bincount(endpoints, minlength=int(n_nodes))
        active_counts = node_counts[node_counts > 0]

        n_edges = int(len(edge_set))
        n_endpoint_occurrences = int(endpoints.numel())
        n_active_nodes = int(active_counts.numel())

        top_k = min(int(top_k), n_active_nodes)
        if top_k > 0:
            top_counts, top_nodes = torch.topk(node_counts, k=top_k)
        else:
            top_counts = torch.empty(0, dtype=torch.long)
            top_nodes = torch.empty(0, dtype=torch.long)

        top1_share = float(top_counts[:1].sum().item() / n_endpoint_occurrences) if n_endpoint_occurrences else 0.0
        top5_share = float(top_counts[:min(5, top_k)].sum().item() / n_endpoint_occurrences) if n_endpoint_occurrences else 0.0
        top10_share = float(top_counts[:min(10, top_k)].sum().item() / n_endpoint_occurrences) if n_endpoint_occurrences else 0.0

        probs = active_counts.float() / active_counts.sum().float()
        effective_nodes = float(1.0 / torch.sum(probs ** 2).item()) if active_counts.numel() > 0 else 0.0

        sorted_counts = torch.sort(active_counts.float()).values
        m = sorted_counts.numel()

        if m > 1:
            idx = torch.arange(1, m + 1, dtype=torch.float32)
            gini = float(
                (2.0 * torch.sum(idx * sorted_counts) / (m * torch.sum(sorted_counts)))
                - (m + 1.0) / m
            )
        else:
            gini = 0.0

        top_items = [
            (int(node), int(count))
            for node, count in zip(top_nodes.tolist(), top_counts.tolist())
            if int(count) > 0
        ]

        return {
            **base,
            "edges": n_edges,
            "endpoint_occurrences": n_endpoint_occurrences,
            "active_nodes": n_active_nodes,
            "effective_nodes": float(effective_nodes),
            "gini": float(gini),
            "top1_endpoint_share": float(top1_share),
            "top5_endpoint_share": float(top5_share),
            "top10_endpoint_share": float(top10_share),
            "top_nodes": ";".join(str(node) for node, _count in top_items),
            "top_counts": ";".join(str(count) for _node, count in top_items),
        }

    @staticmethod
    def _print_node_dominance_stats(stats: dict, top_k: int = 10):
        name = stats["name"]

        if int(stats["edges"]) == 0:
            print(f"[{name} NODE STATS] empty set; no node dominance statistics available.")
            return

        print(
            f"[{name} NODE STATS] edges={int(stats['edges'])} | "
            f"endpoint_occurrences={int(stats['endpoint_occurrences'])} | "
            f"active_nodes={int(stats['active_nodes'])}/{int(stats['n_nodes'])} | "
            f"effective_nodes={float(stats['effective_nodes']):.2f} | "
            f"gini={float(stats['gini']):.4f} | "
            f"top1_endpoint_share={100.0 * float(stats['top1_endpoint_share']):.2f}% | "
            f"top5_endpoint_share={100.0 * float(stats['top5_endpoint_share']):.2f}% | "
            f"top10_endpoint_share={100.0 * float(stats['top10_endpoint_share']):.2f}%"
        )

        top_nodes = [x for x in str(stats.get("top_nodes", "")).split(";") if x != ""]
        top_counts = [x for x in str(stats.get("top_counts", "")).split(";") if x != ""]
        top_items = list(zip(top_nodes, top_counts))[:int(top_k)]

        print(
            f"[{name} NODE TOP{len(top_items)}] "
            + ", ".join(
                f"node={int(node)}:count={int(count)}"
                for node, count in top_items
            )
        )

    @staticmethod
    def record_selection_statistics(
            tried_set,
            harmful_set,
            n_nodes: int,
            device=None,
            stats_dir: str = "cache",
            csv_prefix: str = "selection",
            top_k: int = 10,
            extra: dict | None = None,
    ) -> dict:
        """
        Prints and records all selection statistics in CSV files.

        This replaces the inline block that printed:
          - TRIED node dominance
          - HARMFUL node dominance
          - tried/harmful ratio

        It writes two CSVs:
          - {stats_dir}/{csv_prefix}_node_dominance.csv
          - {stats_dir}/{csv_prefix}_selection_stats.csv
        """
        import os

        if extra is None:
            extra = {}

        os.makedirs(stats_dir, exist_ok=True)

        node_stats_csv_path = os.path.join(stats_dir, f"{csv_prefix}_node_dominance.csv")
        selection_stats_csv_path = os.path.join(stats_dir, f"{csv_prefix}_selection_stats.csv")

        n_tried = int(len(tried_set) if tried_set is not None else 0)
        n_harmful = int(len(harmful_set) if harmful_set is not None else 0)
        harmful_ratio = float(n_harmful / n_tried) if n_tried > 0 else 0.0

        tried_node_stats = PRBCD._linear_edge_set_node_dominance_stats(
            name="TRIED",
            edge_set=tried_set,
            n_nodes=int(n_nodes),
            top_k=int(top_k),
            device=device,
            extra=extra,
        )
        harmful_node_stats = PRBCD._linear_edge_set_node_dominance_stats(
            name="HARMFUL",
            edge_set=harmful_set,
            n_nodes=int(n_nodes),
            top_k=int(top_k),
            device=device,
            extra=extra,
        )

        PRBCD._print_node_dominance_stats(tried_node_stats, top_k=top_k)
        PRBCD._print_node_dominance_stats(harmful_node_stats, top_k=top_k)

        PRBCD._append_dict_to_csv(node_stats_csv_path, tried_node_stats)
        PRBCD._append_dict_to_csv(node_stats_csv_path, harmful_node_stats)

        selection_stats = {
            **extra,
            "tried_edges": n_tried,
            "harmful_edges": n_harmful,
            "harmful_ratio": harmful_ratio,
            "harmful_ratio_percent": 100.0 * harmful_ratio,
        }

        PRBCD._append_dict_to_csv(selection_stats_csv_path, selection_stats)

        print(
            f"[SELECTION STATS] tried_edges={n_tried} | "
            f"harmful_edges={n_harmful} | "
            f"harmful/tried={harmful_ratio:.4f} "
            f"({100.0 * harmful_ratio:.2f}%)"
        )

        return {
            "tried_node_stats": tried_node_stats,
            "harmful_node_stats": harmful_node_stats,
            "selection_stats": selection_stats,
            "node_stats_csv_path": node_stats_csv_path,
            "selection_stats_csv_path": selection_stats_csv_path,
        }