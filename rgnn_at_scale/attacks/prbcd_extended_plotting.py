import copy
import logging

import time

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
from scipy.stats import spearmanr
#from AttackerGNN.PreEdgeSelector import PRBCDSelectorGNN
from AttackerGNN.PriorSelector import PriorSelector
from AttackerGNN.NodeBlockScorer import NodeBlockScorer
import AttackerGNN.gnn_least_likely_edge as lle
import AttackerGNN.gnn_score_all as sall
from AttackerGNN.GCNLinkPredictor import GCNLinkPredictor
from AttackerGNN.GCNMarginGradientPredictor import TinyGCN, tanh_margin_loss_label_free
from AttackerGNN.ShadowModelLinkPredictor import LinkPredictionGNN

# from rgnn_at_scale.models import MODEL_TYPE
from rgnn_at_scale.helper import utils
from rgnn_at_scale.attacks.base_attack import Attack, SparseAttack
import os
import csv
from datetime import datetime

from rgnn_at_scale.helper.csvLogger import CSVMetricLogger


class PRBCDExtendedPlotting(SparseAttack):
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

        # --- Pre-GNN definition ---
        # Map original node features to a smaller hidden space
        '''
        self.pre_gnn = PRBCDSelectorGNN(
                    in_channels = self.attr.shape[1],
                    hidden_dim = pre_hidden,
                    num_pairs = 64
                    )
        '''

        self.pre_gnn_prior = PriorSelector(
            in_channels = self.attr.shape[1],
            hidden_dim = pre_hidden,
        )

        self.selector = NodeBlockScorer(
            in_channels=self.attr.shape[1],
            hidden_dim=pre_hidden,  # reuse your arg
        ).to(self.device)

        self._margin_trained = False

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

        self.intra_decay_mode = kwargs.get("intra_decay_mode", "exp")
        self.intra_halflife_epochs = int(max(1, kwargs.get("intra_halflife_epochs", 5)))
        self.initial_max_intra_ratio = 0.2  # start ratio
        self.final_max_intra_ratio = 0.0  # end ratio after all resampling steps

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
        plot_gen = use_cert
        self.use_cert = plot_gen
        self.dataset = kwargs.get('dataset')
        self.seed = kwargs.get('seed')

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

        if plot_gen in ("block_eval_random", "block_eval_selector"):
            if plot_gen == "block_eval_selector":
                print(plot_gen, "-> sampling with accuracy drop selector")

                self.n_candidates_k_sample=2000
                self.n_candidates_one_sample=5000
                self.acc_drop_threshold_k_samples=1e-3
                self.loss_drop_threshold_k_samples=1e-3
                self.k_samples_batch=10
                self.ads_mode="one_sample"
                self.drop_mode="endpoint"

                out_dir="extendedPlotting/gradVsScores/selector"

                if self.drop_mode == "acc": #TODO: logging für alle ads_modes+drop_modes
                    cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.acc_drop_threshold_k_samples}.pt"
                elif self.drop_mode == "loss":
                    cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.loss_drop_threshold_k_samples}.pt"
                elif self.drop_mode == "endpoint":
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
                    ) = PRBCDExtendedPlotting.load_selection(
                        cache_path,
                        device=self.device,
                    )
                else:
                    print("[CACHE] computing selection and saving:", cache_path)
                    y_out, edge_index_lab, y_label, tried_set, harmful_set = self.label_edge_flips_prbcd_selfsample_fast(
                        mode=ads_mode,
                        drop_mode=self.drop_mode,
                        n_candidates_k_sample=self.n_candidates_k_sample,
                        n_candidates_one_sample=self.n_candidates_one_sample,
                        acc_drop_threshold_k_samples=self.acc_drop_threshold_k_samples,
                        loss_drop_threshold_k_samples=self.loss_drop_threshold_k_samples,
                        k_samples_batch=self.k_samples_batch
                    )
                    meta = {
                        "ads_mode": self.ads_mode,
                        "n_candidates_k_sample": self.n_candidates_k_sample,
                        "acc_drop_threshold_k_samples": self.acc_drop_threshold_k_samples,
                        "loss_drop_threshold_k_samples": self.loss_drop_threshold_k_samples,
                    }
                    PRBCDExtendedPlotting.save_selection(cache_path, y_out, edge_index_lab, y_label, tried_set, harmful_set, meta=meta)

                X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)
                stats = PRBCDExtendedPlotting.tried_add_del_proportion(tried_set, edge_index_struct, n=int(self.n))
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
                self.sample_block_from_linkpred_threshold(graph=graph, n_perturbations=n_perturbations)
                self.sample_random_block(n_perturbations)

                selftried_set = tried_set

                with torch.no_grad():

                    logits = self._get_logits(self.attr, self.edge_index, self.edge_weight)
                    loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
                    accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

                    logging.info(f'\nBefore the attack - Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

                    self._append_attack_statistics(loss.item(), accuracy, 0., 0.)

                    del logits, loss

                # ------ Get model eval for block ------

                h = self.lp_model.encoder(X, edge_index_struct)
                logits = self.lp_model.edge_head(h, PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space))
                scores = torch.sigmoid(logits)

                # ------ Get PRBCD Gradient For Block ------

                grad_abs_cum = torch.zeros_like(self.perturbed_edge_weight)

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
                    loss = self.calculate_loss(logits[self.idx_attack], self.labels[
                        self.idx_attack])  # Todo: Hier wird der loss und gradient für perturbed edge weight erzeugt.
                    # Retreive gradient towards the current block (Algorithm 1, line 7)
                    gradient = utils.grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
                    grad_abs_cum += gradient.detach().abs()
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

                        self._append_attack_statistics(loss, accuracy, probability_mass_update,
                                                       probability_mass_projected)

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

                grad_abs_mean = grad_abs_cum / float(self.epochs)
                self.save_block_scores_and_gradients_csv(scores, grad_abs_mean, out_dir)

        elif plot_gen in ("accuracy_rejection_curve",):
            print(plot_gen, "-> sampling with accuracy drop selector")

            self.n_candidates_k_sample = 2000
            self.n_candidates_one_sample = 5000
            self.acc_drop_threshold_k_samples = 1e-3
            self.loss_drop_threshold_k_samples = 1e-3
            self.k_samples_batch = 10
            self.ads_mode = "one_sample"
            self.drop_mode = "endpoint"

            out_dir = "extendedPlotting/gradVsScores/selector"

            if self.drop_mode == "acc":  # TODO: logging für alle ads_modes+drop_modes
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.acc_drop_threshold_k_samples}.pt"
            elif self.drop_mode == "loss":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.loss_drop_threshold_k_samples}.pt"
            elif self.drop_mode == "endpoint":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{self.ads_mode}_k{self.n_candidates_one_sample}_drpmd{self.drop_mode}.pt"

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
                ) = PRBCDExtendedPlotting.load_selection(
                    cache_path,
                    device=self.device,
                )
            else:
                print("[CACHE] computing selection and saving:", cache_path)
                y_out, edge_index_lab, y_label, tried_set, harmful_set = self.label_edge_flips_prbcd_selfsample_fast(
                    mode=self.ads_mode,
                    drop_mode=self.drop_mode,
                    n_candidates_k_sample=self.n_candidates_k_sample,
                    n_candidates_one_sample=self.n_candidates_one_sample,
                    acc_drop_threshold_k_samples=self.acc_drop_threshold_k_samples,
                    loss_drop_threshold_k_samples=self.loss_drop_threshold_k_samples,
                    k_samples_batch=self.k_samples_batch
                )
                meta = {
                    "ads_mode": self.ads_mode,
                    "n_candidates_k_sample": self.n_candidates_k_sample,
                    "acc_drop_threshold_k_samples": self.acc_drop_threshold_k_samples,
                    "loss_drop_threshold_k_samples": self.loss_drop_threshold_k_samples,
                }
                PRBCDExtendedPlotting.save_selection(cache_path, y_out, edge_index_lab, y_label, tried_set,
                                                     harmful_set, meta=meta)

            X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)
            stats = PRBCDExtendedPlotting.tried_add_del_proportion(tried_set, edge_index_struct, n=int(self.n))
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

            self.lp_model = self.lp_model.to(self.device).eval()

            with torch.no_grad():
                h = self.lp_model.encoder(X, edge_index_struct)
                logits = self.lp_model.edge_head(h, edge_index_lab).view(-1)
                scores = torch.sigmoid(logits)

            y_true = y_label.view(-1).long()
            y_pred = (scores >= 0.5).long()

            # confidence for accuracy–rejection
            conf = torch.maximum(scores, 1.0 - scores)

            correct = (y_true == y_pred).long()

            # move to cpu for csv
            conf = conf.cpu().numpy()
            scores = scores.cpu().numpy()
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            correct = correct.cpu().numpy()

            # ---- write csv ----
            out_dir = "extendedPlotting/accuracyRejection"
            os.makedirs(out_dir, exist_ok=True)

            csv_path = os.path.join(
                out_dir,
                f"acc_rejection_dataset{self.dataset}_seed{self.seed}.csv"
            )

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["confidence", "score", "y_true", "y_pred", "correct"])
                for i in range(len(conf)):
                    writer.writerow([
                        float(conf[i]),
                        float(scores[i]),
                        int(y_true[i]),
                        int(y_pred[i]),
                        int(correct[i]),
                    ])

            print("[CSV] saved accuracy–rejection data to:", csv_path)

        elif plot_gen in ("spearman_rho",):
            print(plot_gen, "-> computing spearman rho")

            self.n_candidates_k_sample = 2000
            self.n_candidates_one_sample = 5000
            self.acc_drop_threshold_k_samples = 1e-3
            self.loss_drop_threshold_k_samples = 1e-3
            self.k_samples_batch = 10
            self.ads_mode = "one_sample"
            self.drop_mode = "endpoint"

            if self.drop_mode == "acc":  # TODO: logging für alle ads_modes+drop_modes
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{self.ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.acc_drop_threshold_k_samples}.pt"
            elif self.drop_mode == "loss":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{self.ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.loss_drop_threshold_k_samples}.pt"
            elif self.drop_mode == "endpoint":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{self.ads_mode}_k{self.n_candidates_one_sample}_drpmd{self.drop_mode}.pt"

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
                ) = PRBCDExtendedPlotting.load_selection(
                    cache_path,
                    device=self.device,
                )
            else:
                print("[CACHE] computing selection and saving:", cache_path)
                y_out, edge_index_lab, y_label, tried_set, harmful_set = self.label_edge_flips_prbcd_selfsample_fast(
                    mode=self.ads_mode,
                    drop_mode=self.drop_mode,
                    n_candidates_k_sample=self.n_candidates_k_sample,
                    n_candidates_one_sample=self.n_candidates_one_sample,
                    acc_drop_threshold_k_samples=self.acc_drop_threshold_k_samples,
                    loss_drop_threshold_k_samples=self.loss_drop_threshold_k_samples,
                    k_samples_batch=self.k_samples_batch
                )
                meta = {
                    "ads_mode": self.ads_mode,
                    "n_candidates_k_sample": self.n_candidates_k_sample,
                    "acc_drop_threshold_k_samples": self.acc_drop_threshold_k_samples,
                    "loss_drop_threshold_k_samples": self.loss_drop_threshold_k_samples,
                }
                PRBCDExtendedPlotting.save_selection(cache_path, y_out, edge_index_lab, y_label, tried_set, harmful_set,
                                                     meta=meta)

            X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)
            stats = PRBCDExtendedPlotting.tried_add_del_proportion(tried_set, edge_index_struct, n=int(self.n))
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
            self.sample_block_from_linkpred_threshold(graph=graph, n_perturbations=n_perturbations)
            self.sample_random_block(n_perturbations)

            with torch.no_grad():

                logits = self._get_logits(self.attr, self.edge_index, self.edge_weight)
                loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
                accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

                logging.info(f'\nBefore the attack - Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

                self._append_attack_statistics(loss.item(), accuracy, 0., 0.)

                del logits, loss

            # ------ Get model eval for block ------

            h = self.lp_model.encoder(X, edge_index_struct)
            logits = self.lp_model.edge_head(h, PRBCDExtendedPlotting.linear_to_triu_idx(self.n,
                                                                                         self.current_search_space))
            scores = torch.sigmoid(logits)

            # ------ Get PRBCD Gradient For Block ------

            grad_abs_cum = torch.zeros_like(self.perturbed_edge_weight)

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
                loss = self.calculate_loss(logits[self.idx_attack], self.labels[
                    self.idx_attack])  # Todo: Hier wird der loss und gradient für perturbed edge weight erzeugt.
                # Retreive gradient towards the current block (Algorithm 1, line 7)
                gradient = utils.grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
                grad_abs_cum += gradient.detach().abs()
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

                    self._append_attack_statistics(loss, accuracy, probability_mass_update,
                                                   probability_mass_projected)

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

            grad_abs_mean = grad_abs_cum / float(self.epochs)

            spearman_score = scores.detach().cpu().view(-1).numpy()
            spearman_gradient = grad_abs_mean.detach().cpu().view(-1).numpy()

            rho, p = spearmanr(spearman_score, spearman_gradient)
            print("Spearman rho:", rho, "p:", p)

            self._append_attack_statistics_spearman_rho(rho)

        elif plot_gen in ("jaccard_topk",):
            print(plot_gen, "-> computing jaccard_topk")

            self.n_candidates_k_sample = 2000
            self.n_candidates_one_sample = 5000
            self.acc_drop_threshold_k_samples = 1e-3
            self.loss_drop_threshold_k_samples = 1e-3
            self.k_samples_batch = 10
            self.ads_mode = "one_sample"
            self.drop_mode = "endpoint"

            if self.drop_mode == "acc":  # TODO: logging für alle ads_modes+drop_modes
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{self.ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.acc_drop_threshold_k_samples}.pt"
            elif self.drop_mode == "loss":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{self.ads_mode}_k{self.n_candidates_k_sample}_bt{self.k_samples_batch}_drpmd{self.drop_mode}_drp{self.loss_drop_threshold_k_samples}.pt"
            elif self.drop_mode == "endpoint":
                cache_path = f"cache/selection_dataset{self.dataset}_seed{self.seed}_ads_{self.ads_mode}_k{self.n_candidates_one_sample}_drpmd{self.drop_mode}.pt"

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
                ) = PRBCDExtendedPlotting.load_selection(
                    cache_path,
                    device=self.device,
                )
            else:
                print("[CACHE] computing selection and saving:", cache_path)
                y_out, edge_index_lab, y_label, tried_set, harmful_set = self.label_edge_flips_prbcd_selfsample_fast(
                    mode=self.ads_mode,
                    drop_mode=self.drop_mode,
                    n_candidates_k_sample=self.n_candidates_k_sample,
                    n_candidates_one_sample=self.n_candidates_one_sample,
                    acc_drop_threshold_k_samples=self.acc_drop_threshold_k_samples,
                    loss_drop_threshold_k_samples=self.loss_drop_threshold_k_samples,
                    k_samples_batch=self.k_samples_batch
                )
                meta = {
                    "ads_mode": self.ads_mode,
                    "n_candidates_k_sample": self.n_candidates_k_sample,
                    "acc_drop_threshold_k_samples": self.acc_drop_threshold_k_samples,
                    "loss_drop_threshold_k_samples": self.loss_drop_threshold_k_samples,
                }
                PRBCDExtendedPlotting.save_selection(cache_path, y_out, edge_index_lab, y_label, tried_set, harmful_set,
                                                     meta=meta)

            X, edge_index_struct = self.extract_X_and_edge_index_from_sparsegraph(graph)
            stats = PRBCDExtendedPlotting.tried_add_del_proportion(tried_set, edge_index_struct, n=int(self.n))
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
            self.sample_block_from_linkpred_threshold(graph=graph, n_perturbations=n_perturbations)
            self.sample_random_block(n_perturbations)

            with torch.no_grad():

                logits = self._get_logits(self.attr, self.edge_index, self.edge_weight)
                loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
                accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

                logging.info(f'\nBefore the attack - Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

                self._append_attack_statistics(loss.item(), accuracy, 0., 0.)

                del logits, loss

            # ------ Get model eval for block ------

            h = self.lp_model.encoder(X, edge_index_struct)
            logits = self.lp_model.edge_head(h, PRBCDExtendedPlotting.linear_to_triu_idx(self.n,
                                                                                         self.current_search_space))
            scores = torch.sigmoid(logits)

            # ------ Get PRBCD Gradient For Block ------

            grad_abs_cum = torch.zeros_like(self.perturbed_edge_weight)

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
                loss = self.calculate_loss(logits[self.idx_attack], self.labels[
                    self.idx_attack])  # Todo: Hier wird der loss und gradient für perturbed edge weight erzeugt.
                # Retreive gradient towards the current block (Algorithm 1, line 7)
                gradient = utils.grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
                grad_abs_cum += gradient.detach().abs()
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

                    self._append_attack_statistics(loss, accuracy, probability_mass_update,
                                                   probability_mass_projected)

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

            grad_abs_mean = grad_abs_cum / float(self.epochs)

            jaccard_score = scores.detach().cpu().view(-1).numpy()
            jaccard_gradient = grad_abs_mean.detach().cpu().view(-1).numpy()

            k = 25000  # choose your k

            # Top-k indices (largest values)
            topk_score = np.argsort(jaccard_score)[-k:]
            topk_grad = np.argsort(jaccard_gradient)[-k:]

            set_score = set(topk_score)
            set_grad = set(topk_grad)

            intersection = len(set_score & set_grad)
            union = len(set_score | set_grad)

            jaccard = intersection / union

            print(f"Jaccard@{k}: {jaccard:.4f}")

        elif plot_gen in ("accuracy_drop_selector_subgraph_random", "accuracy_drop_selector_subgraph_khop", "accuracy_drop_selector_subgraph_growhop"):
            print(plot_gen, "-> sampling with accuracy drop selector subgraph")

            self.n_candidates_one_sample = 5000
            self.n_candidates_k_sample = 2000
            self.k_subgraph = 3
            self.subgraph_size = int(self.n/2)
            self.drop_mode = "endpoint"
            self.k_samples_batch = 10
            self.acc_drop_threshold_k_samples = 1e-3
            self.loss_drop_threshold_k_samples = 1e-3

            if plot_gen in ("accuracy_drop_selector_subgraph_random",):
                self.ads_mode = "random_subgraph"
            elif plot_gen in ("accuracy_drop_selector_subgraph_khop",):
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
                ) = PRBCDExtendedPlotting.load_selection(
                    cache_path,
                    device=self.device,
                )
            else:
                print("[CACHE] computing selection and saving:", cache_path)
                if self.ads_mode == "random_subgraph":
                    sub_nodes, edge_index_sub, edge_weight_sub, g2l, l2g = PRBCDExtendedPlotting.extract_random_induced_subgraph(
                        edge_index_struct,
                        self.n,
                        subgraph_size=self.subgraph_size,
                    )
                elif self.ads_mode == "khop_subgraph": #no local values
                    sub_nodes, edge_index_sub, edge_weight_sub, g2l, l2g = PRBCDExtendedPlotting.extract_k_hop_induced_subgraph_of_random_node(
                        edge_index_struct,
                        self.n,
                        k=self.k_subgraph,
                    )
                elif self.ads_mode == "growhop_subgraph":
                    sub_nodes, edge_index_sub, edge_weight_sub, g2l, l2g, hops_used = PRBCDExtendedPlotting.extract_growing_hop_subgraph_of_random_node(
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

                X_sub, edge_index_struct_local = PRBCDExtendedPlotting.build_lp_struct_for_subgraph(
                    X=X,
                    edge_index_struct=edge_index_struct,
                    sub_nodes=sub_nodes,
                    device=self.device,
                )

                meta = {
                    "n_candidates_one_sample": self.n_candidates_one_sample,
                }
                PRBCDExtendedPlotting.save_selection(cache_path, y_out, edge_index_lab, y_label, tried_set, harmful_set, sub_nodes, edge_index_sub, edge_weight_sub, edge_index_lab_local, X_sub, edge_index_struct_local, meta=meta)

            stats = PRBCDExtendedPlotting.tried_add_del_proportion(tried_set, edge_index_struct, n=int(self.n))
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
            print(plot_gen, "run sampling with no certificate")
            self.sample_random_block(n_perturbations)

    def _get_logits(self, x, edge_index, edge_weight):
        return self.attacked_model(
            data=x.to(self.device),
            adj=(edge_index.to(self.device), edge_weight.to(self.device))
        )

    def _get_logits_pre_gnn(self, x, edge_index, edge_weight):
        new_edge_index = self.pre_gnn(x, edge_index, edge_weight)
        return self.attacked_model(
            data=x.to(self.device),
            adj=(new_edge_index.to(self.device),
                 torch.ones(new_edge_index.shape[1], device=self.device))
        )

    def save_block_scores_and_gradients_csv(
            self,
            scores: torch.Tensor,
            gradient: Optional[torch.Tensor] = None,
            base_path: str = "cache/block_eval",
            include_edges: bool = True,
    ) -> str:
        """
        Save a CSV combining block-sized link-pred scores and PRBCD gradients.

        base_path:
            Directory in which the CSV will be written.
            A timestamped filename is automatically appended.

        Returns:
            Full path to the written CSV file.
        """
        if gradient is None:
            if not hasattr(self, "gradient") or self.gradient is None:
                raise ValueError("gradient is None and self.gradient is not set.")
            gradient = self.gradient

        if scores is None:
            raise ValueError("scores must not be None.")

        # flatten + move to CPU
        scores_cpu = scores.detach().view(-1).to("cpu")
        grad_cpu = gradient.detach().view(-1).to("cpu")

        if scores_cpu.numel() != grad_cpu.numel():
            raise ValueError(
                f"scores and gradient must have the same length, "
                f"got {scores_cpu.numel()} vs {grad_cpu.numel()}"
            )

        B = int(scores_cpu.numel())

        # prepare edge metadata if available
        lin_ids = None
        uv = None
        if include_edges:
            if hasattr(self, "current_search_space") and self.current_search_space is not None:
                lin_ids = self.current_search_space.detach().view(-1).to("cpu")

            if hasattr(self, "modified_edge_index") and self.modified_edge_index is not None:
                uv = self.modified_edge_index.detach().to("cpu")
                if uv.dim() != 2 or uv.size(0) != 2:
                    uv = None

        # --- directory + timestamped filename ---
        os.makedirs(base_path, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"block_scores_grads_{timestamp}.csv"
        csv_path = os.path.join(base_path, filename)

        fieldnames = ["idx", "score", "grad", "abs_grad"]
        if include_edges:
            fieldnames = ["idx", "lin_id", "u", "v", "score", "grad", "abs_grad"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(B):
                row = {
                    "idx": i,
                    "score": float(scores_cpu[i].item()),
                    "grad": float(grad_cpu[i].item()),
                    "abs_grad": float(abs(grad_cpu[i].item())),
                }

                if include_edges:
                    row["lin_id"] = int(lin_ids[i].item()) if lin_ids is not None and lin_ids.numel() == B else ""
                    if uv is not None and uv.size(1) == B:
                        row["u"] = int(uv[0, i].item())
                        row["v"] = int(uv[1, i].item())
                    else:
                        row["u"], row["v"] = "", ""

                writer.writerow(row)

        return csv_path

    @staticmethod
    def acc_rejection_curve(y_true, y_pred, conf):
        idx = torch.argsort(conf, descending=True)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

        correct = (y_true == y_pred).float()
        cum_correct = torch.cumsum(correct, dim=0)
        k = torch.arange(1, len(y_true) + 1, device=y_true.device).float()

        acc = cum_correct / k
        cov = k / float(len(y_true))
        return cov.cpu(), acc.cpu()

    @staticmethod
    def jaccard_topk(arr1, arr2, k):
        # get top-k indices (highest values)
        topk1 = np.argsort(arr1)[-k:]
        topk2 = np.argsort(arr2)[-k:]

        set1 = set(topk1)
        set2 = set(topk2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

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

    def sample_random_block(self, n_perturbations: int = 0):
        for _ in range(self.max_final_samples):
            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.block_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(self.n, self.current_search_space)
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
                cand_ei = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, cand_lin)  # (2, M)
            else:
                cand_ei = PRBCDExtendedPlotting.linear_to_full_idx(self.n, cand_lin)  # (2, M)
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
                self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(self.n, self.current_search_space)
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
                cand_ei = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCDExtendedPlotting.linear_to_full_idx(self.n, cand_lin)
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
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
        else:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(self.n, self.current_search_space)
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
                cand_ei_local = PRBCDExtendedPlotting.linear_to_triu_idx(m, cand_lin_local)  # (2, B) local ids
                # convert to GLOBAL lin to optionally exclude tried
                u_l = cand_ei_local[0]
                v_l = cand_ei_local[1]
            else:
                cand_ei_local = PRBCDExtendedPlotting.linear_to_full_idx(m, cand_lin_local)
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
                cand_lin_global = PRBCDExtendedPlotting.triu_idx_to_linear_idx(int(self.n), cand_ei_global_uv).long()
            else:
                cand_ei_global_uv = torch.stack([u_g, v_g], dim=0)
                cand_lin_global = PRBCDExtendedPlotting.full_idx_to_linear_idx(int(self.n), cand_ei_global_uv).long()

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
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(int(self.n), self.current_search_space)
        else:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(int(self.n), self.current_search_space)
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
                cand_ei = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCDExtendedPlotting.linear_to_full_idx(self.n, cand_lin)
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
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
        else:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(self.n, self.current_search_space)
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
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
        else:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(self.n, self.current_search_space)
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
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
        else:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(self.n, self.current_search_space)
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            self.current_search_space = self.current_search_space[is_not_self_loop]
            self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
            self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]
        '''
        # -----------------------------
        # 3) REFILL PHASE: like resample_random_block
        # -----------------------------
        for _ in range(self.max_final_samples):
            n_edges_resample = int(self.block_size) - int(self.current_search_space.size(0))
            if n_edges_resample <= 0:
                break

            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=device)

            # concat old + random, then unique+sorted
            concat_lin = torch.cat((self.current_search_space, lin_index))

            self.current_search_space, unique_idx = torch.unique(
                concat_lin,
                sorted=True,
                return_inverse=True
            )

            # rebuild modified_edge_index from updated search space
            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)

            # merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full(
                (self.current_search_space.size(0),),
                self.eps,
                dtype=torch.float32,
                device=device,
            )
            # the first len(perturbed_edge_weight_old) entries in concat_lin correspond to old weights
            self.perturbed_edge_weight[unique_idx[:perturbed_edge_weight_old.size(0)]] = perturbed_edge_weight_old

            if not self.make_undirected:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

            if self.current_search_space.size(0) > n_perturbations:
                return
            '''

    def sample_block_from_prior_gumbel(self, n_perturbations: int = 0, tau: float = 1.0):
        """
        Build the initial PR-BCD block by sampling 'block_size' distinct edges
        from the selector's learned prior with straight-through Gumbel-Softmax.
        Also record log-probs for policy-gradient updates.
        """
        # sample_block_from_prior_gumbel / resample_block_from_prior_gumbel

        logits = self.pre_gnn_prior(self.attr, self.edge_index, self.edge_weight)  # (E,)
        logits = logits.to(self.device).float()  # <- ensure float dtype

        k = min(self.block_size, logits.numel())
        selected_idx, logps = [], []
        mask = torch.zeros_like(logits)

        for _ in range(k):
            # p = softmax(logits + mask); y ~ ST GumbelSoftmax(p)
            y = F.gumbel_softmax(logits + mask, tau=tau, hard=True, dim=0)  # (E,)
            j = int(y.argmax())
            selected_idx.append(j)

            # log p_j for PG:
            logp = F.log_softmax(logits + mask, dim=0)[j]
            logps.append(logp)

            mask[j] = float('-inf')  # forbid duplicates

        sel = torch.tensor(selected_idx, device=self.device, dtype=torch.long)

        # build PR-BCD search block (indices are into self.edge_index)
        self.current_search_space = sel
        self.modified_edge_index = self.edge_index[:, sel].to(self.device)
        self.perturbed_edge_weight = torch.full(
            (sel.numel(),), self.eps, dtype=torch.float32, requires_grad=True, device=self.device
        )

        # keep the selector logprobs for PG step this epoch
        self._selector_log_probs = torch.stack(logps)  # (k,)

        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError("Not enough edges sampled for the requested budget.")

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

    import torch
    import torch.nn.functional as F

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

    def _pgd_node_labels(self, loss_nodes=None, loss_type="ce", kappa=0.0):
        import torch
        import torch.nn.functional as F

        with torch.enable_grad():
            N = int(self.n)
            X = self.attr.to(self.device).float()
            y = torch.as_tensor(self.labels, device=self.device, dtype=torch.long)

            if loss_nodes is None:
                idx = (torch.as_tensor(self.idx_attack, device=self.device, dtype=torch.long)
                       if getattr(self, "idx_attack", None) is not None
                       else torch.arange(N, device=self.device, dtype=torch.long))
            else:
                idx = torch.as_tensor(loss_nodes, device=self.device, dtype=torch.long)

            # >>> use *current* graph, not the clean one
            ei, ew = self.get_modified_adj()
            ei = ei.to(self.device)
            ew = ew.to(self.device).float()

            g = torch.zeros(N, device=self.device, requires_grad=True)
            Xg = X * (1.0 + g.unsqueeze(1))

            logits = self.attacked_model(data=Xg, adj=(ei, ew))

            if loss_type == "ce":
                loss = F.cross_entropy(logits[idx], y[idx])
            else:
                Z = logits[idx];
                yy = y[idx]
                true = Z[torch.arange(Z.size(0), device=Z.device), yy]
                Zm = Z.clone();
                Zm[torch.arange(Z.size(0), device=Z.device), yy] = -1e9
                other = Zm.max(dim=1).values
                margin = true - other
                if kappa:
                    margin = torch.maximum(margin, torch.tensor(-float(kappa), device=Z.device))
                loss = -margin.mean()

            (grad_g,) = torch.autograd.grad(loss, g, retain_graph=False, create_graph=False)
            s = grad_g.abs()
            return (s - s.mean()) / (s.std() + 1e-6)

    def _pretrain_node_scorer_from_pgd(self, epochs=20, lr=1e-3, wd=5e-4,
                                       loss_nodes=None, loss_type="ce", kappa=0.0, log_every=5):
        """
        Distill PGD-style node saliency into your NodeBlockScorer:
          target: s_i (sum_j |dL/dA_ij|)
          pred:   selector.node_head(selector.embed(...))
        """
        assert hasattr(self, "selector") and (self.selector is not None), \
            "self.selector (NodeBlockScorer) is required."

        self.selector.train()
        X = self.attr.to(self.device).float()
        ei = self.edge_index.to(self.device)
        ew = (self.edge_weight.to(self.device).float() if getattr(self, "edge_weight", None) is not None else None)

        # labels from PGD gradient
        y_node = self._pgd_node_labels(loss_type=loss_type, kappa=kappa)  # (N,)

        opt = torch.optim.Adam(self.selector.parameters(), lr=lr, weight_decay=wd)

        for e in range(1, epochs + 1):
            opt.zero_grad()
            h = self.selector.embed(X, ei, ew)  # (N,d)
            pred = self.selector.node_head(h).squeeze(-1)  # (N,)
            loss = F.smooth_l1_loss(pred, y_node)
            loss.backward()
            opt.step()

            if (e % max(1, log_every) == 0) or e in (1, epochs):
                # you can swap to logging if you prefer
                print(f"[PGD-NodePretrain] epoch {e}/{epochs}  loss={loss.item():.4f}")

        self.selector.eval()
        # flag for downstream checks
        self._pgd_node_trained = True

    def sample_block_from_pgdtopk(self,
                                  k_nodes: int = 350,
                                  prefer_edges: str = "intra",  # 'intra'|'incident'|'mix'
                                  loss_nodes=None,
                                  loss_type="ce",
                                  kappa: float = 0.0,
                                  pretrain_epochs: int = 20,
                                  pretrain_lr: float = 1e-3,
                                  pretrain_wd: float = 5e-4,
                                  max_intra_ratio: float = 0.8):
        """
        1) Pretrains your node scorer on PGD-style node labels.
        2) Extracts Top-K nodes by the trained scorer.
        3) (Optional) Forms a PR-BCD block from those nodes.

        Sets:
          self.current_search_space (linear upper-tri idx)
          self.modified_edge_index  (2, K) upper-tri edges
          self.perturbed_edge_weight (K,)
          self.topk_nodes (for inspection)
        """
        # --- (1) pretrain on PGD node labels ---
        self._pretrain_node_scorer_from_pgd(
            epochs=pretrain_epochs, lr=pretrain_lr, wd=pretrain_wd,
            loss_nodes=loss_nodes, loss_type=loss_type, kappa=kappa
        )

        # --- (2) score nodes and select Top-K ---
        X = self.attr.to(self.device).float()
        ei = self.edge_index.to(self.device)
        ew = (self.edge_weight.to(self.device).float() if getattr(self, "edge_weight", None) is not None else None)

        with torch.no_grad():
            h = self.selector.embed(X, ei, ew)
            node_scores = self.selector.node_head(h).squeeze(-1)
            # z-score (optional)
            node_scores = (node_scores - node_scores.mean()) / (node_scores.std() + 1e-6)
            k = int(min(max(2, k_nodes), self.n))
            topk = torch.topk(node_scores, k=k, largest=True).indices.to(self.device)

        self.topk_nodes = topk

        # --- (3) build a block from Top-K nodes (upper-tri pairs) ---
        import math
        N = int(self.n)
        # candidates within topk
        if topk.numel() >= 2:
            comb = torch.combinations(topk, r=2, with_replacement=False)  # (C(k,2), 2)
            intra_u, intra_v = comb[:, 0], comb[:, 1]
        else:
            intra_u = intra_v = torch.empty(0, dtype=torch.long, device=self.device)

        # incident candidates: connect topk to outside
        all_idx = torch.arange(N, device=self.device)
        is_top = torch.zeros(N, dtype=torch.bool, device=self.device);
        is_top[topk] = True
        outside = all_idx[~is_top]

        # score edges by node scores (sum rule)
        def score_pairs(U, V):
            if U.numel() == 0:
                return torch.empty(0, device=self.device)
            return node_scores[U] + node_scores[V]

        intra_scores = score_pairs(intra_u, intra_v)

        # incident pool (rough sampling to keep it light)
        target_block = int(self.block_size)
        intra_quota = int(round(max_intra_ratio * target_block)) if prefer_edges in ("intra", "mix") else 0
        remaining = max(0, target_block - intra_quota)

        inc_u_list, inc_v_list, inc_s_list = [], [], []
        if prefer_edges in ("incident", "mix") and outside.numel() > 0:
            per_u = max(1, math.ceil(remaining / max(1, topk.numel())))
            for u in topk:
                # sample a small set of partners
                if outside.numel() <= per_u:
                    v = outside
                else:
                    sel = torch.randperm(outside.numel(), device=self.device)[:per_u]
                    v = outside[sel]
                if v.numel() == 0:
                    continue
                inc_u_list.append(u.repeat(v.numel()))
                inc_v_list.append(v)
                inc_s_list.append(score_pairs(u.repeat(v.numel()), v))
        if len(inc_u_list) > 0:
            inc_u = torch.cat(inc_u_list);
            inc_v = torch.cat(inc_v_list);
            inc_scores = torch.cat(inc_s_list)
            # unique
            lin = (torch.minimum(inc_u, inc_v) * N + torch.maximum(inc_u, inc_v))
            uniq_lin, uniq_idx = torch.unique(lin, sorted=False, return_inverse=False, return_counts=False,
                                              return_indices=True)
            inc_u, inc_v, inc_scores = inc_u[uniq_idx], inc_v[uniq_idx], inc_scores[uniq_idx]
        else:
            inc_u = inc_v = torch.empty(0, dtype=torch.long, device=self.device)
            inc_scores = torch.empty(0, device=self.device)

        chosen_u, chosen_v = [], []

        # take intra edges first
        if intra_u.numel() > 0 and intra_quota > 0:
            take = min(intra_quota, intra_u.numel())
            top = torch.topk(intra_scores, k=take, largest=True).indices
            chosen_u.append(intra_u[top]);
            chosen_v.append(intra_v[top])

        # then incident edges to fill the rest
        need = target_block - (0 if len(chosen_u) == 0 else chosen_u[-1].numel())
        if need > 0 and inc_u.numel() > 0:
            take = min(need, inc_u.numel())
            top = torch.topk(inc_scores, k=take, largest=True).indices
            chosen_u.append(inc_u[top]);
            chosen_v.append(inc_v[top])

        if len(chosen_u) == 0:
            # fallback: random upper-tri picks
            tri_u, tri_v = torch.triu_indices(N, N, offset=1, device=self.device)
            perm = torch.randperm(tri_u.numel(), device=self.device)[:target_block]
            final_u, final_v = tri_u[perm], tri_v[perm]
        else:
            final_u = torch.cat(chosen_u);
            final_v = torch.cat(chosen_v)
            # pad if short
            if final_u.numel() < target_block:
                tri_u, tri_v = torch.triu_indices(N, N, offset=1, device=self.device)
                rest = target_block - final_u.numel()
                perm = torch.randperm(tri_u.numel(), device=self.device)[:rest]
                final_u = torch.cat([final_u, tri_u[perm]])
                final_v = torch.cat([final_v, tri_v[perm]])

        # ensure upper-tri (i<j)
        uu = torch.minimum(final_u, final_v)
        vv = torch.maximum(final_u, final_v)
        mask = uu < vv
        uu, vv = uu[mask], vv[mask]

        full_idx = torch.stack([uu, vv], dim=0)  # (2, K')

        # convert to PRBCD's linear indexing
        lin_idx = PRBCDExtendedPlotting.triu_idx_to_linear_idx(self.n, full_idx)

        self.current_search_space = lin_idx
        self.modified_edge_index = full_idx
        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.numel(),),
            self.eps, dtype=torch.float32, device=self.device, requires_grad=True
        )

    def sample_block_from_linkpred_gcn(self, graph):
        import numpy as np
        import torch
        import torch.optim as optim
        from torch.nn import BCEWithLogitsLoss
        from torch_geometric.data import Data

        device = getattr(self, "device", torch.device("cpu"))

        # ---------- Build PyG Data from provided graph (SciPy -> torch) ----------
        # Features (dense float32)
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32)
        X = torch.sparse_coo_tensor(X_idx, X_val, size=graph.attr_matrix.shape).coalesce().to_dense()

        # Adjacency -> undirected edge_index (only once per undirected edge; upper-tri)
        A_coo = graph.adj_matrix.tocoo()
        ei_full = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
        # keep (u<v) to get the undirected set without duplicates/self-loops
        mask_upper = ei_full[0] < ei_full[1]
        ei_upper = ei_full[:, mask_upper]  # (2, E_u)

        # for message passing we want both directions of the TRAIN edges:
        def make_bidir(ei):
            return torch.cat([ei, ei.flip(0)], dim=1)  # (2, 2*E_subset)

        data = Data(x=X, edge_index=None)  # will set train_ei later
        data = data.to(device)
        N = data.num_nodes
        F = data.num_features

        # ---------- Manual split on undirected edges ----------
        E_u = ei_upper.size(1)
        perm = torch.randperm(E_u)
        # 5% val, 10% test (same as before)
        n_val = max(1, int(0.05 * E_u))
        n_test = max(1, int(0.10 * E_u))
        n_train = E_u - n_val - n_test
        idx_train = perm[:n_train]
        idx_val = perm[n_train:n_train + n_val]
        idx_test = perm[n_train + n_val:]

        train_pos_u = ei_upper[:, idx_train]  # (2, Etr_u)
        val_pos_u = ei_upper[:, idx_val]
        test_pos_u = ei_upper[:, idx_test]

        # Edge index used for message passing during training = bidirectional train edges
        train_edge_index = make_bidir(train_pos_u).to(device)

        # ---------- Model / opt ----------
        model = GCNLinkPredictor(in_feats=F, hidden_feats=64, out_feats=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = BCEWithLogitsLoss()

        # ---------- Train link predictor ----------
        from torch_geometric.utils import negative_sampling
        model.train()
        rolling_sum, window = 0.0, 10

        x = data.x.to(device)
        for epoch in range(1, 201):
            # negatives: same count as positive train edges (bidirectional count is double, so use undirected count)
            neg_edge_index = negative_sampling(
                edge_index=train_edge_index,
                num_nodes=N,
                num_neg_samples=train_pos_u.size(1) * 2  # match bidir positives
            )

            pos_edge_index = train_edge_index  # (2, 2*Etr_u)
            combined_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

            pos_labels = torch.ones(pos_edge_index.size(1), device=device, dtype=torch.float32)
            neg_labels = torch.zeros(neg_edge_index.size(1), device=device, dtype=torch.float32)
            combined_labels = torch.cat([pos_labels, neg_labels], dim=0)

            # shuffle
            perm_edges = torch.randperm(combined_edge_index.size(1), device=device)
            combined_edge_index = combined_edge_index[:, perm_edges]
            combined_labels = combined_labels[perm_edges]

            optimizer.zero_grad()
            score_logits = model(x, pos_edge_index, combined_edge_index)
            loss = loss_fn(score_logits.view_as(combined_labels), combined_labels)
            loss.backward()
            optimizer.step()

            rolling_sum += loss.item()
            if epoch % window == 0 or epoch == 1:
                avg = rolling_sum / (window if epoch % window == 0 else 1)
                print(f"Epoch {epoch:03d} | loss: {loss.item():.4f} | avg{window}: {avg:.4f}")
                if epoch % window == 0:
                    rolling_sum = 0.0

        # ---------- Score ALL unordered pairs by entropy ----------
        model.eval()
        with torch.no_grad():
            # all u<v pairs exactly once
            all_pairs = torch.combinations(torch.arange(N, device=device), r=2)  # (M, 2)
            all_edge_index = all_pairs.t().contiguous()  # (2, M)
            edge_logits_all = model(x, train_edge_index, all_edge_index)  # (M,)
            p = torch.sigmoid(edge_logits_all)
            eps = 1e-12
            edge_uncertainty_all = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))  # (M,)

        # ---------- Select top-K & store for PR-BCD ----------
        K = min(50_000, all_edge_index.size(1))
        topk_idx = torch.topk(edge_uncertainty_all, k=K, largest=True).indices

        block_pairs = all_edge_index[:, topk_idx].detach().cpu()  # (2, K)
        block_lin = PRBCDExtendedPlotting.triu_idx_to_linear_idx(N, block_pairs)  # (K,), CPU

        # IMPORTANT: match PR-BCD expectations
        self.current_search_space = block_lin.detach().to(device)  # (K,)
        self.modified_edge_index = block_pairs.detach().to(device)  # (2, K)
        self.perturbed_edge_weight = torch.full(
            (block_lin.numel(),),
            getattr(self, "eps", 1e-7),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        print(f"Built ALL-PAIRS uncertainty block with {block_lin.numel()} edges.")

    def sample_block_from_linkpred_gnn_all_pairs(self, graph, b):

        # Suppose you already have: graph.attr_matrix (N x d, scipy COO/CSR) and graph.adj_matrix (N x N)
        N, d = graph.attr_matrix.shape

        # Convert attributes to torch sparse COO
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32)
        X_sparse = torch.sparse_coo_tensor(X_idx, X_val, size=(N, d)).coalesce()

        # Convert adjacency to torch sparse COO (ensure symmetric for undirected graphs)
        A_coo = graph.adj_matrix.tocoo()
        A_idx = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
        A_val = torch.tensor(A_coo.data, dtype=torch.float32)
        A_sparse = torch.sparse_coo_tensor(A_idx, A_val, size=(N, N)).coalesce()

        # Optional modified edges M as sparse COO (or set to None)

        N, d = X_sparse.size()

        model = sall.AllPairsLinkPredictor(in_dim=d, hidden=64, out_dim=64, dropout=0.1).to(X_sparse.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

        # train a few epochs on ALL pairs
        for ep in range(10):
            opt.zero_grad(set_to_none=True)
            loss = model.loss_all_pairs(X_sparse, A_sparse)
            loss.backward()
            opt.step()
            print(f"epoch {ep + 1:02d} | loss {loss.item():.4f}")

        # inference: top-20 pairs overall
        model.eval()
        pairs, probs = model.bottomk_pairs(X_sparse, A_sparse, k=b)
        print("Top pairs:", pairs[:, :5].T.tolist(), "…")
        print(pairs.shape)
        print("Probs:", probs[:5].tolist(), "…")

        # --- 4) Map to linear index space and trim/unique like sample_random_block ---
        # edges_idx are already u < v (upper-triangle). Convert to linear upper-tri indices.
        lin = self.pairs_to_linear_uppertri(pairs.to(self.device), N)  # (b_sel,)
        # keep unique + sorted (should already be unique, but mirror random_block behavior)
        lin = torch.unique(lin, sorted=True)

        # Cap to self.block_size to honor the contract
        if lin.numel() > self.block_size:
            # because edges came sorted by ascending prob, keep front part
            lin = lin[: self.block_size]

        self.current_search_space = lin  # (m,)

        # --- 5) Set modified_edge_index exactly like sample_random_block ---
        if self.make_undirected:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(N, self.current_search_space)
        else:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(N, self.current_search_space)
            # drop self-loops in the directed/full case
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            if not torch.all(is_not_self_loop):
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        # --- 6) Initialize perturbed_edge_weight like sample_random_block ---
        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.size(0),),
            self.eps, dtype=torch.float32, device=self.device, requires_grad=True
        )

    def sample_block_from_margin_loss_gcn(self, graph, alpha_Sigmoid_APert=0):
        import numpy as np
        import torch

        device = getattr(self, "device", torch.device("cpu"))

        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long, device=device)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32, device=device)
        X_sparse = torch.sparse_coo_tensor(X_idx, X_val, size=graph.attr_matrix.shape, device=device).coalesce()
        X = X_sparse.to_dense()

        A_coo = graph.adj_matrix.tocoo()
        A_idx = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=device)
        A_val = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
        A_sparse = torch.sparse_coo_tensor(A_idx, A_val, size=graph.adj_matrix.shape, device=device).coalesce()
        A_base = A_sparse.to_dense()
        A_base = (A_base + A_base.t()).clamp(max=1.0)

        N, F = X.shape

        num_classes = int(np.max(graph.labels)) + 1

        gcn = TinyGCN(in_feats=F, hidden=64, out_feats=num_classes).to(device)
        gcn.eval()

        iu, ju = torch.triu_indices(N, N, offset=1, device=device)  # all pairs once
        M = iu.numel()

        is_existing = (A_base[iu, ju] > 0)
        sign = torch.where(is_existing,
                           torch.tensor(-1.0, device=device),
                           torch.tensor(+1.0, device=device))  # (M,)

        p = torch.zeros(M, device=device, requires_grad=True)

        if alpha_Sigmoid_APert != 0:
            pert_strength = torch.sigmoid(alpha_Sigmoid_APert * p)  # (M,)
            Pmat = torch.zeros((N, N), device=device, dtype=torch.float32)
            Pmat[iu, ju] = sign * pert_strength
            Pmat[ju, iu] = sign * pert_strength
        else:
            Pmat = torch.zeros((N, N), device=device, dtype=torch.float32)
            Pmat[iu, ju] = sign * p
            Pmat[ju, iu] = sign * p

        A_pert = A_base + Pmat

        logits = gcn(X, A_pert)
        loss = tanh_margin_loss_label_free(logits)
        gcn.zero_grad(set_to_none=True)
        loss.backward()  # populates p.grad

        scores = p.grad.abs().detach()  # (M,)


        K_cap = 50_000
        K = min(getattr(self, "block_size", K_cap), K_cap, M)
        topk = torch.topk(scores, k=K, largest=True)
        chosen = topk.indices  # indices into iu/ju

        block_pairs = torch.stack([iu[chosen].cpu(), ju[chosen].cpu()], dim=0).contiguous()  # (2, K)
        block_lin = PRBCDExtendedPlotting.triu_idx_to_linear_idx(N, block_pairs)  # (K,), CPU

        self.current_search_space = block_lin.detach().to(device)  # (K,)
        self.modified_edge_index = block_pairs.detach().to(device)  # (2, K)
        self.perturbed_edge_weight = torch.full(
            (block_lin.numel(),),
            getattr(self, "eps", 1e-7),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        print(f"Built p-grad block over ALL pairs with {block_lin.numel()} edges (top-{K} by |dL'/dp|).")
        print(loss.item())

    def sample_block_from_linkpred_gnn_old(self, graph, n_perturbations):
        # Suppose you already have: graph.attr_matrix (N x d, scipy COO/CSR) and graph.adj_matrix (N x N)
        N, d = graph.attr_matrix.shape

        # Convert attributes to torch sparse COO
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32)
        X_sparse = torch.sparse_coo_tensor(X_idx, X_val, size=(N, d)).coalesce()

        # Convert adjacency to torch sparse COO (ensure symmetric for undirected graphs)
        A_coo = graph.adj_matrix.tocoo()
        A_idx = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
        A_val = torch.tensor(A_coo.data, dtype=torch.float32)
        A_sparse = torch.sparse_coo_tensor(A_idx, A_val, size=(N, N)).coalesce()

        # Optional modified edges M as sparse COO (or set to None)

        N, d = X_sparse.size()
        model = lle.GNNLeastLikelyEdges(in_dim=d, hidden=64, out_dim=64, dropout=0.1).to(X_sparse.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # Train with BCE (positives vs sampled non-edges)
        for ep in range(1, 71):
            model.train()
            loss = model.bce_loss(X_sparse, A_sparse, M_sparse=None, neg_ratio=1, max_pos_per_batch=64000)
            opt.zero_grad();
            loss.backward();
            opt.step()
            if ep % 10 == 0 or ep == 1:
                print(f"Epoch {ep:03d} | loss={float(loss):.6f}")

            # --- 3) Inference: take least-likely existing edges ---
        model.eval()
        # default b: fill a block or use user-specified
        want = self.block_size
        edges_idx, edge_probs, _ = model(X_sparse, A_sparse, M_sparse=None, b=want)  # edges_idx: (2, b_sel) with u<v

        # --- 4) Map to linear index space and trim/unique like sample_random_block ---
        # edges_idx are already u < v (upper-triangle). Convert to linear upper-tri indices.
        lin = self.pairs_to_linear_uppertri(edges_idx.to(self.device), N)  # (b_sel,)
        # keep unique + sorted (should already be unique, but mirror random_block behavior)
        lin = torch.unique(lin, sorted=True)

        # Cap to self.block_size to honor the contract
        if lin.numel() > self.block_size:
            # because edges came sorted by ascending prob, keep front part
            lin = lin[: self.block_size]

        self.current_search_space = lin  # (m,)

        # --- 5) Set modified_edge_index exactly like sample_random_block ---
        if self.make_undirected:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(N, self.current_search_space)
        else:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(N, self.current_search_space)
            # drop self-loops in the directed/full case
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            if not torch.all(is_not_self_loop):
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        # --- 6) Initialize perturbed_edge_weight like sample_random_block ---
        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.size(0),),
            self.eps, dtype=torch.float32, device=self.device, requires_grad=True
        )

        # --- 7) Budget sanity (mirror sample_random_block) ---
        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError(
                f"GNN block smaller than n_perturbations: "
                f"{self.current_search_space.size(0)} < {n_perturbations}. "
                f"Try increasing b or lowering the budget."
            )

    def sample_block_from_linkpred_gnn(self, graph, n_perturbations):
        import numpy as np
        import torch
        import torch.nn.functional as F

        # --- 1) Convert inputs to torch sparse ---
        N, d = graph.attr_matrix.shape

        # X (features) -> sparse COO
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long, device=self.device)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32, device=self.device)
        X_sparse = torch.sparse_coo_tensor(X_idx, X_val, size=(N, d), device=self.device).coalesce()

        # A (adjacency) -> sparse COO (assume undirected input; if not, symmetrize outside)
        A_coo = graph.adj_matrix.tocoo()
        A_idx = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=self.device)
        A_val = torch.tensor(A_coo.data, dtype=torch.float32, device=self.device)
        A_sparse = torch.sparse_coo_tensor(A_idx, A_val, size=(N, N), device=self.device).coalesce()

        # --- 2) Build & train the LP GNN (same as before) ---
        N, d = X_sparse.size()
        model = lle.GNNLeastLikelyEdges(in_dim=d, hidden=64, out_dim=64, dropout=0.1).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        for ep in range(1, 121):
            model.train()
            loss = model.bce_loss(X_sparse, A_sparse, M_sparse=None, neg_ratio=5, max_pos_per_batch=64000)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ep % 10 == 0 or ep == 1:
                print(f"Epoch {ep:03d} | loss={float(loss):.6f}")

        # --- 3) Inference: use HIGH-LP adds/removes initializer (new logic) ---
        model.eval()
        # total candidates you want in the block
        B = self.block_size

        # NOTE: init_block_highLP_add_remove must be added as a method on the model class (as discussed).
        # It returns (u<v) pairs for both adds (non-edges) and removes (edges).
        edges_idx, probs, is_add_mask = model.init_block_highLP_add_remove(
            X_sparse, A_sparse, M_sparse=None,
            B=B,
            add_ratio=0.5,  # tune split
            p_min=0.6,  # plausibility threshold
            per_node_cap=8,  # diversity
            rank_by="prob",  # or "prob"
            dense_limit=7500,
            block_size=50000
        )  # edges_idx: (2, m), probs: (m,), is_add_mask: (m,)

        # If init pool smaller than B, edges_idx may be shorter. That's fine; PRBCD just gets fewer candidates.

        # --- 4) Map (u,v) to your linear upper-triangle indices (like random) ---
        # edges_idx are already u < v (upper-triangle).
        lin = self.pairs_to_linear_uppertri(edges_idx.to(self.device), N)  # (m,)

        # Keep unique + sorted (mirrors random block behavior)
        lin = torch.unique(lin, sorted=True)

        # Cap to block_size (keep highest-priority first: since we sorted only by index, we reselect by probs if needed)
        if lin.numel() > self.block_size:
            # Re-rank the kept indices by their original priority (probs desc) before truncation:
            # Build a map from linear -> score
            with torch.no_grad():
                # rebuild linear for original ordering
                lin_all = self.pairs_to_linear_uppertri(edges_idx.to(self.device), N)
                # sort by score (desc), then stable-filter to uniques
                order = torch.argsort(probs.to(self.device), descending=True)
                lin_ranked = lin_all[order]
                # keep first occurrences up to block_size
                seen = torch.zeros(lin.max().item() + 1, dtype=torch.bool, device=self.device)
                sel = []
                for i in range(lin_ranked.numel()):
                    li = lin_ranked[i].item()
                    if not seen[li]:
                        sel.append(lin_ranked[i])
                        seen[li] = True
                        if len(sel) >= self.block_size:
                            break
                lin = torch.stack(sel) if sel else lin[: self.block_size]

        self.current_search_space = lin  # (m,)

        # --- 5) Set modified_edge_index like sample_random_block ---
        if self.make_undirected:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(N, self.current_search_space)
        else:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(N, self.current_search_space)
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            if not torch.all(is_not_self_loop):
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        # --- 6) Initialize perturbed_edge_weight like random ---
        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.size(0),),
            self.eps, dtype=torch.float32, device=self.device, requires_grad=True
        )

        # (Optional) If PRBCD needs to know add vs remove for each pair:
        # You could store a boolean mask aligned to current_search_space.
        # Build mask by recomputing linear indices for adds/removes separately and intersecting with lin.
        # Example (cheap and safe even if we truncated/reordered above):
        #   lin_all = self.pairs_to_linear_uppertri(edges_idx.to(self.device), N)
        #   is_add_full = is_add_mask.to(self.device)
        #   add_lin = lin_all[is_add_full]
        #   is_add_flag = torch.isin(self.current_search_space, add_lin)
        #   self.is_add_mask = is_add_flag  # save if your PRBCD uses it

        # --- 7) Budget sanity ---
        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError(
                f"GNN block smaller than n_perturbations: "
                f"{self.current_search_space.size(0)} < {n_perturbations}. "
                f"Try increasing B or lowering the budget."
            )

    def sample_block_from_pgdtopk_direct(
            self,
            k_nodes: int = 350,
            prefer_edges: str = "intra",  # 'intra' | 'incident' | 'mix'
            loss_nodes=None,
            loss_type: str = "ce",
            kappa: float = 0.0,
            max_intra_ratio: float = 0.1,
    ):
        """
        Build PR-BCD block directly from a single PGD-style node saliency pass:
          1) y_node = PGD node importance (N,)
          2) topk = arg top-K y_node
          3) candidate edges = intra(topk) plus incident(topk <-> outside) to fill block_size
          4) finalize PR-BCD block tensors
        """
        import math
        import torch

        dev = self.device
        N = int(self.n)

        # --- (1) PGD node importances (teacher) ---

        y_node = self._pgd_node_labels(loss_nodes=loss_nodes, loss_type=loss_type, kappa=kappa)  # (N,)
            # z-score is already split_by_eps in _pgd_node_labels; if you remove it there, uncomment:
            # y_node = (y_node - y_node.mean()) / (y_node.std() + 1e-6)

        # --- (2) pick Top-K nodes directly from y_node ---
        k = int(min(max(2, k_nodes), N))
        topk = torch.topk(y_node, k=k, largest=True).indices.to(dev)
        self.topk_nodes = topk  # (for inspection)

        # --- helper to convert (u,v) to upper-tri linear index ---
        def pair_to_lin(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            uu = torch.minimum(u, v)
            vv = torch.maximum(u, v)
            mask = uu < vv
            if mask.sum() == 0:
                return torch.empty(0, dtype=torch.long, device=dev)
            uu, vv = uu[mask], vv[mask]
            full_idx = torch.stack([uu, vv], dim=0)
            return PRBCDExtendedPlotting.triu_idx_to_linear_idx(N, full_idx)

        # --- (3) candidate edges: intra-TopK and incident to outside ---
        # Intra-TopK
        if topk.numel() >= 2:
            comb = torch.combinations(topk, r=2, with_replacement=False)  # (C(k,2), 2)
            intra_u, intra_v = comb[:, 0], comb[:, 1]
            intra_scores = y_node[intra_u] + y_node[intra_v]
            intra_lin = pair_to_lin(intra_u, intra_v)
        else:
            intra_u = intra_v = torch.empty(0, dtype=torch.long, device=dev)
            intra_scores = torch.empty(0, device=dev)
            intra_lin = torch.empty(0, dtype=torch.long, device=dev)

        # Incident: connect TopK to outside
        all_idx = torch.arange(N, device=dev)
        is_top = torch.zeros(N, dtype=torch.bool, device=dev)
        is_top[topk] = True
        outside = all_idx[~is_top]

        incident_lin = torch.empty(0, dtype=torch.long, device=dev)
        incident_scores = torch.empty(0, device=dev)
        if outside.numel() > 0 and prefer_edges in ("incident", "mix"):
            target_block = int(self.block_size)
            intra_quota = int(round(max_intra_ratio * target_block)) if prefer_edges in ("intra", "mix") else 0
            remaining = max(0, target_block - intra_quota)
            per_u = max(1, math.ceil(remaining / max(1, topk.numel()))) if remaining > 0 else 0

            inc_u_list, inc_v_list, inc_s_list = [], [], []
            if per_u > 0:
                for u in topk:
                    if outside.numel() <= per_u:
                        v = outside
                    else:
                        sel = torch.randperm(outside.numel(), device=dev)[:per_u]
                        v = outside[sel]
                    if v.numel() == 0:
                        continue
                    inc_u_list.append(u.repeat(v.numel()))
                    inc_v_list.append(v)
                    inc_s_list.append(y_node[u].repeat(v.numel()) + y_node[v])

            if len(inc_u_list) > 0:
                inc_u = torch.cat(inc_u_list)
                inc_v = torch.cat(inc_v_list)
                inc_scores = torch.cat(inc_s_list)

                # unique within incident pool (upper-tri)
                uu = torch.minimum(inc_u, inc_v)
                vv = torch.maximum(inc_u, inc_v)
                lin = (uu * N + vv).to(torch.long)

                # dedupe via sort + mask to get original indices
                lin_sorted, perm = torch.sort(lin)
                keep = torch.ones_like(lin_sorted, dtype=torch.bool, device=lin.device)
                keep[1:] = lin_sorted[1:] != lin_sorted[:-1]
                uniq_idx = perm[keep]

                inc_u, inc_v, inc_scores = inc_u[uniq_idx], inc_v[uniq_idx], inc_scores[uniq_idx]
                incident_scores = inc_scores
                incident_lin = pair_to_lin(inc_u, inc_v)

        # --- (4) assemble the block by scores ---
        chosen = []
        need_total = int(self.block_size)

        # how much intra to take first
        intra_quota = int(round(max_intra_ratio * need_total)) if prefer_edges in ("intra", "mix") else 0

        if intra_lin.numel() > 0 and intra_quota > 0:
            take = min(intra_quota, intra_lin.numel())
            idx = torch.topk(intra_scores, k=take, largest=True).indices
            chosen.append(intra_lin[idx])

        taken = sum(x.numel() for x in chosen) if len(chosen) else 0
        need = max(0, need_total - taken)

        if need > 0 and incident_lin.numel() > 0:
            take = min(need, incident_lin.numel())
            idx = torch.topk(incident_scores, k=take, largest=True).indices
            chosen.append(incident_lin[idx])
            need -= take

        # fallback: random upper-tri edges to pad
        if need > 0:
            n_possible = N * (N - 1) // 2
            chosen.append(torch.randint(n_possible, (need,), device=dev, dtype=torch.long))

        sel_lin = torch.unique(torch.cat(chosen), sorted=True) if len(chosen) else torch.empty(0, dtype=torch.long,
                                                                                               device=dev)

        # --- finalize PR-BCD tensors ---
        self.current_search_space = sel_lin
        self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(N, sel_lin)
        self.perturbed_edge_weight = torch.full(
            (sel_lin.numel(),), self.eps, dtype=torch.float32, device=dev, requires_grad=True
        )

        # (optional) warn if very small
        if self.current_search_space.size(0) < need_total:
            import logging
            logging.warning("[PGD-TopK-Direct] Assembled %d < block_size=%d edges.",
                            self.current_search_space.size(0), need_total)

    def _selector_online_step(self):
        # no-op / lazy init guards (keep yours if present)
        if getattr(self, "selector", None) is None:
            return
        if getattr(self, "selector_opt", None) is None:
            self.selector_opt = torch.optim.Adam(
                self.selector.parameters(), lr=1e-3, weight_decay=5e-4
            )

        # teacher labels don't need grad
        with torch.enable_grad():
            y = self._make_margin_labels().to(self.device)

        x = self.attr.to(self.device).float()
        ei, ew = self.get_modified_adj()
        ei = ei.to(self.device)
        ew = ew.to(self.device).float()

        # <-- turn grads back on just for the selector step
        with torch.enable_grad():
            self.selector.train()
            self.selector_opt.zero_grad(set_to_none=True)

            h = self.selector.embed(x, ei, ew)  # requires grad
            pred = self.selector.node_head(h).squeeze(-1)  # requires grad
            loss = F.smooth_l1_loss(pred, y.detach())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.selector.parameters(), 5.0)
            self.selector_opt.step()
            self.selector.eval()

    def _selector_online_step_pgd(self):
        if getattr(self, "selector", None) is None:
            return
        if getattr(self, "selector_opt", None) is None:
            self.selector_opt = torch.optim.Adam(self.selector.parameters(), lr=1e-3, weight_decay=5e-4)

        # --- teacher: node vulnerability from a single PGD-style backward
        # IMPORTANT: must be OUTSIDE no_grad, since it runs a backward() internally
        with torch.enable_grad():
            y = self._pgd_node_labels(loss_nodes=self.idx_attack, loss_type="ce").detach()  # (N,)

        # --- data on current (attacked) graph
        x = self.attr.to(self.device).float()
        ei, ew = self.get_modified_adj()
        ei = ei.to(self.device)
        ew = ew.to(self.device).float()

        # --- one training step for the selector
        with torch.enable_grad():
            self.selector.train()
            self.selector_opt.zero_grad(set_to_none=True)
            h = self.selector.embed(x, ei, ew)  # (N, d)
            pred = self.selector.node_head(h).squeeze(-1)  # (N,)
            loss = F.smooth_l1_loss(pred, y)  # teacher already z-scored
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.selector.parameters(), 5.0)
            self.selector_opt.step()
            self.selector.eval()

    def sample_block_from_selector(
            self,
            n_perturbations: int = 0,
            k_nodes: int = None,
            prioritize_intra: bool = True,
            ensure_pretrain: bool = True,
            pretrain_epochs: int = 20,
            pretrain_lr: float = 1e-3,
            pretrain_wd: float = 5e-4,
            # blend control
            alpha_saliency: float = 0.1,  # node_score = α·sal + (1-α)·pred
            saliency_edges: bool = True,  # True => edge probs from saliency only
            gumbel_nodes: bool = True,  # sample top-k nodes via Gumbel-softmax
            gumbel_edges: bool = True,  # sample edges via Gumbel-softmax
            tau_nodes: float = 1.0,  # temperature for node sampling
            tau_edges: float = 1.0,  # temperature for edge sampling
            max_intra_ratio: float = 0.1,  # quota for intra-topk edges
            log_examples: int = 10,
    ):
        """
        Build a PR-BCD block over ALL undirected pairs (i<j), using stochastic
        Gumbel-Softmax sampling:
          • Nodes: sample k via Gumbel-top-k over softmax(node_score / τ_nodes)
          • Edges: sample via Gumbel-top-k over softmax(edge_score / τ_edges)

        Edge scores can be taken from *pure saliency* (default) or the blended node score.
        """
        import math
        import torch

        dev, N = self.device, self.n

        # 0) Availability Check
        if ensure_pretrain and not getattr(self, "_margin_trained", False):
            logging.info("[Selector] Pretraining NodeBlockScorer...")
            self._pretrain_margin_scorer(epochs=pretrain_epochs, lr=pretrain_lr, wd=pretrain_wd)
        else:
            logging.info("[Selector] Using NodeBlockScorer (pretrained=%s).",
                         getattr(self, "_margin_trained", False))

        sal = self._node_saliency_from_victim(subset="attack").to(dev)
        sal = (sal - sal.mean()) / (sal.std() + 1e-6)

        self.selector.eval()
        with torch.no_grad():
            x = self.attr.to(dev).float()
            ei = self.edge_index.to(dev)
            ew = (self.edge_weight.to(dev).float()
                  if self.edge_weight is not None
                  else torch.ones(ei.size(1), device=dev))
            h = self.selector.embed(x, ei, ew)
            pred = self.selector.node_head(h).squeeze(-1)
            pred = (pred - pred.mean()) / (pred.std() + 1e-6)

        # 2) Blended node score for *ranking/probabilities*
        alpha = float(alpha_saliency)
        node_score = alpha * sal + (1.0 - alpha) * pred

        # Helper: mapped (u,v) -> linear upper-tri index
        def pair_to_lin(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            uu = torch.minimum(u, v)
            vv = torch.maximum(u, v)
            mask = (uu < vv)
            if mask.sum() == 0:
                return torch.empty(0, dtype=torch.long, device=dev)
            uu, vv = uu[mask], vv[mask]
            return PRBCDExtendedPlotting.triu_idx_to_linear_idx(N, torch.stack([uu, vv], dim=0))

        # 4) Pick k nodes — either greedy top‑k or Gumbel‑top‑k over softmax(node_score/τ)
        if k_nodes is None:
            k_nodes = int(math.ceil((1.0 + math.sqrt(1.0 + 8.0 * self.block_size)) / 2.0))
        k = int(min(k_nodes, N))

        if gumbel_nodes:
            node_logits = torch.log_softmax(node_score / max(1e-6, float(tau_nodes)), dim=0)
            topk_nodes = PRBCDExtendedPlotting._gumbel_topk(self, node_logits, k).to(dev)
        else:
            topk_nodes = torch.topk(node_score, k=k, largest=True).indices.to(dev)

        if log_examples > 0:
            logging.info("[Selector] k=%d | example top nodes: %s", k, topk_nodes[:log_examples].tolist())

        # 5) Build intra-topk candidate edges + scores
        if topk_nodes.numel() >= 2:
            comb = torch.combinations(topk_nodes, r=2, with_replacement=False)  # (C(k,2), 2)
            intra_u, intra_v = comb[:, 0], comb[:, 1]
            intra_lin = pair_to_lin(intra_u, intra_v)
        else:
            intra_u = intra_v = torch.empty(0, dtype=torch.long, device=dev)
            intra_lin = torch.empty(0, dtype=torch.long, device=dev)

        # Edge scores: by default use *pure saliency* (requested), or blended
        node_for_edges = sal if bool(saliency_edges) else node_score
        intra_scores = (node_for_edges[intra_u] + node_for_edges[intra_v]) if intra_lin.numel() > 0 \
            else torch.empty(0, device=dev)

        # 6) Incident candidates (topk ↔ outside) — sample partners by node probabilities
        all_idx = torch.arange(N, device=dev)
        is_top = torch.zeros(N, dtype=torch.bool, device=dev)
        is_top[topk_nodes] = True
        outside = all_idx[~is_top]

        incident_lin_list, incident_scores_list = [], []
        need_total = int(self.block_size)
        intra_quota = int(round(max_intra_ratio * need_total)) if prioritize_intra else 0
        # aim to fill remaining with incident; split roughly across top nodes
        remaining_need = max(0, need_total - (intra_lin.numel() if intra_quota > 0 else 0))
        per_top_quota = max(1, math.ceil(remaining_need / max(1, topk_nodes.numel()))) if remaining_need > 0 else 0

        if outside.numel() > 0 and per_top_quota > 0:
            outside_logits = torch.log_softmax(node_for_edges[outside] / max(1e-6, float(tau_edges)), dim=0)
            for u in topk_nodes:
                if per_top_quota <= 0:
                    break
                # sample partners for u (with replacement) by Gumbel-top-k over outside logits
                # (equivalent to drawing the highest Gumbel-perturbed log-probs)
                v_sel = self._gumbel_topk(outside_logits, per_top_quota)
                v = outside[v_sel]
                u_rep = u.repeat(v.numel())
                lin = pair_to_lin(u_rep, v)
                if lin.numel() == 0:
                    continue
                incident_lin_list.append(lin)
                incident_scores_list.append(node_for_edges[u_rep] + node_for_edges[v])

        if len(incident_lin_list) > 0:
            incident_lin = torch.cat(incident_lin_list, dim=0)
            incident_scores = torch.cat(incident_scores_list, dim=0)
            # unique within incident pool
            uniq_lin, uniq_idx = torch.unique(incident_lin, sorted=False, return_inverse=False, return_counts=False,
                                              return_indices=True)
            incident_lin = uniq_lin
            incident_scores = incident_scores[uniq_idx]
        else:
            incident_lin = torch.empty(0, dtype=torch.long, device=dev)
            incident_scores = torch.empty(0, device=dev)

        # 7) Assemble block: sample edges via Gumbel-top-k (or greedy if gumbel_edges=False)
        def pick_edges(idx_lin: torch.Tensor, scores: torch.Tensor, m: int) -> torch.Tensor:
            if idx_lin.numel() == 0 or m <= 0:
                return torch.empty(0, dtype=torch.long, device=dev)
            m = min(m, idx_lin.numel())
            if gumbel_edges:
                logits = torch.log_softmax(scores / max(1e-6, float(tau_edges)), dim=0)
                sel = self._gumbel_topk(logits, m)
            else:
                sel = torch.topk(scores, k=m, largest=True).indices
            return idx_lin[sel]

        chosen, need = [], need_total

        if prioritize_intra and intra_lin.numel() > 0:
            take_intra = pick_edges(intra_lin, intra_scores, min(intra_quota, need))
            chosen.append(take_intra)
            need -= take_intra.numel()

        if need > 0 and incident_lin.numel() > 0:
            take_inc = pick_edges(incident_lin, incident_scores, need)
            chosen.append(take_inc)
            need -= take_inc.numel()

        # fallback: random from the whole upper-tri space to keep exploration
        if need > 0:
            n_possible = N * (N - 1) // 2
            chosen.append(torch.randint(n_possible, (need,), device=dev, dtype=torch.long))
            need = 0

        sel_lin = torch.unique(torch.cat(chosen), sorted=True) if len(chosen) else torch.empty(0, dtype=torch.long,
                                                                                               device=dev)
        if sel_lin.numel() < self.block_size:
            logging.warning("[Selector] Only %d candidates assembled (requested %d).", sel_lin.numel(), self.block_size)

        # 8) Finalize PR-BCD block tensors
        self.current_search_space = sel_lin
        self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(N, sel_lin)
        self.perturbed_edge_weight = torch.full((sel_lin.numel(),), self.eps, dtype=torch.float32, device=dev,
                                                requires_grad=True)

        # Budget sanity
        if self.current_search_space.size(0) < n_perturbations:
            logging.warning("[Selector] Block smaller than n_perturbations: %d < %d.",
                            self.current_search_space.size(0), n_perturbations)

    def _current_node_scores(self, blend_alpha: float = 0.0) -> torch.Tensor:
        """
        Compute node scores used for guided sampling.
        blend_alpha=0 -> pure selector prediction (fast);
        blend_alpha>0 -> blend with saliency for stability: alpha*sal + (1-alpha)*pred
        """
        x = self.attr.to(self.device).float()
        ei, ew = self.get_modified_adj()
        ei = ei.to(self.device)
        ew = ew.to(self.device).float()

        with torch.no_grad():
            h = self.selector.embed(x, ei, ew)
            pred = self.selector.node_head(h).squeeze(-1)
            pred = (pred - pred.mean()) / (pred.std() + 1e-6)

        if blend_alpha <= 1e-9:
            return pred

        with torch.enable_grad():
            sal = self._node_saliency_from_victim(subset="attack").to(self.device)
            sal = (sal - sal.mean()) / (sal.std() + 1e-6)
        return float(blend_alpha) * sal + (1.0 - float(blend_alpha)) * pred

    def _assemble_edges_from_nodes(
            self,
            node_scores: torch.Tensor,
            need: int,
            prefer: str = "mix",
            max_intra_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Build up to `need` new upper-tri *linear* edge indices guided by node_scores.
        prefer: 'intra' | 'incident' | 'mix'
        """
        import math
        dev = self.device
        N = int(self.n)

        # ---- choose k so C(k,2) isn't trivially < need ----
        k_nodes = int(min(N, max(2, math.ceil(0.5 * (1 + math.sqrt(1 + 8 * need))))))  # ~inverse of C(k,2)
        topk = torch.topk(node_scores, k=k_nodes, largest=True).indices.to(dev)

        # ---- intra-topk candidates ----
        if topk.numel() >= 2:
            comb = torch.combinations(topk, r=2, with_replacement=False)  # (C(k,2), 2)
            intra_u, intra_v = comb[:, 0], comb[:, 1]
            uu = torch.minimum(intra_u, intra_v)
            vv = torch.maximum(intra_u, intra_v)
            intra_lin = PRBCDExtendedPlotting.triu_idx_to_linear_idx(N, torch.stack([uu, vv], dim=0))
            intra_scores = node_scores[intra_u] + node_scores[intra_v]
        else:
            intra_lin = torch.empty(0, dtype=torch.long, device=dev)
            intra_scores = torch.empty(0, device=dev)

        # ---- incident candidates (topk ↔ outside) ----
        all_idx = torch.arange(N, device=dev)
        is_top = torch.zeros(N, dtype=torch.bool, device=dev)
        is_top[topk] = True
        outside = all_idx[~is_top]

        inc_lin = torch.empty(0, dtype=torch.long, device=dev)
        inc_scores = torch.empty(0, device=dev)

        if outside.numel() > 0 and prefer in ("incident", "mix"):
            per_u = max(1, math.ceil(need / max(1, topk.numel())))
            us, vs, ss = [], [], []
            for u in topk:
                if outside.numel() <= per_u:
                    v = outside
                else:
                    sel = torch.randperm(outside.numel(), device=dev)[:per_u]
                    v = outside[sel]
                if v.numel() == 0:
                    continue
                uu = torch.minimum(u.repeat(v.numel()), v)
                vv = torch.maximum(u.repeat(v.numel()), v)
                us.append(uu);
                vs.append(vv)
                ss.append(node_scores[u].repeat(v.numel()) + node_scores[v])

            if len(us) > 0:
                uu = torch.cat(us);
                vv = torch.cat(vs);
                ss = torch.cat(ss)

                # ---- dedupe (uu,vv): use sort+mask (works on all PyTorch) ----
                lin_raw = uu * N + vv  # linearize pairs (upper-tri since uu<=vv)
                lin_sorted, perm = torch.sort(lin_raw)  # sort to detect dups
                keep = torch.ones_like(lin_sorted, dtype=torch.bool, device=dev)
                keep[1:] = lin_sorted[1:] != lin_sorted[:-1]
                uniq_idx = perm[keep]  # indices into uu/vv/ss to keep

                uu = uu[uniq_idx];
                vv = vv[uniq_idx];
                ss = ss[uniq_idx]
                inc_lin = PRBCDExtendedPlotting.triu_idx_to_linear_idx(N, torch.stack([uu, vv], dim=0))
                inc_scores = ss

        # ---- assemble by quota ----
        chosen = []
        quota_intra = int(round(max_intra_ratio * need)) if prefer in ("intra", "mix") else 0

        if intra_lin.numel() > 0 and quota_intra > 0:
            take = min(quota_intra, intra_lin.numel())
            idx = torch.topk(intra_scores, k=take, largest=True).indices
            chosen.append(intra_lin[idx])

        taken = sum(x.numel() for x in chosen) if chosen else 0
        need_rest = max(0, need - taken)

        if need_rest > 0 and inc_lin.numel() > 0:
            take = min(need_rest, inc_lin.numel())
            idx = torch.topk(inc_scores, k=take, largest=True).indices
            chosen.append(inc_lin[idx])
            need_rest -= take

        if need_rest > 0:
            # pad with random upper-tri edges
            n_possible = N * (N - 1) // 2
            chosen.append(torch.randint(n_possible, (need_rest,), device=dev, dtype=torch.long))

        # final linear indices (unique to be safe against intra∩incident overlap)
        return torch.unique(torch.cat(chosen), sorted=True) if chosen else torch.empty(0, dtype=torch.long, device=dev)

    def resample_block_from_selector(
            self,
            n_perturbations: int,
            blend_alpha: float = 1.0,  # <-- default to using saliency (α=1.0)
            prefer_edges: str = "mix",
            max_intra_ratio: float = 0.5,
    ):
        if self.keep_heuristic != "WeightOnly":
            raise NotImplementedError("Only keep_heuristic=`WeightOnly` supported")

        # --- keep phase (same as before) ---
        sorted_idx = torch.argsort(self.perturbed_edge_weight)  # ascending
        idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
        if idx_keep < sorted_idx.size(0) // 2:
            idx_keep = sorted_idx.size(0) // 2
        keep_idx = sorted_idx[idx_keep:]

        kept_lin = self.current_search_space[keep_idx].to(self.device)
        kept_w = self.perturbed_edge_weight[keep_idx].to(self.device)
        kept_pairs = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, kept_lin) if self.make_undirected \
            else PRBCDExtendedPlotting.linear_to_full_idx(self.n, kept_lin)

        # --- selector update (PGD teacher) ---
        self._selector_online_step_pgd()

        # --- refill guided by (α·saliency + (1−α)·selector) node scores ---
        need = int(self.block_size) - int(kept_lin.numel())
        if need <= 0:
            self.current_search_space = kept_lin
            self.modified_edge_index = kept_pairs
            self.perturbed_edge_weight = kept_w
            return


        node_scores = self._current_node_scores(blend_alpha=blend_alpha)  # α>0 uses saliency
        add_lin = self._assemble_edges_from_nodes(
            node_scores, need, prefer=prefer_edges, max_intra_ratio=max_intra_ratio
        )
        with torch.no_grad():
            if add_lin.numel() > 0:
                merged = torch.cat([kept_lin, add_lin.to(self.device)], dim=0)
                all_lin, inv = torch.unique(merged, sorted=True, return_inverse=True)
                pos_kept = inv[: kept_lin.numel()]
            else:
                all_lin = kept_lin
                pos_kept = torch.arange(kept_lin.numel(), device=self.device, dtype=torch.long)

            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, all_lin) if self.make_undirected \
                else PRBCDExtendedPlotting.linear_to_full_idx(self.n, all_lin)

            new_w = torch.full((all_lin.numel(),), self.eps, dtype=torch.float32, device=self.device)
            new_w[pos_kept] = kept_w
            self.perturbed_edge_weight = new_w
            self.current_search_space = all_lin

            if not self.make_undirected:
                is_not_self = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self]

        if self.current_search_space.size(0) <= n_perturbations:
            logging.warning("[SelectorResample] Block size %d ≤ n_perturbations %d; consider increasing block_size.",
                            self.current_search_space.size(0), n_perturbations)

    def resample_block_from_prior_gumbel(self, n_perturbations: int, tau: float = 1.0):
        """
        Keep at most half of the current block (largest weights),
        then refill from the selector prior via Gumbel-Softmax + masking.
        """
        # --- keep heuristic (same semantics as your resample_random_block) ---
        if self.keep_heuristic != 'WeightOnly':
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = torch.argsort(self.perturbed_edge_weight)  # ascending
        idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
        if idx_keep < sorted_idx.size(0) // 2:
            idx_keep = sorted_idx.size(0) // 2
        keep_idx = sorted_idx[idx_keep:]  # keep largest half

        # slice current block
        self.current_search_space = self.current_search_space[keep_idx]
        self.modified_edge_index = self.modified_edge_index[:, keep_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[keep_idx]

        # --- refill from prior ---
        need = self.block_size - self.current_search_space.numel()
        if need <= 0:
            # nothing to refill
            self._selector_log_probs = None
            return

        # sample_block_from_prior_gumbel / resample_block_from_prior_gumbel

        logits = self.pre_gnn_prior(self.attr, self.edge_index, self.edge_weight)  # (E,)
        logits = logits.to(self.device).float()  # <- ensure float dtype

        # mask already-selected global edges
        mask = torch.zeros_like(logits)
        mask[self.current_search_space] = float('-inf')

        picked, logps = [], []
        for _ in range(need):
            y = F.gumbel_softmax(logits + mask, tau=tau, hard=True, dim=0)  # (E,)
            j = int(y.argmax())
            picked.append(j)

            logp = F.log_softmax(logits + mask, dim=0)[j]
            logps.append(logp)

            mask[j] = float('-inf')

        add = torch.tensor(picked, device=self.device, dtype=torch.long)

        # extend the block; no duplicates because of mask
        self.current_search_space = torch.cat([self.current_search_space, add], dim=0)
        self.modified_edge_index = self.edge_index[:, self.current_search_space]
        self.perturbed_edge_weight = torch.cat([
            self.perturbed_edge_weight,
            torch.full((add.numel(),), self.eps, dtype=torch.float32, device=self.device)
        ], dim=0)

        # store log-probs for PG update
        self._selector_log_probs = torch.stack(logps)

    def resample_random_block(self, n_perturbations: int): #TODO: still work to be split_by_eps
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
            n_edges_resample = self.block_size - self.current_search_space.size(0)
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(self.n, self.current_search_space)

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
                return PRBCDExtendedPlotting.linear_to_triu_idx(self.n, cand_lin)
            ei = PRBCDExtendedPlotting.linear_to_full_idx(self.n, cand_lin)
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
                cand_ei = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCDExtendedPlotting.linear_to_full_idx(self.n, cand_lin)
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
                self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(self.n, self.current_search_space)

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
            tau: float = 0.8,
            max_sampling_tries: int = 2_000_000,
            score_batch_size: int = 1000,
            rng_seed: int = 0,
            exclude_tried: bool = True,
    ):
        """
        Drop-in replacement that encodes the link-pred GNN on the *current perturbed graph*
        (via self.get_modified_adj()) instead of the unperturbed input `graph`.

        Notes:
          - We binarize the current attacked adjacency for the LP encoder by default:
                keep edges with edge_weight > 0.5
            (Adjust the threshold below if your LP model expects something else.)
          - Everything else follows your original logic.
        """
        import torch

        if not hasattr(self, "lp_model") or self.lp_model is None:
            raise RuntimeError("self.lp_model is not set.")

        # -----------------------------
        # tried_mask init (same intent)
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

            # Binarize attacked adjacency for message passing
            present = edge_weight_mod > 0.5
            edge_index_struct = edge_index_mod[:, present]

        # encode once
        self.lp_model = self.lp_model.to(self.device).eval()
        with torch.no_grad():
            h = self.lp_model.encoder(X, edge_index_struct)

        g = torch.Generator(device=self.device)
        g.manual_seed(int(rng_seed))

        # masks for uniqueness within the block
        block_mask = torch.zeros(self.n_possible_edges, device=self.device, dtype=torch.bool)
        block_mask[self.current_search_space] = True

        accepted = []
        tries = 0
        n_needed = int(self.block_size - self.current_search_space.numel())

        while len(accepted) < n_needed and tries < int(max_sampling_tries):
            tries += 1

            cand_lin = torch.randint(
                self.n_possible_edges, (int(score_batch_size),), device=self.device, generator=g
            )
            cand_lin = torch.unique(cand_lin, sorted=False)

            # exclude already in block
            cand_lin = cand_lin[~block_mask[cand_lin]]

            # optional exclude tried
            if exclude_tried:
                cand_lin = cand_lin[~self.tried_mask[cand_lin]]

            if cand_lin.numel() == 0:
                continue

            # decode to pairs
            if self.make_undirected:
                cand_ei = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, cand_lin)
            else:
                cand_ei = PRBCDExtendedPlotting.linear_to_full_idx(self.n, cand_lin)
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
            block_mask[cand_keep] = True

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
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
        else:
            self.modified_edge_index = PRBCDExtendedPlotting.linear_to_full_idx(self.n, self.current_search_space)

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
                return int(PRBCDExtendedPlotting.triu_idx_to_linear_idx(self.n, full).item())
            else:
                full = torch.tensor([[u], [v]], dtype=torch.long)
                return int(PRBCDExtendedPlotting.full_to_linear_idx(self.n, full).item())

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
            n_candidates_one_sample: int = 2000,
            n_candidates_k_sample: int = 2000,
            p_add: float = 0.06,  # unused in uniform flip
            p_del: float = 0.1,  # unused in uniform flip
            acc_drop_threshold_one_sample: float = 3e-3,
            acc_drop_threshold_k_samples: float = -1,
            acc_drop_threshold_k_hop: float = 1e-2,
            drop_mode: str = "acc",  # "acc" | "loss" | "endpoint"
            loss_drop_threshold_one_sample: float = 1e-3,
            loss_drop_threshold_k_samples: float = 1e-3,
            loss_drop_threshold_k_hop: float = 1e-3,
            rng_seed: int = 0,
            max_sampling_tries: int = 100_000,
            mode: str = "k_hop",  # # one_sample | k_action | k_action_individual | k_hop
            k_samples_batch: int = 10,  # only used in k_action
            n_candidates_k_hop_sample: int = 2000,  # target # of harmful flips in k_hop mode
            k_hop: int = 2,  # hop radius for k_hop mode
            prev_tried_set: "Set[tuple[int, float]] | None" = None,
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

        # ---- base adjacency
        ei_base = self.edge_index.to(device=device, dtype=torch.long).contiguous()  # (2,E)
        ew_base = (self.edge_weight.to(device).float()
                   if getattr(self, "edge_weight", None) is not None
                   else torch.ones(ei_base.size(1), device=device, dtype=torch.float32)).contiguous()

        # ---- build undirected membership bitset
        present = PRBCDExtendedPlotting._build_uppertri_bitset(ei_base, n)  # (num_pairs,) bool

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
        elif drop_mode == "endpoint":
            thr_one = None
            thr_k = None
            thr_khop = None
        else:
            raise ValueError(f"Unknown drop_mode='{drop_mode}'. Use 'acc' or 'loss'.")

        # ---- outputs over *all* upper-tri pairs
        num_pairs = n * (n - 1) // 2
        y_out = torch.zeros(num_pairs, dtype=torch.uint8, device=device)

        # ---- lists to collect labeled pairs (only tried pairs)
        lab_u: list[int] = []
        lab_v: list[int] = []
        lab_k: list[int] = []  # linear indices k_lin for those pairs

        # start tried_set from previous run if given
        tried_set: Set[Tuple[int, float]] = set(prev_tried_set) if prev_tried_set is not None else set()
        harmful_set: Set[Tuple[int, float]] = set()

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

        # ----- helper: compute drop for purturbation selection -----
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

        # ----- helper: compute drop for purturbation selection -----
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
            elif drop_mode == "endpoint":
                pass
            else:
                raise ValueError(f"Unknown drop_mode='{drop_mode}'. Use 'acc' or 'loss'.")

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

        # ----- helper: builds n/2 flips -----
        def _sample_node_matching_batch():
            """
            Returns a batch of unique undirected pairs covering each node at most once.
            Size ~ floor(n/2). Skips already-seen k_lin and re-draws a few times if needed.
            """
            # target unique undirected edges this batch
            target = n // 2
            batch = []
            local_tries = 0
            max_local_tries = 50  # cheap; each try is just a randperm

            while len(batch) < target and local_tries < max_local_tries:
                perm = torch.randperm(n, generator=g, device=device)
                # pair consecutive nodes
                u_all = perm[0:2 * target:2]
                v_all = perm[1:2 * target:2]

                # enforce u < v for upper-tri indexing
                uu = torch.minimum(u_all, v_all)
                vv = torch.maximum(u_all, v_all)

                # compute linear indices (upper-tri)
                # full_idx shape (2, m)
                full_idx = torch.stack([uu, vv], dim=0)
                k_all = PRBCDExtendedPlotting.triu_idx_to_linear_idx(n, full_idx).long()  # (m,)

                # filter: not already seen
                not_seen = ~seen[k_all]
                uu = uu[not_seen]
                vv = vv[not_seen]
                k_all = k_all[not_seen]

                if k_all.numel() == 0:
                    local_tries += 1
                    continue

                # action from membership bitset
                # present[k] == True => edge exists => "del", else "add"
                for u, v, k_lin in zip(uu.tolist(), vv.tolist(), k_all.tolist()):
                    action = "del" if bool(present[k_lin].item()) else "add"
                    batch.append((int(u), int(v), action, int(k_lin)))
                    if len(batch) >= target:
                        break

                local_tries += 1

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

        # -------- switch: sampling mode --------
        if mode == "one_sample":
            while (flips_done == 0 or flips_done < n_candidates_one_sample) and tries < max_sampling_tries:
                u, v, action, k_lin = PRBCDExtendedPlotting._sample_one_flip(n, present, g, device)
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
                else:
                    if _endpoint_flipped_correct_to_incorrect(u, v, logits_pert):
                        y_out[k_lin] = 1
                        harmful_set.add((int(k_lin), drop))
                        flips_done += 1

        elif mode == "node_matching":
            if drop_mode != "endpoint":
                raise ValueError("mode='node_matching' is intended for drop_mode='endpoint' (batched endpoint check).")

            # collect harmful edges until target reached
            while (flips_done == 0 or flips_done < n_candidates_k_hop_sample) and tries < max_sampling_tries:
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
                    # store tried (drop is None in endpoint mode)
                    tried_set.add((int(k_lin), None))
                    lab_u.append(int(u))
                    lab_v.append(int(v))
                    lab_k.append(int(k_lin))

                    if _endpoint_flipped_correct_to_incorrect(u, v, logits_pert):
                        y_out[k_lin] = 1
                        harmful_set.add((int(k_lin), None))
                        flips_done += 1

                # optional: stop early if you already have enough harmful flips
                if flips_done >= n_candidates_one_sample:
                    break

        elif mode == "k_action":
            while (flips_done == 0 or flips_done < n_candidates_k_sample) and tries < max_sampling_tries:
                batch = PRBCDExtendedPlotting._sample_k_flips(
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
                batch = PRBCDExtendedPlotting._sample_k_flips(
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
                    if drop_ind > thr_k/k_samples_batch: #TODO: hier individual drop festlegen
                        y_out[k_lin] = 1
                        harmful_set.add((int(k_lin), drop_ind))
                        flips_done += 1

        elif mode == "k_hop":
            if adj_list is None:
                raise RuntimeError("adj_list must be built for k_hop mode.")

            max_root_tries = 100_000

            while (flips_done == 0 or flips_done < n_candidates_k_hop_sample) and tries < max_sampling_tries:
                batch = PRBCDExtendedPlotting._sample_khop_flips(
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
                harmful_set = {(idx, drop) for (idx, drop) in harmful_set if idx in keep_set}
                flips_done = n_candidates_k_hop_sample

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'one_sample', 'k_action' or 'k_hop'.")

        # ---- build edge_index_lab and y_label ----
        # create mapping from k_lin -> (u, v) for all actually evaluated pairs
        k_to_uv = {k: (u, v) for u, v, k in zip(lab_u, lab_v, lab_k)}

        if not tried_set or len(k_to_uv) == 0:
            edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
            y_label = torch.empty((0,), device=device, dtype=torch.uint8)
            return y_out, edge_index_lab, y_label, tried_set, harmful_set

        # ---------------- endpoint mode: balanced labels ----------------
        if drop_mode == "endpoint":
            # Extract k_lin indices from sets of (k_lin, drop)
            harmful_k = {int(k) for (k, _d) in harmful_set}
            tried_k = {int(k) for (k, _d) in tried_set}

            # only keep those we can map to (u,v)
            harmful_k = [k for k in harmful_k if k in k_to_uv]
            if len(harmful_k) == 0:
                edge_index_lab = torch.empty((2, 0), device=device, dtype=torch.long)
                y_label = torch.empty((0,), device=device, dtype=torch.uint8)
                return y_out, edge_index_lab, y_label, tried_set, harmful_set

            # candidate negatives = tried but not harmful
            neg_candidates = [k for k in tried_k if (k not in set(harmful_k)) and (k in k_to_uv)]

            # sample same number of negatives as positives (or as many as available)
            n_pos = len(harmful_k)
            n_neg = min(len(neg_candidates), n_pos)

            # reproducible RNG (use your existing rng_seed)
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
        present = PRBCDExtendedPlotting._build_uppertri_bitset(ei_sub, m)  # (m*(m-1)//2,) bool

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
            u_l, v_l, action, k_lin_local = PRBCDExtendedPlotting._sample_one_flip(m, present, g, device)
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
        present = PRBCDExtendedPlotting._build_uppertri_bitset(ei_base, n)

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

            batch = PRBCDExtendedPlotting._sample_k_flips(
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
        lin = PRBCDExtendedPlotting.pairs_to_linear_uppertri(edge_index, n)  # (E',)
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
        uv = PRBCDExtendedPlotting.linear_to_triu_idx(n, torch.tensor([k], device=device))  # (2,1)
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
        uv = PRBCDExtendedPlotting.linear_to_triu_idx(n, pick.long())  # shape (2, m)
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

            k_lin_all = PRBCDExtendedPlotting.pairs_to_linear_uppertri(pair_ei, n).long()  # (m,)

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

    def _pretrain_margin_scorer(self, epochs: int = 20, lr: float = 1e-3, wd: float = 5e-4):
        logging.info(f"[Selector] Pretraining NodeBlockScorer for {epochs} epochs "
                     f"(lr={lr}, wd={wd}, device={self.device}).")
        self.selector.train()
        y = self._make_margin_labels().to(self.device)

        x = self.attr.to(self.device)
        ei = self.edge_index.to(self.device)
        ew = self.edge_weight.to(self.device)

        opt = torch.optim.Adam(self.selector.parameters(), lr=lr, weight_decay=wd)
        with torch.enable_grad():
            for e in range(epochs):
                opt.zero_grad()
                h = self.selector.embed(x, ei, ew)
                pred = self.selector.node_head(h).squeeze(-1)
                loss = F.smooth_l1_loss(pred, y)
                loss.backward()
                opt.step()

                # log a few times during training
                if (e + 1) % max(1, epochs // 5) == 0 or e == 0 or (e + 1) == epochs:
                    logging.info(f"[Selector] pretrain epoch {e + 1}/{epochs} - loss={loss.item():.4f}")

        self.selector.eval()
        self._margin_trained = True
        logging.info("[Selector] Pretraining finished.")

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
            lr: float = 1e-3,
            weight_decay: float = 5e-4,
            use_tqdm: bool = True,
            verbose: bool = True,
    ):
        """
        Train a link prediction GNN on labeled edge pairs with detailed logging.

        Returns:
            model: trained LinkPredictionGNN
        """

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

        # ---- train/val split (stratified 50/50, negatives first half, positives second half) ----
        M = edge_index_lab.size(1)
        assert M == y_label.numel(), "edge_index_lab and y_label must have the same number of examples"
        assert M % 2 == 0, "Expected equal number of negatives/positives (M must be even)."

        half = M // 2
        neg_idx_all = torch.arange(0, half, device=device)
        pos_idx_all = torch.arange(half, M, device=device)

        # shuffle within each class
        neg_perm = neg_idx_all[torch.randperm(half, device=device)]
        pos_perm = pos_idx_all[torch.randperm(half, device=device)]

        train_size_per_class = int(0.8 * half)

        train_idx = torch.cat([neg_perm[:train_size_per_class], pos_perm[:train_size_per_class]], dim=0)
        val_idx = torch.cat([neg_perm[train_size_per_class:], pos_perm[train_size_per_class:]], dim=0)

        # optional: shuffle final indices so batches aren't class-blocked
        train_idx = train_idx[torch.randperm(train_idx.numel(), device=device)]
        val_idx = val_idx[torch.randperm(val_idx.numel(), device=device)]

        if verbose:
            print(f"[LP-GNN] Total labeled pairs: {M}")
            print(f"[LP-GNN] Train pairs: {train_size_per_class}, Val pairs: {M - train_size_per_class}")

        # ---- Define model ----
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
        loss_fn = nn.BCEWithLogitsLoss()

        # progress bar
        epoch_iter = tqdm(range(num_epochs), desc="[LP-GNN] Training") if use_tqdm else range(num_epochs)

        # ---- Training Loop ----
        for epoch in epoch_iter:
            model.train()
            optimizer.zero_grad()

            logits_train = model(x, edge_index_struct, edge_index_lab[:, train_idx])
            loss = loss_fn(logits_train.view(-1), y_label[train_idx])
            loss.backward()
            optimizer.step()

            # ---- Validation ----
            model.eval()
            with torch.no_grad():
                logits_val = model(x, edge_index_struct, edge_index_lab[:, val_idx])
                val_loss = loss_fn(logits_val.view(-1), y_label[val_idx])

                probs_val = torch.sigmoid(logits_val.view(-1))
                preds_val = (probs_val >= 0.5).long()
                acc_val = (preds_val == y_label[val_idx].long()).float().mean()

            # update tqdm bar text
            if use_tqdm:
                epoch_iter.set_postfix({
                    "train_loss": f"{loss.item():.4f}",
                    "val_loss": f"{val_loss.item():.4f}",
                    "val_acc": f"{acc_val.item():.4f}"
                })

            # print verbose log every N epochs
            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                print(
                    f"[LP-GNN] Epoch {epoch + 1:03d}/{num_epochs} | "
                    f"Train Loss={loss.item():.4f} | "
                    f"Val Loss={val_loss.item():.4f} | "
                    f"Val Acc={acc_val.item():.4f}"
                )

        if verbose:
            print("\n[LP-GNN] Training complete.")
            print(f"[LP-GNN] Final Val Loss={val_loss.item():.4f}, Val Acc={acc_val.item():.4f}")

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
            csv_path = PRBCDExtendedPlotting.make_selector_gnn_log_path(
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

                tp, fp, tn, fn = PRBCDExtendedPlotting._confusion_counts(preds_val, yv)
                auc, ap = PRBCDExtendedPlotting._safe_auc_ap_sklearn(probs_val, yv)

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

    def _pretrain_pregnn_linkpred(
            self,
            epochs: int = 20,
            lr: float = 1e-3,
            wd: float = 5e-4,
            neg_per_pos: int = 1,
            batch_size: int = 65536,
            use_bilinear: bool = True,
            edge_dropout: float = 0.1,
            temp: float = 1.0,
            log_every: int = 5,
            # ---- NEW knobs for Top-K sampling ----
            topk_pos_ratio: float | None = None,  # e.g., 0.25 -> keep top 25% positives
            topk_pos_mode: str = "loss",  # {"loss","score"} ranking criterion
            hard_negatives: bool = True,  # do hard negative mining
            neg_oversample: int = 4,  # oversample pool for hard negs
    ):
        """
        Pretrain the GNN encoder via link prediction, with optional Top-K edge sampling.

        Top-K positives:
            - If topk_pos_ratio is not None, compute scores (or per-edge loss)
              for *all* positives this epoch and keep only the Top-K fraction.

        Hard negatives:
            - If hard_negatives is True, for each batch oversample K*neg_oversample
              random non-edges, score them, and keep only the highest-scoring K.

        Loss: BCEWithLogits(pos=1, neg=0)
        """
        import math
        import torch
        import logging
        from torch import nn
        from torch.nn import functional as F

        device = self.device
        self.selector.train()

        # --- Data on device ---
        x = self.attr.to(device).float()
        ei = self.edge_index.to(device)  # (2, E)
        ew = (self.edge_weight.to(device).float()
              if self.edge_weight is not None else None)

        N = int(self.n)
        E = int(ei.size(1))

        # --- Undirected unique positives (u<v) ---
        u = torch.minimum(ei[0], ei[1])
        v = torch.maximum(ei[0], ei[1])
        mask = (u != v)
        u, v = u[mask], v[mask]
        pos_uv = torch.unique(torch.stack([u, v], dim=0), dim=1)  # (2, Epos)
        Epos = pos_uv.size(1)

        # --- Membership set for negatives ---
        pos_keys = (pos_uv[0].long() * N + pos_uv[1].long()).tolist()
        pos_set = set(pos_keys)

        # --- Optional bilinear scorer ---
        bilinear = None
        if use_bilinear:
            bilinear = None  # lazy-init after we see embedding dim

        def score_pairs(h, a, b):
            nonlocal bilinear
            if use_bilinear:
                if bilinear is None:
                    d = h.size(-1)
                    bilinear = nn.Parameter(torch.empty(d, d, device=device))
                    nn.init.xavier_uniform_(bilinear)
                    self.selector.register_parameter("pregnn_bilinear_W", bilinear)
                return (h[a] @ bilinear @ h[b].T).diag()
            else:
                return (h[a] * h[b]).sum(dim=-1)

        def sample_negatives(num_neg: int):
            out_u, out_v = [], []
            needed, tries, max_tries = num_neg, 0, 10
            while needed > 0 and tries < max_tries:
                a = torch.randint(0, N, (needed * 2,), device=device)
                b = torch.randint(0, N, (needed * 2,), device=device)
                valid = (a != b)
                a, b = a[valid], b[valid]
                if a.numel() == 0:
                    tries += 1
                    continue
                uu = torch.minimum(a, b)
                vv = torch.maximum(a, b)
                keys = (uu.long() * N + vv.long()).tolist()
                keep = [k not in pos_set for k in keys]
                keep = torch.tensor(keep, device=device, dtype=torch.bool)
                uu, vv = uu[keep], vv[keep]
                if uu.numel() == 0:
                    tries += 1
                    continue
                take = min(needed, uu.numel())
                out_u.append(uu[:take])
                out_v.append(vv[:take])
                needed -= take
            if len(out_u) == 0:
                uu = torch.arange(0, num_neg, device=device) % (N - 1)
                vv = (uu + 1) % N
                return uu, vv
            return torch.cat(out_u), torch.cat(out_v)

        # --- Optimizer ---
        params = list(self.selector.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

        bce = torch.nn.BCEWithLogitsLoss()
        logging.info(
            f"[PreGNN] LinkPred pretrain: epochs={epochs}, lr={lr}, wd={wd}, "
            f"neg_per_pos={neg_per_pos}, batch_size={batch_size}, edge_dropout={edge_dropout}, "
            f"topk_pos_ratio={topk_pos_ratio}, topk_pos_mode={topk_pos_mode}, "
            f"hard_negatives={hard_negatives}, neg_oversample={neg_oversample}"
        )

        for ep in range(1, epochs + 1):
            self.selector.train()

            # DropEdge for robustness
            if edge_dropout > 0.0 and E > 0:
                keep_mask = torch.rand(E, device=device) > edge_dropout
                ei_train = ei[:, keep_mask]
                ew_train = (ew[keep_mask] if ew is not None else None)
            else:
                ei_train, ew_train = ei, ew

            # Full-batch embeddings (one forward per epoch)
            h = self.selector.embed(x, ei_train, ew_train)  # (N, d)

            # ---- Top-K POSITIVES selection (optional) ----
            pos_idx_pool = torch.arange(Epos, device=device)
            if topk_pos_ratio is not None and 0.0 < topk_pos_ratio < 1.0:
                pu_all = pos_uv[0, :]
                pv_all = pos_uv[1, :]
                s_pos_all = score_pairs(h, pu_all, pv_all)
                if temp != 1.0:
                    s_pos_all = s_pos_all / temp

                if topk_pos_mode == "score":
                    # keep the *highest-scoring* positive edges
                    k_pos = max(1, int(math.ceil(Epos * topk_pos_ratio)))
                    topk = torch.topk(s_pos_all, k=k_pos, largest=True).indices
                    pos_idx_pool = topk
                elif topk_pos_mode == "loss":
                    # keep positives with highest BCE loss vs label=1 (hardest positives)
                    # loss_pos = -log(sigmoid(s_pos))
                    loss_pos = F.softplus(-s_pos_all)  # numerically stable
                    k_pos = max(1, int(math.ceil(Epos * topk_pos_ratio)))
                    topk = torch.topk(loss_pos, k=k_pos, largest=True).indices
                    pos_idx_pool = topk
                else:
                    logging.warning(f"[PreGNN] Unknown topk_pos_mode={topk_pos_mode}; using full positives.")
            else:
                # shuffle all positives if not doing top-k
                perm = torch.randperm(Epos, device=device)
                pos_idx_pool = pos_idx_pool[perm]

            # batching over selected positives
            total_loss = 0.0
            num_batches = int(math.ceil(pos_idx_pool.numel() / batch_size))

            for b in range(num_batches):
                start = b * batch_size
                end = min(pos_idx_pool.numel(), (b + 1) * batch_size)
                idx = pos_idx_pool[start:end]

                pu = pos_uv[0, idx]
                pv = pos_uv[1, idx]

                # negatives needed for this mini-batch
                need_neg = (end - start) * max(1, int(neg_per_pos))

                if hard_negatives:
                    # oversample a pool, then keep the Top-K hardest (highest score)
                    nu_pool, nv_pool = sample_negatives(need_neg * max(1, int(neg_oversample)))
                    s_neg_pool = score_pairs(h, nu_pool, nv_pool)
                    if temp != 1.0:
                        s_neg_pool = s_neg_pool / temp
                    # Top hardest negatives
                    keep_k = min(need_neg, s_neg_pool.numel())
                    hard_idx = torch.topk(s_neg_pool, k=keep_k, largest=True).indices
                    nu, nv = nu_pool[hard_idx], nv_pool[hard_idx]
                else:
                    nu, nv = sample_negatives(need_neg)

                # compute scores
                s_pos = score_pairs(h, pu, pv)
                s_neg = score_pairs(h, nu, nv)
                if temp != 1.0:
                    s_pos = s_pos / temp
                    s_neg = s_neg / temp

                y_pos = torch.ones_like(s_pos)
                y_neg = torch.zeros_like(s_neg)

                loss = bce(s_pos, y_pos) + bce(s_neg, y_neg)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.selector.parameters(), max_norm=5.0)
                opt.step()

                total_loss += loss.item()

            if (ep % max(1, log_every) == 0) or ep == 1 or ep == epochs:
                avg_loss = total_loss / max(1, num_batches)
                logging.info(f"[PreGNN] epoch {ep}/{epochs} - linkpred loss={avg_loss:.4f}")

        self.selector.eval()
        self._pregnn_trained = True
        logging.info("[PreGNN] Link prediction pretraining finished with Top-K sampling.")

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
        self.modified_edge_index = PRBCDExtendedPlotting.linear_to_triu_idx(self.n, self.current_search_space)
        # if the new logic is usable this method is redundant
        return

    def edges_to_current_search_space(self, n: int):
        # first we cut edges so that index build a triu matrix
        # (function triu_idx_to_linear only support triu matrix idx)
        self.edges_to_attack_index = PRBCDExtendedPlotting.flip_matrix_idx_to_triu_idx(self.edges_to_attack_index)
        # we then build linear idx which is the current_search_space
        lin_idx = PRBCDExtendedPlotting.triu_idx_to_linear_idx(n, self.edges_to_attack_index)
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
                                  probability_mass_update: float, probability_mass_projected: float, spearman_rho: float = None):
        self.attack_statistics['loss'].append(loss)
        self.attack_statistics['accuracy'].append(accuracy)
        self.attack_statistics['nonzero_weights'].append((self.perturbed_edge_weight > self.eps).sum().item())
        self.attack_statistics['probability_mass_update'].append(probability_mass_update)
        self.attack_statistics['probability_mass_projected'].append(probability_mass_projected)

        if spearman_rho is not None:
            self.attack_statistics['spearman_rho'].append(spearman_rho)

    def _append_attack_statistics_spearman_rho(self, spearman_rho: float):
        self.attack_statistics['spearman_rho'].append(spearman_rho)

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
        matrix = PRBCDExtendedPlotting.cut_diagonal_entries(matrix)
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
        undirected_edges = PRBCDExtendedPlotting._build_undirected_edge_set(edge_index_struct)

        n_del_like = 0  # pair exists in graph -> would be a deletion candidate
        n_add_like = 0  # pair not in graph -> would be an addition candidate
        missing_decode = 0

        for (k_lin, _drop) in tried_set:
            k_lin = int(k_lin)
            if k_lin < 0 or k_lin >= n * (n - 1) // 2:
                missing_decode += 1
                continue
            u, v = PRBCDExtendedPlotting._uppertri_decode(k_lin, n)
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