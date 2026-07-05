
import logging
from typing import Any, Dict, Sequence, Union, Optional

import numpy as np
from sacred import Experiment

import torch

from rgnn_at_scale.attacks import Attack, create_attack
from experiments.common import prepare_attack_experiment, run_global_attack

try:
    import seml
except:  # noqa: E722
    seml = None

ex = Experiment()

if seml is not None:
    seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None

    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    # default params
    data_dir = './data'
    dataset = 'cora_ml'
    make_undirected = True
    binary_attr = False
    data_device = 0

    device = 0
    seed = 0

    attack = 'PRBCD'
    attack_params = dict(
        epochs=500,
        fine_tune_epochs=100,
        keep_heuristic="WeightOnly",
        block_size=100_000,
        do_synchronize=True,
        loss_type="tanhMargin",
    )
    epsilons = [0.01, 0.1]

    artifact_dir = 'cache'
    model_label = "Soft Median GDC (T=0.5)"
    model_storage_type = 'pretrained'
    pert_adj_storage_type = 'evasion_global_adj'
    pert_attr_storage_type = 'evasion_global_attr'

    debug_level = "info"

@ex.automain
def run(graph, data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], selector_params: Dict[str, Any], epsilons: Sequence[float],
        binary_attr: bool, make_undirected: bool, seed: int, artifact_dir: str, pert_adj_storage_type: str,
        pert_attr_storage_type: str, model_label: str, model_storage_type: str, device: Union[str, int],
        data_device: Union[str, int], debug_level: str, semi: bool, use_cert: str = "none"):

    results = []
    surrogate_model_label = False

    (
        attr, adj, labels, _, _, idx_test, storage, attack_params, pert_params, model_params, m
    ) = prepare_attack_experiment(
        data_dir, dataset, attack, attack_params, epsilons, binary_attr, make_undirected, seed, artifact_dir,
        pert_adj_storage_type, pert_attr_storage_type, model_label, model_storage_type, device, surrogate_model_label,
        data_device, debug_level, ex
    )

    if model_label:
        model_params['label'] = model_label

    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    last_gradient = None
    last_attack_stats = None

    def _copy_attack_statistics(obj):
        """Detach/copy nested attack statistics without flattening dictionaries."""
        if torch.is_tensor(obj):
            value = obj.detach().cpu()
            return value.item() if value.ndim == 0 else value.clone()

        if isinstance(obj, np.ndarray):
            return obj.copy()

        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, dict):
            return {
                key: _copy_attack_statistics(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [_copy_attack_statistics(value) for value in obj]

        if isinstance(obj, tuple):
            return tuple(_copy_attack_statistics(value) for value in obj)

        return obj
        return out

    for model, hyperparams in models_and_hyperparams:
        model_label = hyperparams["label"]
        logging.info(f"Evaluate {attack} for model '{model_label}'.")

        adversary = create_attack(
            attack,
            attr=attr,
            adj=adj,
            labels=labels,
            model=model,
            idx_attack=idx_test,
            device=device,
            data_device=data_device,
            binary_attr=binary_attr,
            make_undirected=make_undirected,
            **attack_params
        )

        for epsilon in epsilons:
            gradient = run_global_attack(
                graph=graph,
                dataset=dataset,
                epsilon=epsilon,
                m=m,
                storage=storage,
                pert_adj_storage_type=pert_adj_storage_type,
                pert_attr_storage_type=pert_attr_storage_type,
                pert_params=pert_params,
                adversary=adversary,
                model_label=model_label,
                semi=semi,
                use_cert=use_cert,
                seed=seed,
                selector_params=selector_params,
            )
            last_gradient = gradient

            adj_adversary = adversary.adj_adversary
            attr_adversary = adversary.attr_adversary

            logits, accuracy = Attack.evaluate_global(
                model.to(device),
                attr_adversary.to(device),
                adj_adversary.to(device),
                labels,
                idx_test
            )

            results.append({
                'label': model_label,
                'epsilon': epsilon,
                'accuracy': accuracy
            })

            stats_obj = getattr(adversary, "attack_statistics", None)
            if stats_obj is not None:
                last_attack_stats = _copy_attack_statistics(stats_obj)
            else:
                last_attack_stats = {}

            if getattr(adversary, "rq1_enabled", False):
                rq1_stats = last_attack_stats.get("rq1")
                if not isinstance(rq1_stats, dict):
                    raise RuntimeError(
                        "RQ1 is enabled, but attack_statistics['rq1'] "
                        "was not returned as a dictionary."
                    )
                if "final_linear_ids" not in rq1_stats:
                    raise RuntimeError(
                        "RQ1 statistics are missing 'final_linear_ids'."
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    assert len(results) > 0

    return {
        'results': results,
        'gradient': last_gradient,
        'attack_statistics': last_attack_stats,
    }
