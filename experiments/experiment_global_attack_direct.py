
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
def run(ads_mode, graph, data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float],
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
    last_attack_stats = None  # dict[str, list]

    # --- helper to normalize stats -> plain lists/floats ---
    def _to_plain_list(seq):
        out = []
        for x in seq:
            # try tensor -> .item(), numpy -> .item(), else keep as-is
            try:
                out.append(float(x))
            except Exception:
                try:
                    out.append(x.item())
                except Exception:
                    out.append(x)
        return out

    for model, hyperparams in models_and_hyperparams:
        model_label = hyperparams["label"]
        logging.info(f"Evaluate  {attack} for model '{model_label}'.")
        adversary = create_attack(
            attack, attr=attr, adj=adj, labels=labels, model=model, idx_attack=idx_test,
            device=device, data_device=data_device, binary_attr=binary_attr,
            make_undirected=make_undirected, **attack_params
        )

        for epsilon in epsilons:
            # run the attack (may load from cache or actually optimize)
            gradient = run_global_attack(ads_mode=ads_mode,graph=graph, dataset=dataset,
                epsilon=epsilon, m=m, storage=storage, pert_adj_storage_type=pert_adj_storage_type, pert_attr_storage_type=pert_attr_storage_type,
                pert_params=pert_params, adversary=adversary, model_label=model_label, semi=semi, use_cert=use_cert, seed=seed,
            )
            last_gradient = gradient  # keep for return

            # evaluate on the adversarial graph
            adj_adversary = adversary.adj_adversary
            attr_adversary = adversary.attr_adversary

            logits, accuracy = Attack.evaluate_global(
                model.to(device), attr_adversary.to(device),
                adj_adversary.to(device), labels, idx_test
            )

            results.append({
                'label': model_label,
                'epsilon': epsilon,
                'accuracy': accuracy
            })

            # ---- ALWAYS capture per-epoch stats for the *current* run ----
            stats_obj = getattr(adversary, "attack_statistics", None)
            if stats_obj:
                last_attack_stats = {k: _to_plain_list(v) for k, v in stats_obj.items()}
            else:
                # ensure the key exists in the final return even if empty (e.g., cached path)
                last_attack_stats = {}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    assert len(results) > 0

    # ---- return dict includes stats no matter the use_cert mode ----
    return {
        'results': results,
        'gradient': last_gradient,
        'attack_statistics': last_attack_stats,  # <— your CSV writer will find this
    }
