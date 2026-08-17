from pathlib import Path

import numpy as np
import torch


def load_saved_candidate_score_cache(
    folder,
    *,
    dataset,
    seeds,
    n_nodes,
    candidate_set_sizes=None,
    scoring_modes=None,
    model_name=None,
    subset_fraction=None,
    n_subsets=None,
):
    """
    Load previously scored candidate-set files.

    Each saved file belongs to exactly one victim seed. Files are matched by:

        dataset
        exact seed
        number of nodes
        candidate-set size
        scoring mode

    Optionally also checks:
        model name
        subset_fraction
        n_subsets

    Parameters
    ----------
    folder : str or Path
        Directory containing old candidate-mining .pt files.

    dataset : str
        Dataset name, e.g. "pubmed".

    seeds : iterable[int]
        Exact victim seeds that may be loaded.

    n_nodes : int
        Expected graph node count.

    candidate_set_sizes : iterable[int], optional
        Candidate-set sizes to load.

    scoring_modes : iterable[str], optional
        Scoring modes to load, e.g.
        ["endpoint", "subset_accuracy_drop"].

    model_name : str, optional
        If supplied, require matching "model-..." token in filename.

    subset_fraction : float, optional
        If supplied, validate this for subset_accuracy_drop files when
        available in saved metadata.

    n_subsets : int, optional
        If supplied, validate this for subset_accuracy_drop files when
        available in saved metadata.

    Returns
    -------
    dict
        {
            (seed, candidate_set_size, scoring_mode): {
                "path": Path,
                "saved": dict,
            },
            ...
        }

    Notes
    -----
    Identical duplicate files are ignored. If two files describe the same
    logical run but contain different candidates or scores, an error is raised.
    """

    folder = Path(folder)

    if not folder.exists():
        print(
            f"[MINING CACHE] Folder does not exist: {folder}. "
            "No saved candidate scores loaded."
        )
        return {}

    expected_seeds = {
        int(seed)
        for seed in seeds
    }

    expected_n_nodes = int(n_nodes)

    expected_sizes = (
        {
            int(size)
            for size in candidate_set_sizes
        }
        if candidate_set_sizes is not None
        else None
    )

    expected_modes = (
        {
            str(mode)
            for mode in scoring_modes
        }
        if scoring_modes is not None
        else None
    )

    dataset_token = (
        f"dataset-{str(dataset).lower()}"
    )

    model_token = (
        f"model-{str(model_name).lower()}"
        if model_name is not None
        else None
    )

    def _load_pt(path):
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            return torch.load(
                path,
                map_location="cpu",
            )

    def _to_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()

        return np.asarray(value)

    cache = {}
    skipped = 0

    for path in sorted(
        folder.rglob("*.pt")
    ):
        filename = path.name.lower()

        # -----------------------------------------------------
        # Dataset / model checks from filename
        # -----------------------------------------------------

        if dataset_token not in filename:
            skipped += 1
            continue

        if (
            model_token is not None
            and model_token not in filename
        ):
            skipped += 1
            continue

        # -----------------------------------------------------
        # Load saved object
        # -----------------------------------------------------

        try:
            saved = _load_pt(path)
        except Exception as exc:
            print(
                f"[MINING CACHE] Could not load "
                f"{path.name}: {exc}"
            )
            skipped += 1
            continue

        if not isinstance(saved, dict):
            skipped += 1
            continue

        metadata = saved.get("metadata")
        result = saved.get("mining_result")
        candidates = saved.get("candidates")

        if (
            not isinstance(metadata, dict)
            or not isinstance(result, dict)
            or candidates is None
        ):
            skipped += 1
            continue

        # -----------------------------------------------------
        # Exact run identity
        # -----------------------------------------------------

        required_metadata = {
            "seed",
            "n_nodes",
            "candidate_set_size",
            "scoring_mode",
        }

        if not required_metadata.issubset(
            metadata
        ):
            skipped += 1
            continue

        seed = int(
            metadata["seed"]
        )

        saved_n_nodes = int(
            metadata["n_nodes"]
        )

        candidate_set_size = int(
            metadata["candidate_set_size"]
        )

        scoring_mode = str(
            metadata["scoring_mode"]
        )

        # Exact seed match.
        if seed not in expected_seeds:
            skipped += 1
            continue

        if saved_n_nodes != expected_n_nodes:
            skipped += 1
            continue

        if (
            expected_sizes is not None
            and candidate_set_size
            not in expected_sizes
        ):
            skipped += 1
            continue

        if (
            expected_modes is not None
            and scoring_mode
            not in expected_modes
        ):
            skipped += 1
            continue

        # -----------------------------------------------------
        # Optional subset-scoring parameter checks
        # -----------------------------------------------------

        if (
            scoring_mode
            == "subset_accuracy_drop"
        ):
            if (
                subset_fraction is not None
                and "subset_fraction" in metadata
            ):
                if not np.isclose(
                    float(
                        metadata[
                            "subset_fraction"
                        ]
                    ),
                    float(subset_fraction),
                ):
                    skipped += 1
                    continue

            if (
                n_subsets is not None
                and "n_subsets" in metadata
            ):
                if int(
                    metadata["n_subsets"]
                ) != int(n_subsets):
                    skipped += 1
                    continue

        # -----------------------------------------------------
        # Validate candidate / score alignment
        # -----------------------------------------------------

        n_candidates = len(
            candidates
        )

        if (
            n_candidates
            != candidate_set_size
        ):
            print(
                f"[MINING CACHE] Ignoring "
                f"{path.name}: metadata says "
                f"{candidate_set_size} candidates, "
                f"file contains {n_candidates}."
            )
            skipped += 1
            continue

        required_result_fields = {
            "labels_raw",
            "labels_norm",
            "exists",
            "clean_accuracy",
        }

        if not required_result_fields.issubset(
            result
        ):
            skipped += 1
            continue

        labels_raw = _to_numpy(
            result["labels_raw"]
        ).reshape(-1)

        labels_norm = _to_numpy(
            result["labels_norm"]
        ).reshape(-1)

        exists = _to_numpy(
            result["exists"]
        ).reshape(-1)

        if not (
            len(labels_raw)
            == len(labels_norm)
            == len(exists)
            == n_candidates
        ):
            raise ValueError(
                f"Candidate/score mismatch in "
                f"{path.name}: "
                f"candidates={n_candidates}, "
                f"labels_raw={len(labels_raw)}, "
                f"labels_norm={len(labels_norm)}, "
                f"exists={len(exists)}."
            )

        if (
            scoring_mode
            == "subset_accuracy_drop"
        ):
            if "inclusion_count" not in result:
                raise ValueError(
                    f"{path.name} is a "
                    "subset_accuracy_drop file but "
                    "contains no inclusion_count."
                )

            inclusion_count = _to_numpy(
                result["inclusion_count"]
            ).reshape(-1)

            if (
                len(inclusion_count)
                != n_candidates
            ):
                raise ValueError(
                    f"inclusion_count mismatch "
                    f"in {path.name}."
                )

        # -----------------------------------------------------
        # Cache key = exact experimental configuration
        # -----------------------------------------------------

        key = (
            seed,
            candidate_set_size,
            scoring_mode,
        )

        # -----------------------------------------------------
        # Handle duplicate physical files
        # -----------------------------------------------------

        if key in cache:
            previous = cache[key]["saved"]

            previous_candidates = np.asarray(
                previous["candidates"]
            )

            current_candidates = np.asarray(
                candidates
            )

            previous_scores = _to_numpy(
                previous[
                    "mining_result"
                ]["labels_norm"]
            ).reshape(-1)

            current_scores = labels_norm

            same_candidates = (
                previous_candidates.shape
                == current_candidates.shape
                and np.array_equal(
                    previous_candidates,
                    current_candidates,
                )
            )

            same_scores = (
                previous_scores.shape
                == current_scores.shape
                and np.allclose(
                    previous_scores,
                    current_scores,
                    equal_nan=True,
                )
            )

            if (
                same_candidates
                and same_scores
            ):
                print(
                    "[MINING CACHE] Ignoring "
                    "identical duplicate:",
                    path.name,
                )
                continue

            raise RuntimeError(
                "Conflicting saved candidate-score "
                f"files for {key}:\n"
                f"  {cache[key]['path']}\n"
                f"  {path}"
            )

        cache[key] = {
            "path": path,
            "saved": saved,
        }

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        f"[MINING CACHE] Loaded "
        f"{len(cache)} scored candidate sets."
    )

    loaded_seeds = sorted({
        key[0]
        for key in cache
    })

    print(
        "[MINING CACHE] Seeds:",
        loaded_seeds,
    )

    for key in sorted(cache):
        seed, size, mode = key

        print(
            f"  seed={seed} | "
            f"size={size} | "
            f"mode={mode} | "
            f"{cache[key]['path'].name}"
        )

    if skipped:
        print(
            f"[MINING CACHE] Skipped "
            f"{skipped} incompatible files."
        )

    return cache