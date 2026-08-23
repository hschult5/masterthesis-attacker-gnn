"""Activate the PRBCD miss-condition extension without modifying repository files.

Place this folder anywhere inside the project tree and call ``activate()`` before
importing ``experiments.experiment_global_attack_direct`` or ``PRBCD``.

The activation lasts only for the current Python process/kernel. Restarting the
kernel restores the repository's normal modules automatically.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the repository containing ``rgnn_at_scale`` and ``experiments``.

    Search order:
    1. ``PRBCD_PROJECT_ROOT`` environment variable.
    2. This package directory and each of its parents.
    3. The current working directory and each of its parents.
    """
    env_root = os.environ.get("PRBCD_PROJECT_ROOT")
    candidates = []

    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    starts = [Path(start).resolve()] if start is not None else []
    starts.extend([PACKAGE_DIR, Path.cwd().resolve()])

    seen = set()
    for base in starts:
        for candidate in (base, *base.parents):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    for candidate in candidates:
        if (
            (candidate / "rgnn_at_scale").is_dir()
            and (candidate / "experiments").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not locate the project root. Put this folder somewhere inside "
        "the repository, or set PRBCD_PROJECT_ROOT=/absolute/path/to/repository."
    )


PROJECT_ROOT = find_project_root()

# Ensure project imports work even when Jupyter was launched from this folder.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_ORIGINAL_STATE: Dict[str, object] = {}
_ACTIVE = False


def _load_module(module_name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def activate() -> Tuple[type, ModuleType]:
    """Activate local PRBCD diagnostics in the current Python process.

    Returns
    -------
    PRBCD:
        The local diagnostics-enabled PRBCD class.
    experiment_global_attack_direct:
        The local direct experiment module used by the notebook.
    """
    global _ACTIVE

    if _ACTIVE:
        from rgnn_at_scale.attacks.prbcd import PRBCD
        from experiments import experiment_global_attack_direct
        return PRBCD, experiment_global_attack_direct

    import experiments
    import rgnn_at_scale.attacks as attacks_package

    canonical_prbcd_module = importlib.import_module(
        "rgnn_at_scale.attacks.prbcd"
    )
    canonical_common_module = importlib.import_module("experiments.common")

    _ORIGINAL_STATE.update({
        "attacks_create_attack": getattr(attacks_package, "create_attack"),
        "attacks_PRBCD": getattr(attacks_package, "PRBCD", None),
        "canonical_PRBCD": getattr(canonical_prbcd_module, "PRBCD"),
        "experiments_common_module": canonical_common_module,
        "experiments_common_attr": getattr(experiments, "common", None),
        "direct_sys_module": sys.modules.get(
            "experiments.experiment_global_attack_direct"
        ),
        "direct_attr": getattr(
            experiments,
            "experiment_global_attack_direct",
            None,
        ),
    })

    # Load the diagnostics implementation under a private name. It still uses
    # the repository's normal base classes and dependencies.
    local_prbcd_impl = _load_module(
        "_prbcd_miss_local_impl",
        PACKAGE_DIR / "prbcd_miss_diagnostics.py",
    )
    local_prbcd_class = local_prbcd_impl.PRBCD

    # Make canonical imports resolve to the local class for this kernel.
    canonical_prbcd_module.PRBCD = local_prbcd_class
    attacks_package.PRBCD = local_prbcd_class

    original_create_attack: Callable = _ORIGINAL_STATE["attacks_create_attack"]  # type: ignore[assignment]

    def local_create_attack(attack, **kwargs):
        attack_name = str(attack).lower()
        if attack_name == "prbcd":
            return local_prbcd_class(**kwargs)
        return original_create_attack(attack, **kwargs)

    attacks_package.create_attack = local_create_attack

    # Install the diagnostics-aware cache handling only in memory.
    local_common_module = _load_module(
        "experiments.common",
        PACKAGE_DIR / "common_miss_diagnostics.py",
    )
    experiments.common = local_common_module

    # Load the diagnostics-aware direct runner after create_attack/common have
    # been replaced, so its imports bind to the local overlay.
    local_direct_module = _load_module(
        "experiments.experiment_global_attack_direct",
        PACKAGE_DIR / "experiment_global_attack_direct_miss_diagnostics.py",
    )
    experiments.experiment_global_attack_direct = local_direct_module

    _ACTIVE = True
    return local_prbcd_class, local_direct_module


def deactivate() -> None:
    """Restore modules that were active before ``activate()`` was called."""
    global _ACTIVE

    if not _ACTIVE:
        return

    import experiments
    import rgnn_at_scale.attacks as attacks_package
    canonical_prbcd_module = importlib.import_module(
        "rgnn_at_scale.attacks.prbcd"
    )

    attacks_package.create_attack = _ORIGINAL_STATE["attacks_create_attack"]
    canonical_prbcd_module.PRBCD = _ORIGINAL_STATE["canonical_PRBCD"]

    original_attacks_prbcd = _ORIGINAL_STATE.get("attacks_PRBCD")
    if original_attacks_prbcd is not None:
        attacks_package.PRBCD = original_attacks_prbcd

    original_common = _ORIGINAL_STATE["experiments_common_module"]
    sys.modules["experiments.common"] = original_common
    experiments.common = original_common

    original_direct = _ORIGINAL_STATE.get("direct_sys_module")
    if original_direct is None:
        sys.modules.pop("experiments.experiment_global_attack_direct", None)
        if hasattr(experiments, "experiment_global_attack_direct"):
            delattr(experiments, "experiment_global_attack_direct")
    else:
        sys.modules["experiments.experiment_global_attack_direct"] = original_direct
        experiments.experiment_global_attack_direct = original_direct

    sys.modules.pop("_prbcd_miss_local_impl", None)
    _ACTIVE = False


def status() -> dict:
    """Return a small diagnostics dictionary for notebook setup checks."""
    return {
        "active": _ACTIVE,
        "package_dir": str(PACKAGE_DIR),
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(PROJECT_ROOT / "data"),
        "artifact_dir": str(PROJECT_ROOT / "cache"),
        "output_dir": str(PACKAGE_DIR / "outputs"),
    }
