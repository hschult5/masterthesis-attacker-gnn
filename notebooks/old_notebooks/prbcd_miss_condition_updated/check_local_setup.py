#!/usr/bin/env python3
"""Smoke-test the local overlay without running an attack."""

import inspect

from activate_local_prbcd_miss import activate, status

PRBCD, experiment_global_attack_direct = activate()

required = {
    "miss_diagnostics_enabled",
    "miss_sampling_seed",
    "miss_checkpoint_epochs",
    "miss_retention_policy",
    "miss_probe_linear_ids",
    "miss_injection_linear_ids",
}
missing = required - set(inspect.signature(PRBCD.__init__).parameters)
if missing:
    raise RuntimeError(f"Local overlay is incomplete: {sorted(missing)}")

print("Local PRBCD miss-condition overlay is active.")
for key, value in status().items():
    print(f"{key}: {value}")
print("direct runner:", experiment_global_attack_direct.__file__)
