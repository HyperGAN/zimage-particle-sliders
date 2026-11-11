# Explicit cached fresh95 opt-in

`embedded_zimage_shared.fresh95` is a separate initialization profile. Existing
CPU/CUDA backends, defaults, learning rates and old FULL/RAW checkpoints keep
their original routes. Do not migrate or reinitialize a completed cohort.

Use an already compatible, genuinely installed `embedded-zimage-shared-preview`
0.1.4 owner. The original core0.13.0 Git3dd672a7, recursive Source/PG0.9.5/HNDL
guards and Diffusers0.42.1 guards remain active. The installed wheel's generated
RECORD must own both original files and this namespace; a Source prepend,
editable checkout or handwritten RECORD does not replace that ownership. This
addition changes no dependency or installer and does not upgrade the core.

The caller supplies an authenticated original staged model and already-completed
original PreparedResponse/ResponseRow records, with whole output and Source
identities. `save_prepared_cache` serializes those records; it does not call
`adapter.prepare`, an encoder, a teacher or a request-to-target collector. Keep
real completed-native data separate from synthetic software-only data. No
pretrained model/cache availability is supplied or implied by this guide.

The original rank16/alpha16 graph and Parameter objects are retained. The new
scoped public PG0.9.5 initializer applies same-Parameter down QR at the original
Kaiming(a=1) variance1/fan_in. UP remains zero/KEEP, the original HNDLFieldCritic
keeps its constructor values, and no particle bank is added. Initialization
receipt/signature exists before optimizers. The game keeps response_scale1;
target-delta RMS is diagnostic. Weight EMA is refused; exports use current RAW.

The following are library calls, not a model download or ready-made job. Supply
`config`, completed records, immutable paths, `component_factory(request)` and
`owner_admission(request)` from your application. The factory returns the
original staged native model; the owner authenticates origins and admits the
operation before added components. A fresh cohort and owner directory must be
new. Rates/budget in `config` are explicit choices, not selected by this API.

```python
from embedded_zimage_shared.fresh95 import (
    save_prepared_cache, open_training, open_inference,
)

cached = save_prepared_cache(
    new_cache_path, config=config, prepared=completed_original_rows,
    teacher_origins=authenticated_origins, cache_id=cache_id,
    scope="completed-native", owner_admission=owner_admission,
)
with open_training(
    config, cached, new_directory, mode="fresh", cohort_id=new_cohort,
    component_factory=component_factory, owner_admission=owner_admission,
    ema_decay=None,
) as run:
    # Entry FULL0/RAW0 and initialization receipt are already durable.
    # Call run.update() only for the new work your application intends to run.
    full = run.save(new_FULL_path)
    raw = run.export(new_RAW_path)

with open_training(
    config, cached, resume_directory, mode="resume", resume=full,
    cohort_id=new_cohort, component_factory=component_factory,
    owner_admission=owner_admission, ema_decay=None,
) as resumed:
    pass  # Entire typed game/optimizers/cursors/schedule/RNG restored.

with open_inference(
    raw, component_factory=component_factory, owner_admission=owner_admission,
) as inference:
    velocity = inference.response(
        condition=condition, latent=latent, timestep=timestep, strength=1.,
    )
```

Configuration explicitly chooses `cpu-native-fp32` or `cuda-native-fp32`, its
original model bundle identity/response geometry, steps/batch/seed/preset/rates,
`ema_decay=None`, PG0.9.5/release-0.9.5/continuous, device and checkpoint choice.
The CPU lane stays CPU/no checkpoint enablement. The CUDA lane retains the
original caller-configured math/physical ownership/checkpoint policy; this
module does not set startup flags or admit GPUs. Capability strength remains
0..1. Response returns the original raw flow velocity, not a full-image render.

Resume accepts only this profile's typed FULL, receipt and completed cache. It
skips PG initialization and checks whole restored state. RAW inference restores
only the added network and skips cache loading, critic/game/optimizers and PG
initialization; original adapter constructors still precede RAW restoration.
Closing blocks further work, restores only exact owned pending hooks/flags/
modes, retains foreign changes on refusal, and removes the owner marker last.
The caller's native model is never moved or unloaded by cleanup.

Four installed tiny CPU software cases passed with one total update, including
strict FULL/RAW and foreign-safe cleanup. That result does not establish
pretrained image quality, CUDA/native full-image qualification or new LR
optima/defaults. Existing learning-rate results remain scoped to their original
cohorts; no teacher collection, old search, test/proof or output is rerun here.

## Explicit installed-owner software check

The retained four-case fixture is published as `tests/fresh95_owner_check.py`.
That filename is outside pytest’s default `test_*.py` and `*_test.py` discovery,
so ordinary repository CI keeps collecting its existing tests. The fixture body
is unchanged from the retained installed-owner software observations. Publication
routing adds no new software pass or native qualification.

This check requires the genuine installed wheel’s generated RECORD to own both
the original namespace and all seven fresh95 modules, with the frozen compatible
core, ParticleGAN and other runtime dependencies described above. An editable
checkout or a Source-tree import is not that installed owner. For an explicitly
requested owner check, use that compatible installed environment, an independent
working directory, an empty PYTHONPATH and a neutral pytest config:

```sh
cd /absolute/path/to/neutral-working-directory
PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /absolute/path/to/compatible-env/bin/python -m pytest \
  -q -c /dev/null \
  /absolute/path/to/checkout/tests/fresh95_owner_check.py
```

The paths are caller-supplied. The explicit file argument selects these four
software cases; `-c /dev/null` avoids repository configuration and its Source
pythonpath. The fixture includes one total tiny update. This proposal does not
execute or replay it, and the earlier completed software observations remain
separate from public CI and pretrained/native qualification.
