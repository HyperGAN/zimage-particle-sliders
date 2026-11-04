# Current shared-core upgrade

Version 0.1.4 / cohort `embedded-zimage-adapter-gan-pg094-hg-hndl010-v5` uses
core 0.12.0 at `ddb41423bcefc1fe76cd94d0f9b367353b3ea137` and official
ParticleGAN 0.9.4 (source `330eb94ee748cdcad131b0275d8de75fd5a7f96f`).
The core includes the merged shared plan, isolated execution and periodic
checkpoint APIs. HyperGAN/HNDL and Diffusers pins remain explicit in the package.

The source gate now binds the complete distribution closure: 573
`particle_sliders` source/data files plus `concept_slider_core/__init__.py`.
It checks the public VCS origin, exact commit and subdirectory, imported package
location, official ParticleGAN source, native Diffusers bytes and verified HNDL
builder before attachment. The old cohort-v4 qualification, migration record
and 50-file profile are preserved byte-for-byte in `docs/cohort-v4/`; they are
historical evidence and do not qualify this new source pin.

The distribution namespace and existing CLI remain stable. New exports and
checkpoints carry cohort v5/current core identity; old packets and frozen rows
retain their old identities and must use their original immutable source pin.
No historical packet, recipe or weight is rewritten.

The installed `zimage-fixture` training profile selects the existing response
session through the merged dispatcher. It admits an explicit `options.dummy=true`
boolean only, indicating the caller staged a tiny random native API fixture.
It does not qualify a pretrained bundle. From the installed product environment:

```sh
particle-slider plan --config fixture-session.json --profile zimage-fixture --no-reporting
particle-slider train --config fixture-session.json --profile zimage-fixture --check --no-reporting
particle-slider train --config fixture-session.json --profile zimage-fixture \
  --out first --checkpoint-every 1 --stop-after 2 --selection ema --export-format both --no-reporting
particle-slider train --config fixture-session.json --profile zimage-fixture \
  --out resumed --checkpoint-every 1 --resume first/artifacts/checkpoints/step-000002 \
  --selection ema --export-format both --no-reporting
```

`fixture-session.json` is the existing `SessionConfig.as_dict()` JSON described
in README, with `options.dummy` set to true. Paths refer to explicit local files;
no pretrained model is downloaded. The marker does not change the native
geometry, FP32 CPU, source, identity, finite-value or frozen-row gates.

This is the existing adapter-only generator arm (`particles=None`), with official
PG non-saturating logistic/BISPF and DualNorm. It has no SBNP particle bank and
does not claim the full particle-prior recipe. No GAN implementation is vendored.
Native rendering, pretrained quality, GPU/resources and production admission
remain unqualified. Claude RETURN review is required before landing.
