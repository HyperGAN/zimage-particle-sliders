# Current bounded CPU qualification

The active version 0.1.4/cohort v5 pins core0.12 at
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137` and official ParticleGAN0.9.4.
It uses HyperGAN `5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0,
Diffusers0.42.1 and safetensors0.8.0. Current checks use a fresh Python3.11.17
environment with CPU-only Torch2.15.0+cpu and no inherited site-packages.

The exact public Git dependency resolves successfully and `pip check` is clean.
The lock was computed from all 574 tracked core source/data files at the declared
commit and compared with the installed distribution. Its VCS origin and imported
package location are checked before attachment. Official PG source and the
existing HyperGAN/HNDL and native Diffusers source gates remain enforced.

The installed tests use tiny random actual Diffusers native API models on CPU.
They cover adapter outputs/VJPs, critic second derivatives, both BISPF presets,
parameter ownership, seeded initialization and caller RNG, exact split resume,
EMA, strengths, native-only inference after data deletion and base relocation,
old source/format refusal, source/data tampering, rollback and retryable cleanup.
The adapter-only generator remains without a particle bank.

The new installed shared-dispatch test checks profile planning/validation,
periodic checkpoints, step2-to-step4 resume against uninterrupted training,
v1/v2 exports, EMA inference after frozen rows are deleted, and rejection of
`dummy=false`. The boolean marker never bypasses native shape, source or finite
FP32 CPU validation. Source-only/default/native qualification is not promoted.

Exact test results and runtime/source hashes are in `qualification.json`.
Previous receipts are immutable under `docs/cohort-v4/` and `docs/source/`.
They apply to their original source cohorts. The LoRA numerical reference and
bounded native target-name checks do not imply full historical discovery parity.

No pretrained model, GPU job, native renderer, full-model quality/resource
qualification, new weights, release, merge or deployment is included. Deep
same-config training-signature checks still follow reversible preparation;
universal pre-hook packet refusal is not claimed. Claude RETURN review is
required before landing.
