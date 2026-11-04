# Current modern-core qualification

The active candidate uses the exact shared core commit
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, product version 0.1.3 and cohort v4.
The tested product environment is Python 3.12.15 with PyTorch 2.6.0+cpu, official
ParticleGAN 0.9.4, HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`,
HNDL 0.10.0, Diffusers 0.42.1 and safetensors 0.8.0.

The isolated installation has no umbrella checkout on its import path. Imported
core files are compared against all 50 source/data files from the exact Git tree;
direct_url.json verifies the public VCS origin. The dependency-resolution report,
pip check and installed wheel/sdist suites are summarized in qualification.json.
The recursive source/data lock includes the core license and unexpected nested
data rejection, in addition to Python, graph and manifest tamper checks.

Tests use tiny random actual Diffusers ZImageTransformer2DModel instances. They
cover matched outputs and VJPs, critic second derivatives, both public BISPF
presets, parameter ownership, initialization reproducibility, exact split resume,
EMA, strengths 0/0.5/1, native-only inference after data deletion and base
relocation, old-format refusal, old-core identity refusal, rollback, caller RNG,
borrowed model state and retryable cleanup failures. Actual old core/cohort
training, raw and EMA packets were refused before native preparation, and their
file hashes were verified unchanged.

The LoRA numerical reference has a precise boundary: shared LoRAModule matches
the historical module AST, but shared LoRANetwork discovery differs. The native
fixture checks shared target names and numerical parity. Full historical discovery
parity is not claimed. Original public evidence remains unchanged in docs/source/.

The phase 1 cross-install compatibility receipt applies only to the original
version 0.1.2/core 7a8 cohort. It is preserved as historical evidence and is not a
claim that version 0.1.3 can load old packets. CORE-MIGRATION.md describes the new
fail-closed compatibility policy and current source boundary.

No product GPU tests, full model training, pretrained quality, production resource
qualification, HNDL optimization, package release or deployment occurred. Native
response domains and gates are unchanged. Deep same-config signature checks still
follow reversible preparation in the shared core; universal pre-hook refusal is
not claimed. Claude RETURN review is required before landing.
