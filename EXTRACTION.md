# Phase 1 extraction boundary

Source: public HyperGAN/particle-sliders at
`dd4e04f23b89a1d88ef887dbe22b4d5aacd53dc7`, product root
`packages/embedded-zimage-shared/`. Destination: HyperGAN/zimage-particle-sliders.
This is an allowlisted source export, not a mirrored Git history.

At phase 1 commit `350437ed5e26df6a7c4ec58105d5993c224edf3d`, runtime files
and `pyproject.toml` were byte-exact exports. Distribution version 0.1.2, its
namespace, entry points, initialization policy, cohort v3 and native response/row
protocol version 2 were preserved. The active coordinated core migration is a
separate cohort v4 candidate described in CORE-MIGRATION.md; original phase 1
source and packet identities remain available at their original immutable pin.
Original MIT LICENSE is copied with its copyright/attribution intact.

Phase 1 used the exact shared core commit `7a8a321d04323d49319d0b08e6093a0ddce36b27`,
its HyperGAN/HNDL closure, Diffusers 0.42.1 and safetensors 0.8.0. The active
modern cohort now pins the centrally cleared exact `a794e3b4a8813b012f2c4faa3da82b78d4a90214` source;
its public VCS package subdirectory is a shared-library dependency, not an
umbrella-local runtime import. No shared-core API or HNDL graph optimization is
added. Product cleanup wrappers follow the new core's explicit-primary-error and retryable-close contract.

Two tests previously depended on umbrella files. The paired LoRA parity reference
now imports the locked shared-core native_lora.LoRANetwork, which implements the
same ordinary down/up computation. The LoRAModule AST matches the historical
module, but shared LoRANetwork discovery differs. These tests cover numerical
module parity and bounded shared target names, not full historical discovery.
The format-refusal test generates a temporary historical-format safetensors fixture from that reference. It does not claim to
rerun the full historical UNI trainer. The exact original public 36-case native
and 16-case historical receipts remain unchanged under docs/source/.

## Tracked later migration

These historical/model-specific paths remain intact upstream and are not needed
by the installed modern distribution:

- conceptmod/textsliders/train_lora_zimage.py
- conceptmod/textsliders/slider_targets.py (Z-Image sections only)
- conceptmod/textsliders/data/config-zimage.yaml
- conceptmod/textsliders/data/prompts-zimage.yaml
- tests/test_zimage_slider.py
- docs/zimage-slider.md

The historical trainer also uses the multi-model conceptmod/textsliders/lora.py
helper. Moving it wholesale would import unrelated model support; its future
shared/model-specific ownership needs review. No historical files are deleted,
rewritten or silently substituted in phase 1.

## Publication and qualification boundaries

The default branch is only a minimal README/LICENSE scaffold. Actual product
source is on a review branch and draft PR. No workflows, permission/access
configuration, model weights, logs, recordings, run outputs, private evidence,
local environment paths or mirrored upstream history are exported. No package
release, deployment, GPU job, merge or core admission occurs.

CPU qualification uses tiny random actual native Diffusers models. Production
precision/resources, pretrained quality, GPU behavior and media rendering remain
unqualified. Deep same-config training signature validation still follows
reversible preparation in the shared core; universal pre-hook refusal is not
claimed. Root owns Claude RETURN review, final-core qualification and merges.
