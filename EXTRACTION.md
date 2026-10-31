# Phase1 extraction boundary

Source: public HyperGAN/particle-sliders at
`dd4e04f23b89a1d88ef887dbe22b4d5aacd53dc7`, product root
`packages/embedded-zimage-shared/`. Destination: HyperGAN/zimage-particle-sliders.
This is an allowlisted source export, not a mirrored Git history.

The runtime `src/embedded_zimage_shared/*` files and pyproject.toml are byte-exact
exports. Distribution `embedded-zimage-shared-preview`0.1.2, import namespace,
entry points, initialization policy, cohortv3 and native response/row protocolv2
are unchanged. Existing artifact behavior and source identities are preserved.
Original MIT LICENSE is copied with its copyright/attribution intact.

External runtime dependencies remain the exact pinned shared-core candidate7a8,
its HyperGAN/HNDL closure, Diffusers0.42.1, safetensors0.8.0 and their resolved
CPU dependencies. The shared core remains held pending central qualification;
its public VCS package subdirectory is a shared-library dependency, not an
umbrella-local runtime import. No shared-core API or graph optimization is added.

Two tests previously depended on umbrella files. The paired LoRA parity reference
now imports the locked shared-core native_lora.LoRANetwork, which implements the
same ordinary down/up computation. The format-refusal test generates a temporary
historical-format safetensors fixture from that reference. It does not claim to
rerun the full historical UNI trainer. The exact original public36-case/native
and16-case/historical receipts remain unchanged under docs/source/.

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
rewritten or silently substituted in phase1.

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
