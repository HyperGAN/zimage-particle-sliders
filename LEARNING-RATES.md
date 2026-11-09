# Explicit learning rates

Current finished-output decision and sole selection method:
[scoped learning-rate guide](docs/finished-output-learning-rates.md) and
[portable rates/case scores](docs/finished-output-rate.json). Use RAW weights,
no weight EMA, equal-case intended-effect ON−matched OFF, and reuse completed
work; no additional fresh confirmation. The option/default and CPU integration
descriptions below retain their own source-cohort scope.


Direct preparation and common session API/JSON configuration accept `lr`,
`d_lr_mult`, and `prior_lr_mult`. Omitted overrides preserve ParticleGAN 0.9.4
base 0.012 and critic multiplier 1.5. Explicit base and critic values must be
finite and positive. Any non-null prior multiplier is refused because Z-Image
is adapter-only and has no trainable particle bank.

The common CLI consumes the same JSON options; there is no second LR default.
The product CLI additionally accepts `--lr` and `--d-lr-mult`, and refuses
`--prior-lr-mult` before reading native model paths.

The shared game constructs actual optimizer groups and binds the resolved
recipe into exact resumable identity. Changed rates require a new run.
Historical source-bound packets and receipts keep their original identities;
no rebinding or old source-lock rewrite is provided.

Installed tiny-native CPU tests cover actual optimizer rates and resume.
The existing native admission, frozen-teacher, precision and rendering gates
remain in force. Those control checks alone do not establish tuned native learning rates; the
separately captured RAW32 comparison above has its own exact scope.
