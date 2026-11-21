# Explicit learning rates

Current finished-output decision and sole selection method:
[scoped learning-rate guide](docs/finished-output-learning-rates.md) and
[portable rates/case scores](docs/finished-output-rate.json). Use RAW weights,
no weight EMA, equal-case intended-effect ON−matched OFF, and reuse completed
work; no additional fresh confirmation. The option/default and CPU integration
descriptions below retain their own source-cohort scope.


Direct preparation and common session configuration accept `lr` and
`d_lr_mult`. The default `bispf_optimized` preset from ParticleGAN 0.11.1
uses a base rate of 0.012 and a critic multiplier of 1.5. Explicit values must
be finite and positive. Z-Image trains adapters without a particle bank, so
it rejects any non-null `prior_lr_mult`.

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
