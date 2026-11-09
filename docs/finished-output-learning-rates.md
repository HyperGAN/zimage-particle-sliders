# Golden Hour: scored RAW32 and sampled RAW128

The best tested rate is **G0.001**, selected from four saved **RAW32** arms.
Its four-case intended-effect mean is **0.0010827196141084035**. The comparison
used original bridge/sailboat neutral DEV prompts × seeds49001/49002,
native512, 9 steps, CFG0, FP32 base and VAE, batch2/rank16/horizon128. The
other rates were 0.0001, 0.012 and 0.036. This is a frozen SigLIP2 model
instrument and fixed-D comparison, not human or statistical validation.

For the displayed sailboat seed49002 native512 preset, use the existing
selected **G0.001 RAW128** checkpoint with strength4. It genuinely continued
RAW32→128 for96 new updates at the original horizon128. Strength1 is subtle;
strength4 shows the intended direction in that case. **This RAW128/strength4
sampling observation is unscored**: it neither replaces the RAW32 LR comparison
nor establishes a new LR score or pooled result. Do not repeat that completed
continuation, search, rendering or scoring.

## Existing options

A common session's `options` uses this exact fragment:

```json
{"lr": 0.001, "d_lr_mult": 18.0, "ema_decay": null}
```

This records actual D0.018000000000000002. Z-Image is adapter-only: it has no
trainable particle bank and refuses a non-null `prior_lr_mult`. See
[rate admission](../src/embedded_zimage_shared/learning_rates.py) and
[common backend](../src/embedded_zimage_shared/backend.py). The common CLI reads
those JSON options; product CLI options are `--lr` and `--d-lr-mult`, with
non-null `--prior-lr-mult` refused. Existing native preparation and original
conditioning remain necessary. These options do not transplant the captured
research checkpoint into a new package/source cohort or qualify an arbitrary
public sampling path. No additional fresh confirmation is required.

## Method and reuse

For each fixed matched case, subtract intended-effect intensity in OFF from ON,
then take the arithmetic mean with equal case weights. Higher wins. Keep
negative gains, zeros and exact ties. Missing/failed outputs are missing evidence,
not synthetic zero. Teacher/student relative SSE, preservation, beauty,
confidence and completion do not select, veto or break ties.

Use RAW current adapter, critic and particle-bank weights where present, with
weight EMA disabled (`ema_decay=None` / JSON `null`). Official scalar optimizer
statistics are unchanged. Reuse the completed searches, exact source-bound RAW
checkpoints, outputs and first observations. Finish only genuinely missing
declared work or exactly resume its latest complete FULL state; never replay
committed updates. No additional fresh confirmation or worse-arm completion is
required. Older SSE tables and source-cohort reports remain historical diagnostics.

A rate comparison varies G while retaining its declared actual D and bank rate
where present. The option names are `lr` (G), `d_lr_mult` (D/G), and
`prior_lr_mult` (bank/G). Copy the saved literals: rounded quotients can alter
exact recipe identity. These conditional comparisons are not global or joint
optima, and this documentation changes no package defaults or checkpoint pins.

## Inspecting the saved comparison

[The portable summary](finished-output-rate.json) contains every selected-rate
case gain, all candidate means and source-record SHA256 identities. Its scalar
arithmetic is reproducible from those numbers. The retained full records,
original conditioning, native assets and selected RAW checkpoints are private
campaign artifacts and are not bundled as public downloads here. This is not a
standalone image-reproduction recipe; install/plan or CPU conformance is not
native finished-output evidence. Keep historical source-cohort reports within
their recorded scope.
