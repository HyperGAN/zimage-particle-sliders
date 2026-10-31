# Standalone phase1 qualification

The runtime export is byte-identical to public source dd4e04f. Original public
source qualification is preserved, unchanged, under docs/source/. This document
records standalone installed-artifact checks without an umbrella checkout.

Build from repository root with `python -m build --no-isolation .`; install the
wheel or sdist in an isolated public-CPU dependency environment. Run installed
tests from outside the checkout with `python -m pytest -c /dev/null
--import-mode=importlib -q /path/to/this/repository/tests`.

Test-reference adaptation is explicit in EXTRACTION.md. New receipts are recorded
in qualification.json after verification. No model weights or full/GPU native
training qualification is implied. Core7a8 remains held and pinned unchanged.

## Verified standalone receipts

- Installed wheel: **37 passed in17.31s**, no umbrella checkout/import required.
- Installed sdist: **37 passed**, with the same isolated public CPU dependencies.
- pip check: **No broken requirements found**.
- Product runtime/profile bytes match the exact source export and installed
  site-packages; recursive shared-core source lock passes at unchanged7a8.
- Cross-install compatibility: a real checkpoint created by the original installed
  wheel resumes in this standalone install to the exact full checkpoint hash.
  Original raw/EMA exports produce bit-exact responses at0/0.5/1 after deleting
  training rows and relocating byte-identical base files.
- Installed CLI and both registered entry points execute from outside checkout.

All tests use Python3.12.15/Torch2.6.0+cpu and tiny random native models. The two
reference-test adaptations are listed in EXTRACTION.md; original primary evidence
remains unchanged in docs/source/. Runtime, cohort, native guidance0/shared1,
initialization policy and shared dependency pin are unchanged.
