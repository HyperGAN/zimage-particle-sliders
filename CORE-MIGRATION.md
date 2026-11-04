# Shared-core 0.12 / PG 0.9.4 source update

The current opt-in product is version 0.1.4, cohort
`embedded-zimage-adapter-gan-pg094-hg-hndl010-v5`, pinned to core
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`. It reuses the implementation
from PR1 and includes the user-merged shared training/execution APIs.

Official ParticleGAN remains 0.9.4, source
`330eb94ee748cdcad131b0275d8de75fd5a7f96f`. No GAN equation or optimizer is
copied into the product. The existing adapter-only generator has `particles=None`;
it does not implement the full SBNP particle-prior recipe.

The complete 574-file source/data closure covers both installed core namespaces,
including the new execution modules and compatibility namespace. Runtime admission
also checks exact public VCS provenance, official PG source, Diffusers native bytes
and the HNDL builder. The installed `zimage-fixture` profile uses the existing
response lifecycle through the shared isolated worker. It requires an actual
boolean `dummy=true` marker and keeps all native/model/geometry gates.

Previous version 0.1.3/cohort v4 stays bound to core
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. Its qualification JSON, migration
record, qualification prose and source profile are retained byte-for-byte under
`docs/cohort-v4/`. Earlier v3 evidence remains in `docs/source/` and original
Git history. No old packet, frozen cache, recipe or model weight is relabeled.
New source/cohort identities require explicit cache regeneration; old packets
must use their original immutable environment. No cross-cohort portability is
claimed.

See `QUALIFICATION.md`, `qualification.json` and `docs/pg094-upgrade.md` for exact
CPU evidence and command examples. Pretrained/GPU/image-quality/production
qualification remains absent, native rendering remains unsupported, and Claude
RETURN review is still required before landing. The other process's unified
feature branch, default scaffold, PR1 branch and shared-core main are untouched.
