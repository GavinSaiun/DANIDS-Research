# DANIDS Explorer v1: presentation contract

Explorer is a static, read-only React/TypeScript/Vite application. It is not an
experiment runner or an alternative evaluator. No backend, raw datasets, model
checkpoints, or ignored result bundles are required by its build or runtime.

## Evidence boundary

The Python exporter reads a fixed allowlist of checked-in public sources. Reviewed
source SHA-256 identities (UTF-8 with canonical LF) bind curated RDX projections to
the approved public result document. The F03 SVG is checked against its frozen
manifest and projected at its published annotation precision only. This is not a
reconstruction of unrounded measurements. Study-4 resource totals come directly from
the public visual manifest. The Core feature allowlist is read as literal metadata,
never executed as a model. Missing or changed sources fail before any output write.

The browser consumes only versioned JSON under `explorer/public/evidence`. Each
projection has `schema_version`, `sources`, and `data`; source records contain a
repository-relative path, canonical SHA-256, and hash mode. `manifest.json` enumerates
every projection's canonical byte count and SHA-256. No timestamps or machine paths
enter the output. The client verifies all hashes and runtime schemas before rendering.

The exporter supports `--check` for a no-write exact comparison with committed
projections. Generation requires an absent output directory or exactly matching
existing files; it never overwrites differing files. Existing scientific directories
cannot be used as output. No source evidence is modified. There is no automatic
fallback to local ignored evidence, so a clean clone produces identical outputs.

## Pages and units

- Overview: scientific chain and separate recognition/capability/recovery populations.
- Transfer: keyboard-accessible source-target matrix of F03 rounded mean TPR and
  FPR-budget-ratio annotations. No seed-level values, AUROC, or PR-AUC are invented.
- Health: explanatory envelope and exact 28 label-free features; no health computation.
- Recoverability: independent pathways, censoring, action definitions, and the published
  training-evidence sensitivity means/medians and paired contrasts.
- Provenance: original freeze, post-freeze RDX, software version, and public projection
  hashes are explicitly different identities.

Recharts provides the sensitivity plot; an accessible data table accompanies it.
Hash routes support GitHub Pages subpaths without server rewrites. Vite's relative
base keeps local deployment portable. The UI loads no third-party API or remote font.

## Scientific guardrails

Recognition is same-window harm screening, not anticipatory early warning. Oracle is
a non-deployable one-step comparator, not a global bound. Recovery is bounded by the
frozen B100/D1 and A0–A4 regime. Censored horizons are not failures. P1/P2 are capability
estimands over eligible currently-HARMFUL decisions, summarized over 12 rotation ×
seed units; they are not selected-action success rates or independent flow trials.
B100 is historical; B400/B1600 are prospective. RDX is post-freeze, not Study 6 or part
of confirmatory Study 4. Core reduced accepted update frequency, not labels.
Study 5A is family-conditioned binary detection recall, not attribution/open-set
recognition. H7 mixes prior U-T-C-B evidence with three prospective missing rotations.
Frozen verdicts are copied, never recomputed. No general impossibility claim is made.

## Verification and delivery

Dataset-free Python tests cover source pinning, deterministic output, exact schema,
source immutability, output collision refusal, and clean-clone behavior. Frontend tests
cover runtime schemas/hashes, route rendering, source labels and display interactions.
CI checks projection freshness, TypeScript, frontend tests and production build.
This pass does not deploy, commit, push, or open a PR.
