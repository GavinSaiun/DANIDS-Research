# DANIDS Explorer

Static, read-only React/TypeScript/Vite presentation of frozen Studies 1–5 and the
separately frozen post-freeze RDX evidence. No backend, raw data, model execution,
experiment bundles, or runtime external API is used.

## Local development

From the repository root, with DANIDS installed:

```text
python -m danids export-explorer-data --output-dir explorer/public/evidence --check
cd explorer
npm ci
npm run typecheck
npm test
npm run build
npm run dev
```

Node 22.12+ is required. The seven checked-in JSON files are sufficient in a clean
clone. `npm run preview` serves the built site locally. Do not use `file://`; browser
hash verification requires localhost or HTTPS. No deployment is configured or run.
Vite uses a relative base and hash routes, suitable for a GitHub Pages subdirectory.

## Data and trust

See [the architecture contract](../docs/explorer_architecture.md). Sources are
allowlisted and pinned in `danids.evaluation.explorer_export`. RDX numbers are reviewed
transcriptions of the hash-bound public result document. Transfer values are the
rounded annotations of the hash-verified public F03 SVG, not unrounded measurements.
Study-4 resource totals come from the existing public visual manifest.

Generation without `--check` writes only an absent directory; differing existing
outputs are refused, never overwritten. Updating the presentation contract requires
explicit source review and a new output location, not editing frozen evidence.
Every projection records source identities. Browser startup validates the exact file
roster, canonical LF hashes, byte counts and runtime schemas before rendering.
Manifest verification detects corruption, not an attacker replacing both manifest
and projections. Repository review is the trust anchor.

## Page map

- `#/overview`: scientific chain, separate populations and scope limits.
- `#/transfer`: interactive rounded transfer matrix with cell inspection.
- `#/health`: evaluator envelope and expandable label-free feature contract.
- `#/recoverability`: recognition/capability/deployment, actions, training sensitivity.
- `#/provenance`: evidence tags, digests, frozen ledger and downloadable projections.

The charts are accompanied by text/exact-value tables, keyboard-accessible controls,
explicit units and denominator labels. Motion is nonessential and respects reduced
motion preferences. AUROC/PR-AUC, seed-level transfer and unrounded FPR are not exposed
because this clean-clone public-source subset does not provide them.
