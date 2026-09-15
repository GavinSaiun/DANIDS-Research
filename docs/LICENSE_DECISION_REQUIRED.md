# License decision required before public release

## Current status

This repository has no standalone public-use license. The package metadata currently
states `All rights reserved`. That means the repository does not presently grant
permission to copy, modify, redistribute, or create derivative works merely because the
source is visible.

The license is intentionally omitted from `CITATION.cff` until a choice is approved.
Citation metadata and permission to reuse are separate matters.

## Release decision

The repository owner must choose and approve the intended terms before public release.
The decision should address at least:

- whether the source code is released under a standard open-source license;
- whether documentation, frozen result tables, and thesis figures use the same terms or
  a separate content/data license;
- whether third-party dataset terms constrain redistribution of any derived material;
- whether model checkpoints or future archives require separate terms; and
- whether institutional, candidature, funding, or publication obligations apply.

This file does not recommend or select a license and is not legal advice.

## Required release updates

After approval:

1. add the exact approved license text as a root `LICENSE` file;
2. update the `license` field in `pyproject.toml`;
3. add the matching SPDX identifier or license reference to `CITATION.cff` where valid;
4. replace the license warning in `README.md` with the approved terms;
5. state separately how code, documentation, figures, and frozen artifacts are covered
   if their terms differ; and
6. verify that release and archive metadata use the same license information.

Until those steps are complete, retain the all-rights-reserved status and do not imply
that the repository is open source.
