# Codex prompt template

Use this template for substantial implementation tasks in DANIDS-Research.

```text
Work on the current branch only. Read and obey AGENTS.md first, then the task specification I name below and the frozen research docs it references.

Task: <TASK FILE>

Complete the task end-to-end. Do not stop at a plan. Inspect the repository before editing, make the smallest coherent architecture that satisfies the task, add/update tests, run the relevant test/lint/type commands, and leave the branch PR-ready.

Do not silently change frozen research methodology. If you find an ambiguity that materially affects the science, make the conservative implementation choice that preserves leakage safety and document the ambiguity in your completion report rather than changing the research protocol.

Do not implement items explicitly marked out of scope just because they are easy.

At the end, provide a concise completion report with:
1. files changed,
2. key API/design decisions,
3. commands/tests run and results,
4. assumptions/ambiguities,
5. deferred out-of-scope work,
6. any risks I should review before committing/PR.
```

For small fixes, shorten the prompt but still tell Codex to read `AGENTS.md`, run relevant tests, and avoid changing scientific protocol.
