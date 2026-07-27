# CLAUDE.md — Project Operating Manual

> Read this first, every session. Rules here override your defaults.

## 1. State of the project

The dissertation is **complete and submitted for examination** (MPhil in Real Estate
Finance, Cambridge, 2026). This repository is now a **replication package in
maintenance mode**: the priority is preserving the integrity of recorded results, not
producing new ones. The headline findings (baseline reduce recall 0.000; framework
0.206 vs random floor 0.204; month-FE R² 0.43 vs firm features 0.007) are frozen in
`CANONICAL_RESULTS.md` — every number anywhere must reconcile to that ledger.

## 2. Repository topology (important)

- **`main`** — curated public history, force-pushed as a clean tree to
  `github.com/programming-thinker/auditable-reit-llm-framework`. Contains code,
  `audit_log/` evidence, `CANONICAL_RESULTS.md`, `REPLICATION_GUIDE.md`.
- **`archive/full-history`** — local-only branch pinning the complete development
  history. **Never push it.**
- **Local-only working files** (on disk, gitignored, absent from the public tree):
  `paper/` (thesis build chain), `THESIS_DRAFT.md`, `THESIS_SUPPLEMENT.md`, typeset
  PDFs, process records (`FORMAT_AUDIT_*`, `REFERENCE_VERIFICATION_*`,
  `CLAIM_CALIBRATION_*`, `REFERENCES.md`, …), `_legacy/`, `dissertation_past/`,
  `.claude/`. Do not re-add them to git without explicit instruction.
- Large data ships as **Release assets** (`v1.0-submission`), never as commits.

## 3. Hard rules (unchanged from the research phase)

1. **`audit_log/` is append-only.** Never overwrite or delete entries; the schema of
   `predictions.csv` is fixed (see `audit_log/SCHEMA.md`).
2. **Never commit files over 10 MB**, API keys, or `.env` files.
3. **Forbidden dependencies** (the fine-tuning path was explicitly rejected): `trl`,
   `peft`/`lora`/`qlora`, `unsloth`, `bitsandbytes`, `deepspeed`,
   `transformers.Trainer`. If a task seems to need them, stop and ask.
4. **Prompts are SHA-locked** (`config/config.yaml`, lock 2026-06-26). Never edit a
   prompt file in place; new versions get new files, and old ones stay for replay.
5. **`make reproduce_v6` must pass before any commit touching code or config.** If it
   fails after your change, revert.
6. Commit messages: `[ZONE-2|3] <imperative description>` (Zone 2 = scaffolding/docs
   /config, requires the regression gate; Zone 3 = new work).
7. Golden snapshots under `outputs/` and the processed panels under `data/` are
   results of record — treat as read-only; regeneration must reproduce them to 6 dp.
8. LLM API spend: dev-mode first (`make llm_dev_run`); the full test window is
   one-shot and lock-guarded. Track spend in `audit_log/cost_ledger.jsonl`.

## 4. Commands

```bash
make install         # pip install -r requirements.txt
make reproduce_v6    # golden-snapshot regression (must pass before commits)
make test            # unit tests; LLM tests use recorded responses
make lint            # ruff + black
make prompt_sha      # verify prompt SHAs against the config lock
make llm_dev_run     # 1 REIT × 2 months smoke test (needs API key)
```

Thesis rebuild (local only): `python3 paper/build.py && latexmk -cd -xelatex
paper/main.tex`; submission zips via `paper/make_overleaf_final.py` and
`paper/make_supplement_overleaf.py`.

## 5. Canonical documents

- `CANONICAL_RESULTS.md` — the number ledger (single source of truth).
- `REPLICATION_GUIDE.md` — submission contents, environment, exhibit→script→data map.
- `audit_log/SCHEMA.md` — decision-record schema.
- Local process records (verification rounds, format audits, claim calibration) live
  outside the public tree; consult them before re-litigating a settled question.

## 6. When in doubt

1. If a recorded result looks wrong → stop and ask; it may be load-bearing.
2. If `reproduce_v6` fails after your change → revert.
3. If you are reasoning about a "just this once" exception → stop and ask.
