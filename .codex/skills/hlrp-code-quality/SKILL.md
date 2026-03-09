---
name: hlrp-code-quality
description: "Run and repair the HLRP code-quality workflow on the workstation by first resolving a working local Python runner, then executing `scripts/0_setup_environment.py`, `ruff check`, `black --check`, the full `pytest` suite, and the documented coverage run in sequence. If any step fails, fix the issue, rerun the same command until it passes, and only then continue. Use this whenever the user asks to check code quality, run QA, make validation pass, or fix lint, format, or test failures."
---

# HLRP Code Quality

Use this skill when the user wants code quality checked or fixed. Work from the repo root on the workstation and use the documented commands for this repo.

## Workflow

1. Verify execution context with `pwd` and `hostname`.
2. Resolve a Python runner before starting QA:
   - Prefer the active shell if `python` can run the repo tools successfully.
   - Otherwise inspect local repo docs and env hints such as `README.md`, `CLAUDE.md`, `requirements.txt`, and any available env files to find a suitable environment.
   - Treat documented env names as hints, not fixed requirements.
   - Set a single runner prefix for the session, such as `python` or `conda run -n <env> python`.
   - If no working runner can be identified locally, ask the user which environment to use.
3. Run the baseline QA steps in this order and do not skip ahead:
   - `<python-runner> scripts/0_setup_environment.py`
   - `<python-runner> -m ruff check packages/ scripts/ tests/`
   - `<python-runner> -m black --check packages/ scripts/ tests/`
   - `<python-runner> -m pytest tests/`
4. Add the documented coverage pass when the user asks for full QA, coverage, or the broadest available validation:
   - `<python-runner> -m pytest --cov=packages --cov-report=html tests/`
5. If a step fails, stop there, fix the issue, rerun the same command, and only continue once that command passes.
6. After code changes, rerun any earlier QA step that the fix may have invalidated before moving forward.
7. When the user explicitly says not to fix issues, stop after collecting the failures and report them.

## Fix Loop

- Run commands one after another, never in parallel.
- Treat each step as a gate. Do not move to the next command while the current one still fails.
- Prefer minimal fixes that address the reported failure directly.
- For `ruff`, prefer `<python-runner> -m ruff check --fix packages/ scripts/ tests/` when the changes are mechanical, then rerun plain `ruff check`.
- For `black`, run `<python-runner> -m black packages/ scripts/ tests/`, then rerun `black --check`.
- For `pytest` or `scripts/0_setup_environment.py`, make the required code or config fix, use targeted reruns while iterating if helpful, then rerun the full documented gate command before continuing.

## Rules

- Do not hard-code a conda env name. Verify the runner first and reuse that same runner throughout the QA pass.
- Keep to the repo-documented command shapes from `README.md` and `CLAUDE.md`.
- `scripts/0_setup_environment.py` is part of the QA sequence, not a substitute for tests.
- Coverage is an extra QA pass, not a substitute for the normal `pytest tests/` run.
- If a step is blocked by missing gated access or unavailable local dependencies, surface the blocker clearly instead of inventing a workaround.
- When the user asks for review-only or says not to fix issues, report findings and stop without making repairs.
