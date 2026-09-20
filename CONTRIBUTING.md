# Contributing

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv can install it).

```bash
uv sync --frozen
uv run python tasks.py check      # lint, typecheck, test, audit, baseline
```

If the repository sits on a small or slow drive, keep the virtual environment elsewhere:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/<repo>"
uv sync --frozen
```

## Tasks

`tasks.py` is the cross-platform task runner; the Makefile only delegates to it.

| Target | What it runs |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `format` | `ruff format` and `ruff check --fix` |
| `typecheck` | `mypy --strict` over src, tests, scripts, bench and tasks.py |
| `test` | pytest with branch coverage; fails below 85% |
| `bench` | micro-benchmarks and the end-to-end evaluation, both with regression thresholds |
| `demo` | the offline fault-injection demo (no API key) |
| `audit` | bandit and pip-audit |
| `baseline` | `scripts/check_baseline.py` |

## Rules for changes

- **Tests first for behavior.** Mark tests `unit` or `integration`. Use hypothesis for properties
  (invariants, round trips, oracles) rather than hand-picked examples alone.
- **Offline by default.** Tests, demo and bench must not touch the network or need
  `ANTHROPIC_API_KEY`; `tests/conftest.py` blocks non-loopback sockets. Use `FakeProvider` or
  `ReplayProvider` for model behavior.
- **Evidence for claims.** A new README claim needs a row in `CLAIMS.md` pointing to a test,
  bench script or CI job. Performance numbers must be copied from real `bench/` output together
  with the command that produced them.
- **Threat model.** New entry points, tools or artifacts need a STRIDE row in
  `docs/THREAT_MODEL.md` with a control and a test.
- **Decisions.** Record significant trade-offs as ADRs in `docs/adr/`.
- **No pickle.** Use JSON, npz with `allow_pickle=False`, safetensors or faiss `write_index`.
- **No secrets in git.** Configuration goes in `.env` (ignored); document new settings in
  `.env.example` (a test enforces that it lists every setting).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`).
