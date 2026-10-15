# Developer entry points. Everything runs through uv so the lockfile is the
# single source of truth for versions (see pyproject.toml).

.PHONY: sync lint type test check validate freeze-v1 export-requirements

sync:            ## create or update the virtualenv from uv.lock
	uv sync

lint:            ## ruff, no auto-fix
	uv run ruff check src tests scripts

type:            ## mypy over src and tests
	uv run mypy

test:            ## pytest; data-dependent tests skip when parquets are absent
	uv run pytest

check: lint type test  ## what CI runs

validate:        ## pandera contracts over data/processed
	uv run python -m src.contracts

freeze-v1:       ## archive the v1 gold layer as the parity reference
	uv run python scripts/freeze_v1_reference.py

export-requirements:  ## keep requirements.txt in sync with uv.lock for pip users
	uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
