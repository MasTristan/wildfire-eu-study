# Developer entry points. Everything runs through uv so the lockfile is the
# single source of truth for versions (see pyproject.toml).

.PHONY: sync lint type test docs check validate fetch-data freeze-v1 export-requirements dbt-parse dbt-build dagster-dev dvc-status

sync:            ## create or update the virtualenv from uv.lock
	uv sync

lint:            ## ruff, no auto-fix
	uv run ruff check src tests scripts

type:            ## mypy over src and tests
	uv run mypy

test:            ## pytest; data-dependent tests skip when parquets are absent
	uv run pytest

docs:            ## build the site, failing on any broken link or missing page
	uv run mkdocs build --strict

check: lint type test docs  ## what CI runs

fetch-data:      ## download the published star schema and parity reference of the release
	uv run python scripts/fetch_release_data.py

validate:        ## pandera contracts over data/processed
	uv run python -m src.contracts

freeze-v1:       ## archive the v1 gold layer as the parity reference
	uv run python scripts/freeze_v1_reference.py

export-requirements:  ## keep requirements.txt in sync with uv.lock for pip users
	uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt

# dbt resolves the relative paths of profiles.yml and sources.yml (../data/...,
# target/wildfire.duckdb) against the directory it is launched from, so it runs
# from dbt/, as the Dagster dbt resource does.
dbt-parse:       ## regenerate dbt/target/manifest.json (what the Dagster dbt assets are built from)
	cd dbt && uv run --project .. dbt parse --project-dir . --profiles-dir .

dbt-build:       ## build the gold star schema from silver into data/lake/gold/star, with dbt tests
	cd dbt && uv run --project .. dbt build --project-dir . --profiles-dir .

dagster-dev:     ## Dagster UI on http://localhost:3000, state under .dagster/
	DAGSTER_HOME=$(CURDIR)/.dagster uv run dagster dev

# DVC runs as an isolated uv tool, outside the project environment: its
# dependency hydra-core pins antlr4 4.9, which breaks Dagster's asset selection.
DVC = uvx --from "dvc[s3]==3.67.1" dvc

dvc-status:      ## DVC cache vs remote (needs the remote credentials in .dvc/config.local)
	$(DVC) status -c
