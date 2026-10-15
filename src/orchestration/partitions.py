from __future__ import annotations

from datetime import date

from dagster import StaticPartitionsDefinition

from src import config

# One partition per year of the study. Static rather than time-window based:
# the sources are consolidated by calendar year (EFFIS stats, FWI reanalysis
# files, FIRMS archive), and a yearly key is what every v1 ingest already uses.
YEARS = [str(y) for y in range(config.BASE_YEAR_MIN, date.today().year + 1)]
yearly = StaticPartitionsDefinition(YEARS)
