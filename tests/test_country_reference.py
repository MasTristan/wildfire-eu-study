"""The EU flag of the country reference is frozen against the EU27 list."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "src" / "transform" / "reference" / "country_reference.csv"

EU27 = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU",
    "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
})
# Luxembourg is an EU member without any EFFIS record, so it has no row in the reference.
NOT_IN_REFERENCE = frozenset({"LU"})


def _reference() -> pd.DataFrame:
    return pd.read_csv(CSV, dtype=str, keep_default_na=False)


def test_is_eu_matches_the_eu27_list():
    df = _reference()
    flagged = set(df.loc[df["is_eu"] == "true", "country_key"])
    assert flagged == EU27 - NOT_IN_REFERENCE


def test_no_non_eu_country_is_flagged_and_macro_region_agrees():
    df = _reference()
    eu_region = set(df.loc[df["macro_region"] == "EU", "country_key"])
    assert eu_region == set(df.loc[df["is_eu"] == "true", "country_key"])
    assert not (set(df["country_key"]) & EU27 - set(df.loc[df["is_eu"] == "true", "country_key"]))


def test_germany_is_eu_and_not_mediterranean():
    row = _reference().set_index("country_key").loc["DE"]
    assert row["is_eu"] == "true"
    assert row["is_mediterranean"] == "false"
