"""dim_country and dim_commune are built from versioned snapshots, never from
the live APIs: a build must be reproducible for the v1 -> v2 parity test."""
from __future__ import annotations

import pandas as pd
import pytest

from src import contracts
from src.transform import dimensions


def test_build_dim_country_needs_no_network(monkeypatch):
    def _no_network(*args, **kwargs):
        raise AssertionError("dim_country must not call the network during a build")

    monkeypatch.setattr(dimensions.requests, "get", _no_network)
    df = dimensions.build_dim_country()
    contracts.validate("dim_country", df)
    ref = pd.read_csv(dimensions.REFERENCE_DIR / "country_reference.csv")
    assert len(df) == len(ref)
    snap = pd.read_csv(dimensions.WORLDBANK_SNAPSHOT).set_index("iso3")
    fr = df.set_index("iso3").loc["FRA"]
    assert fr["land_area_km2"] == snap.loc["FRA", "land_area_km2"]
    assert fr["forest_area_ha"] == pytest.approx(snap.loc["FRA", "forest_area_km2"] * 100)
    # a country the World Bank does not cover stays null, never imputed
    assert df.set_index("country_key").loc["XK", ["land_area_km2", "forest_area_ha"]].isna().all()


def test_snapshot_is_dated_and_covers_the_reference():
    snap = pd.read_csv(dimensions.WORLDBANK_SNAPSHOT)
    ref = pd.read_csv(dimensions.REFERENCE_DIR / "country_reference.csv")
    assert snap["iso3"].is_unique
    assert set(snap["iso3"]) <= set(ref["iso3"])
    assert snap["retrieved_on"].notna().all()
    assert (snap[["land_area_km2", "forest_area_km2"]].dropna() >= 0).all().all()
    missing = set(ref["iso3"]) - set(snap["iso3"])
    assert missing <= {"XKX"}, f"countries without a snapshot row: {sorted(missing)}"


def test_build_dim_commune_needs_no_network(monkeypatch):
    def _no_network(*args, **kwargs):
        raise AssertionError("dim_commune must not call the network during a build")

    monkeypatch.setattr(dimensions.requests, "get", _no_network)
    df = dimensions.build_dim_commune()
    contracts.validate("dim_commune", df)
    assert df["commune_key"].is_unique and len(df) > 30_000
    # INSEE codes are text: leading zeros and Corsican 2A/2B codes survive
    assert df["commune_key"].str.len().eq(5).all()
    assert df["commune_key"].str.startswith("0").any() and df["commune_key"].str.startswith("2A").any()
    assert "retrieved_on" not in df.columns
