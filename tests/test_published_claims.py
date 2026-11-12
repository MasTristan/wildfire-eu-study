"""Numbers quoted as plain text must still match the files they come from.

Two places quote results outside a table or a measure, where nothing recomputes them:

- the Validation page of the report states spike results (seeds passing, Brier gain
  with and without January to March) in a text box. They are checked against the
  frozen, committed spike outputs, so this part runs in CI;
- each model card names the MLflow run of the tables it describes. That is checked
  against the `run_id` column of the tables, where the model outputs are on disk.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src import lake

ROOT = Path(__file__).resolve().parents[1]
SPIKE_OUT = ROOT / "notebooks" / "spike_q1" / "out"
VALIDATION_PAGE = ROOT / "powerbi" / "WildfireEU.Report" / "definition" / "pages" / "pgmodelvalidation001"
CARDS = ROOT / "docs" / "models"

# card -> tables whose run_id the card must cite
CARD_TABLES = {
    "q1-ignition-spike.md": ["q1_metrics", "q1_criterion", "q1_importance", "q1_reliability"],
    "q2-tail-risk.md": ["tail_risk_country", "tail_risk_levels", "tail_risk_threshold_stability", "tail_risk_gof",
                        "tail_risk_episodes"],
    "q3-attribution.md": ["attribution_coefficients", "attribution_countryyear", "attribution_summary",
                          "attribution_exclusions", "attribution_bayes_coefficients",
                          "attribution_bayes_countryyear", "attribution_robustness"],
}


def _page_text(page: Path) -> str:
    """Every string of every visual of a report page, concatenated."""
    out: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for visual in sorted(page.glob("visuals/*/visual.json")):
        walk(json.loads(visual.read_text(encoding="utf-8")))
    return " ".join(out)


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def test_validation_page_quotes_the_frozen_spike_outputs():
    text = _page_text(VALIDATION_PAGE)
    seeds = pd.read_csv(SPIKE_OUT / "exploratory_seeds.csv")
    n_seeds = seeds["seed"].nunique()
    passing = seeds.groupby("year")["passes"].sum()
    for year in (2022, 2023):
        assert f"{int(passing[year])} seeds out of {n_seeds}" in text, f"seed count of {year} not quoted as computed"

    criterion = json.loads((SPIKE_OUT / "criterion_a.json").read_text(encoding="utf-8"))
    winter = pd.read_csv(SPIKE_OUT / "exploratory_no_winter.csv")
    full_2023 = winter[(winter["year"] == 2023) & (winter["model"] == "lgbm_full")]["brier_rel_improvement"].iloc[0]
    quoted = f"{_pct(criterion['2023']['brier_rel_improvement'])} to {_pct(full_2023)}"
    assert quoted in text, f"expected the page to say the 2023 Brier gain falls from {quoted}"


def test_validation_page_does_not_promise_a_forecast():
    """The spike predicts persistence and place from consolidated perimeters: no operational claim."""
    text = _page_text(VALIDATION_PAGE).lower()
    for phrase in ("next day's ignitions", "missed by one number"):
        assert phrase not in text


@pytest.mark.parametrize("card", sorted(CARD_TABLES))
def test_model_card_cites_the_run_of_its_tables(card: str):
    paths = [lake.GOLD_MODEL_OUT / f"{t}.parquet" for t in CARD_TABLES[card]]
    if not all(p.exists() for p in paths):
        pytest.skip("model outputs not on disk")
    text = (CARDS / card).read_text(encoding="utf-8")
    cited = set(re.findall(r"\b[0-9a-f]{32}\b", text))
    on_disk = {rid for p in paths for rid in pd.read_parquet(p, columns=["run_id"])["run_id"].unique()}
    assert on_disk <= cited, f"{card} does not cite the run of its tables: {sorted(on_disk - cited)}"
    assert cited <= on_disk, f"{card} cites runs that no table carries any more: {sorted(cited - on_disk)}"


def test_model_index_cites_only_current_runs():
    paths = [lake.GOLD_MODEL_OUT / f"{t}.parquet" for tables in CARD_TABLES.values() for t in tables]
    if not all(p.exists() for p in paths):
        pytest.skip("model outputs not on disk")
    cited = set(re.findall(r"\b[0-9a-f]{32}\b", (CARDS / "index.md").read_text(encoding="utf-8")))
    on_disk = {rid for p in paths for rid in pd.read_parquet(p, columns=["run_id"])["run_id"].unique()}
    assert cited <= on_disk, f"index cites stale runs: {sorted(cited - on_disk)}"
