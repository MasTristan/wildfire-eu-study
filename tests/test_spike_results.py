"""Q1 spike results reshaped for the report: same numbers as VERDICT.md, no recomputation."""
import json

from src.models import spike_results as s


def test_metrics_carry_the_verdict_numbers_and_the_lift():
    m = s.metrics().set_index(["year", "model"])
    assert len(m) == 10
    assert round(m.loc[(2022, "lgbm_full"), "pr_auc"], 3) == 0.150
    assert round(m.loc[(2023, "lgbm_full"), "pr_auc"], 3) == 0.224
    assert abs(m.loc[(2022, "lgbm_full"), "lift"] - m.loc[(2022, "lgbm_full"), "pr_auc"] / m.loc[(2022, "lgbm_full"), "base_rate"]) < 1e-12
    assert list(m.loc[2022].sort_values("model_order").index) == list(s.MODELS)


def test_criterion_matches_the_frozen_json():
    c = s.criterion().set_index(["year", "quantity"])
    raw = json.loads((s.SPIKE_OUT / "criterion_a.json").read_text())
    assert round(c.loc[(2022, "brier_rel_improvement"), "value"], 3) == 0.079
    assert not c.loc[(2022, "brier_rel_improvement"), "meets_threshold"]
    assert c.loc[(2023, "brier_rel_improvement"), "meets_threshold"]
    assert bool(c["all_pass"].iloc[0]) is raw["all_pass"] is False
    assert (c["p2_5"] <= c["value"]).all() and (c["value"] <= c["p97_5"]).all()


def test_reliability_and_importance_are_complete():
    r = s.reliability()
    assert set(r["model"]) == set(s.MODELS) and set(r["year"]) == {2022, 2023}
    imp = s.importance()
    assert abs(imp["share"].sum() - 1) < 1e-9
    assert (imp["feature_group"] != "Other").all(), "every feature has a readable name"
    assert imp.iloc[0]["feature"] == "laga_7"
