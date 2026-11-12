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


def _mlflow_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path.resolve() / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))


def test_tables_carry_the_run_id():
    tables = s.all_tables(run_id="abc")
    assert set(tables) == {"q1_metrics", "q1_criterion", "q1_reliability", "q1_importance"}
    assert all((df["run_id"] == "abc").all() for df in tables.values())


def test_spike_run_logs_the_outputs_and_the_criterion(tmp_path, monkeypatch):
    _mlflow_env(tmp_path, monkeypatch)
    import mlflow

    run_id = s.log_spike_run()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.resolve() / 'mlflow.db'}")
    run = mlflow.get_run(run_id)
    raw = json.loads((s.SPIKE_OUT / "criterion_a.json").read_text())
    assert run.data.metrics["all_pass"] == 0.0
    assert abs(run.data.metrics["brier_rel_improvement_2022"] - raw["2022"]["brier_rel_improvement"]) < 1e-12
    assert run.data.metrics["passes_2023"] == 1.0
    assert mlflow.get_experiment(run.info.experiment_id).name == "q1_spike"
    logged = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "out")}
    assert {"out/criterion_a.json", "out/metrics_a.csv", "out/exploratory_recalibration.csv",
            "out/exploratory_no_winter.csv"} <= logged


def test_exploratory_e5_to_e8_outputs_follow_their_declared_rules():
    """E5 to E8 outputs are committed: the figures quoted in VERDICT.md come from them."""
    import pandas as pd

    out = s.SPIKE_OUT
    rec = pd.read_csv(out / "exploratory_recalibration.csv")
    full = rec[rec["model"] == "lgbm_full"].set_index("year")
    assert round(full.loc[2022, "brier_rel_gain_raw"], 3) == 0.079  # reproduces the verdict
    assert round(full.loc[2022, "brier_rel_gain_recal"], 3) == 0.082 and not full.loc[2022, "bar_met_recal"]
    nw = pd.read_csv(out / "exploratory_no_winter.csv")
    assert round(nw[nw["model"] == "lgbm_full"].set_index("year").loc[2023, "share_test_positives_removed"], 3) == 0.545
    blk = pd.read_csv(out / "exploratory_spatial_blocks_clim.csv").set_index(["year", "model"])
    assert blk.loc[(2022, "lgbm_full"), "transfers"] and not blk.loc[(2022, "lgbm_fwi"), "transfers"]
