import pandas as pd

from src.models import tracking


def test_run_is_recorded_in_a_file_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path.resolve() / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    with tracking.run("test_exp", "run1", {"a": 1}) as active:
        tracking.log_metrics({"m": 0.5, "nan_skipped": float("nan")})
        tracking.log_table(pd.DataFrame({"x": [1, 2]}), "t")
        run_id = active.info.run_id
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.resolve() / 'mlflow.db'}")
    got = mlflow.get_run(run_id)
    assert got.data.params["a"] == "1" and got.data.metrics["m"] == 0.5 and "nan_skipped" not in got.data.metrics
    card = tracking.write_model_card(tmp_path / "card.md", name="t", run_id=run_id, experiment="test_exp", date="d",
                                     question="q", data="d", method="m", validation="v", results="r", limits="l", reproduce="p")
    assert run_id in card.read_text()


def test_model_card_states_limits_before_results(tmp_path):
    """Plan rule: the limit sentence is written before the result is shown."""
    headings = [line for line in tracking.MODEL_CARD_TEMPLATE.splitlines() if line.startswith("## ")]
    assert headings == ["## Question", "## Data", "## Method", "## Validation", "## Limits", "## Results", "## Reproduce"]
    card = tracking.write_model_card(tmp_path / "c.md", name="t", run_id="r", experiment="e", date="d", question="q",
                                     data="d", method="m", validation="v", limits="LIM", results="RES", reproduce="p")
    text = card.read_text(encoding="utf-8")
    assert text.index("LIM") < text.index("RES")
