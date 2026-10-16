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
