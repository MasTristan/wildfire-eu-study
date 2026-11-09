"""MLflow in a local SQLite store (ADR 0004): every model run is recorded
with its parameters, metrics and artefacts under mlruns/ (one SQLite file
for the metadata, a folder for the artefacts, no server), and every figure
or table published by the project carries the run id that produced it.
`mlflow ui --backend-store-uri <uri>` browses it when needed.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src import config

TRACKING_DIR = Path(os.getenv("MLFLOW_TRACKING_DIR", config.ROOT / "mlruns"))


def tracking_uri() -> str:
    if os.getenv("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{TRACKING_DIR.resolve() / 'mlflow.db'}"


def artifact_root() -> str:
    root = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", TRACKING_DIR / "artifacts"))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve().as_uri()


def _ensure_experiment(name: str) -> None:
    import mlflow

    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_root())
    mlflow.set_experiment(name)


@contextmanager
def run(experiment: str, run_name: str, params: dict[str, Any] | None = None, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Context manager yielding the active mlflow run; params are logged on entry."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    _ensure_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags) as active:
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield active


def mark_published(experiment: str, run_id: str) -> int:
    """Tag run_id published=true and every other run of the experiment published=false, so that
    exactly one run (the one that wrote the tables now in model_out) is published. Call it once
    the tables are written, so a failed run never takes the tag. Returns the runs demoted."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(f"unknown experiment {experiment!r}")
    client = mlflow.MlflowClient()
    others = [r.info.run_id for r in client.search_runs([exp.experiment_id], filter_string="tags.published = 'true'")
              if r.info.run_id != run_id]
    for rid in others:
        client.set_tag(rid, "published", "false")
    client.set_tag(run_id, "published", "true")
    return len(others)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    import mlflow

    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v}, step=step)


def log_table(df, name: str) -> None:
    """A DataFrame as a CSV artefact."""
    import mlflow

    mlflow.log_table(df, artifact_file=f"{name}.json")


def log_artifact(path: Path) -> None:
    import mlflow

    mlflow.log_artifact(str(path))


MODEL_CARD_TEMPLATE = """# Model card: {name}

Run: `{run_id}` (experiment `{experiment}`), {date}.

## Question
{question}

## Data
{data}

## Method
{method}

## Validation
{validation}

## Limits
{limits}

## Results
{results}

## Reproduce
{reproduce}
"""


def write_model_card(path: Path, **fields: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MODEL_CARD_TEMPLATE.format(**fields), encoding="utf-8")
    return path
