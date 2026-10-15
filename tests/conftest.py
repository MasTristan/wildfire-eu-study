from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DBT_DIR = ROOT / "dbt"


def run_dbt(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run the dbt CLI of this environment against the project, capturing output."""
    full_env = {**os.environ, **(env or {})}
    cmd = [sys.executable, "-m", "dbt.cli.main", *args, "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    return subprocess.run(cmd, cwd=ROOT, env=full_env, capture_output=True, text=True)


@pytest.fixture(scope="session")
def dbt_manifest() -> Path:
    """`dbt parse` once per session so dagster-dbt can build its assets from the manifest."""
    res = run_dbt(["parse", "--no-partial-parse"])
    assert res.returncode == 0, res.stdout + res.stderr
    manifest = DBT_DIR / "target" / "manifest.json"
    assert manifest.exists()
    return manifest
