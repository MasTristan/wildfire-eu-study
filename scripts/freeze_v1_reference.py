"""Freeze the v1 gold layer as the parity reference for the v2 migration.

Run once, locally, on the v1 pipeline output (data/processed/*.parquet built
by the v1 code at tag v1.0):

    python scripts/freeze_v1_reference.py

It copies the gold parquets to data/reference/v1/, writes manifest.json
(row counts, columns, sha256 of each file) and zips the folder to
dist/gold-v1.zip for upload as a release asset on tag v1.0. The manifest is
small enough to commit; the parquets are not committed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, contracts  # noqa: E402

REFERENCE = Path(config.ROOT) / "data" / "reference" / "v1"
DIST = Path(config.ROOT) / "dist"


def main() -> int:
    REFERENCE.mkdir(parents=True, exist_ok=True)
    DIST.mkdir(exist_ok=True)
    tables: dict[str, dict] = {}
    for name in contracts.GOLD_TABLES:
        src = config.DATA_PROCESSED / f"{name}.parquet"
        if not src.exists():
            print(f"skip  {name}: not built")
            continue
        contracts.validate(name, pd.read_parquet(src))
        dst = REFERENCE / src.name
        shutil.copy2(src, dst)
        df = pd.read_parquet(dst)
        tables[name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        }
        print(f"froze {name}: {len(df):,} rows, {len(df.columns)} columns")
    manifest = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tag": "v1.0",
        "tables": tables,
    }
    (REFERENCE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = DIST / "gold-v1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(REFERENCE.iterdir()):
            zf.write(path, arcname=path.name)
    print(f"\n{len(tables)} tables frozen in {REFERENCE}\narchive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
