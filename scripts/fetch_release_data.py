"""Download the published data of a release into the lake, so the report opens
and the data tests run without rebuilding anything.

    uv run python scripts/fetch_release_data.py

Two archives, both assets of the GitHub release named by RELEASE_TAG:

- gold-star-v2.zip: the star schema the report reads, written by dbt
  (data/lake/gold/star);
- gold-v1-reference.zip: the frozen v1 tables the parity test compares the
  star with (data/reference/v1). Every file is checked against the sha256 of
  the committed manifest, so a reference that does not match the repository
  is refused.

The model outputs (data/lake/gold/model_out) are in the repository already.
Raw sources are not distributed: rebuilding from them is the procedure of
docs/runbooks/materialise-v2.md. After the download:

    uv run pytest -q tests/test_gold_parity.py tests/test_powerbi_model.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, lake  # noqa: E402

REPOSITORY = "MasTristan/wildfire-eu-study"
RELEASE_TAG = "v2.0"
STAR_ASSET = "gold-star-v2.zip"
REFERENCE_ASSET = "gold-v1-reference.zip"
REFERENCE_DIR = Path(config.ROOT) / "data" / "reference" / "v1"


def asset_url(asset: str, tag: str = RELEASE_TAG) -> str:
    return f"https://github.com/{REPOSITORY}/releases/download/{tag}/{asset}"


def extract_parquets(archive: bytes, target: Path, expected_sha256: dict[str, str] | None = None) -> list[str]:
    """Write the Parquet members of a zip archive into target and return their names.

    Only flat `*.parquet` members are written: an archive with a path separator in a
    member name is refused, so nothing can land outside target. With expected_sha256
    (file name to digest), every listed file must be present and match, and the check
    happens before anything is written."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        members = [m for m in zf.namelist() if m.endswith(".parquet")]
        for name in zf.namelist():
            if "/" in name or "\\" in name or name.startswith("."):
                raise ValueError(f"unexpected member in archive: {name!r}")
        payload = {name: zf.read(name) for name in members}
    if expected_sha256 is not None:
        missing = sorted(set(expected_sha256) - set(payload))
        if missing:
            raise ValueError(f"archive lacks {missing}")
        for name, digest in expected_sha256.items():
            actual = hashlib.sha256(payload[name]).hexdigest()
            if actual != digest:
                raise ValueError(f"{name}: sha256 {actual[:12]} does not match the committed manifest ({digest[:12]})")
    target.mkdir(parents=True, exist_ok=True)
    for name, content in payload.items():
        (target / name).write_bytes(content)
    return sorted(payload)


def reference_digests(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {f"{name}.parquet": entry["sha256"] for name, entry in manifest["tables"].items()}


def _download(url: str) -> bytes:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (fixed https URL of this repository)
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tag", default=RELEASE_TAG, help="release tag to download from")
    args = parser.parse_args()
    star = extract_parquets(_download(asset_url(STAR_ASSET, args.tag)), lake.GOLD_STAR)
    print(f"{len(star)} tables in {lake.GOLD_STAR}")
    digests = reference_digests(REFERENCE_DIR / "manifest.json")
    reference = extract_parquets(_download(asset_url(REFERENCE_ASSET, args.tag)), REFERENCE_DIR, digests)
    print(f"{len(reference)} reference tables in {REFERENCE_DIR}, checked against the committed manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
