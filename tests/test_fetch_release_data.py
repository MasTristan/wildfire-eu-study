"""The release download: only flat Parquet members are written, and a reference that
does not match the committed manifest is refused before anything touches the disk.
No network: the archives are built in memory."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_release_data as frd  # noqa: E402


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def test_parquet_members_are_written_and_the_manifest_is_left_alone(tmp_path):
    archive = _zip({"dim_a.parquet": b"a", "fact_b.parquet": b"bb", "manifest.json": b"{}"})
    assert frd.extract_parquets(archive, tmp_path / "star") == ["dim_a.parquet", "fact_b.parquet"]
    assert (tmp_path / "star" / "fact_b.parquet").read_bytes() == b"bb"
    assert not (tmp_path / "star" / "manifest.json").exists()


@pytest.mark.parametrize("name", ["../evil.parquet", "sub/dim_a.parquet", "..\\evil.parquet"])
def test_a_member_with_a_path_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match="unexpected member"):
        frd.extract_parquets(_zip({name: b"x"}), tmp_path / "star")
    assert not (tmp_path / "star").exists()


def test_a_reference_that_does_not_match_the_manifest_is_refused_before_writing(tmp_path):
    good = {"dim_a.parquet": hashlib.sha256(b"a").hexdigest()}
    assert frd.extract_parquets(_zip({"dim_a.parquet": b"a"}), tmp_path / "ok", good) == ["dim_a.parquet"]
    with pytest.raises(ValueError, match="does not match the committed manifest"):
        frd.extract_parquets(_zip({"dim_a.parquet": b"changed"}), tmp_path / "bad", good)
    with pytest.raises(ValueError, match="archive lacks"):
        frd.extract_parquets(_zip({"other.parquet": b"a"}), tmp_path / "bad", good)
    assert not (tmp_path / "bad").exists()


def test_digests_come_from_the_committed_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tables": {"dim_a": {"sha256": "abc"}, "fact_b": {"sha256": "def"}}}), encoding="utf-8")
    assert frd.reference_digests(manifest) == {"dim_a.parquet": "abc", "fact_b.parquet": "def"}
    committed = frd.reference_digests(frd.REFERENCE_DIR / "manifest.json")
    assert "dim_country.parquet" in committed and all(len(d) == 64 for d in committed.values())


def test_asset_urls_point_at_the_release_of_this_repository():
    url = frd.asset_url(frd.STAR_ASSET)
    assert url == "https://github.com/MasTristan/wildfire-eu-study/releases/download/v2.0/gold-star-v2.zip"
