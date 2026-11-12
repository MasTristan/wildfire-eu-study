"""The Power BI semantic model against its known failure modes and against the data.

Static checks (always run) guard the traps that make Desktop refuse to open
the model, which no linter catches: a duplicated lineageTag, and a `//`
comment between two TMDL blocks (comments are valid only inside a DAX body).

The data check runs where the gold is on disk: every column the model loads
from Parquet must have the Parquet type the model declares. The parity test
compares values in a canonical form and is blind to types; Desktop is not,
and silently rewrites the model when they disagree.
"""
from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import lake

MODEL = Path(__file__).resolve().parents[1] / "powerbi" / "WildfireEU.SemanticModel" / "definition"
TABLES = sorted((MODEL / "tables").glob("*.tmdl"))


def parse_table(path: Path) -> dict:
    """Columns loaded from the source (name, dataType, sourceColumn), the
    Parquet file the partition reads and the folder it reads it from."""
    text = path.read_text(encoding="utf-8")
    cols = []
    for m in re.finditer(r"^\tcolumn (.+?)\n((?:\t\t.*\n|\n)*)", text, re.M):
        head, body = m.group(1), m.group(2)
        if "=" in head:  # calculated column
            continue
        dtype = re.search(r"^\t\tdataType: (\S+)", body, re.M)
        src = re.search(r"^\t\tsourceColumn: (.+)$", body, re.M)
        if dtype and src:
            cols.append((head.strip("'"), dtype.group(1), src.group(1).strip()))
    file = re.search(r'\\\\?(?:model_out\\\\?)?([A-Za-z0-9_]+\.parquet)"', text)
    folder = "model_out" if "model_out" in text else "star"
    return {"columns": cols, "file": file.group(1) if file else None, "folder": folder, "text": text}


def test_lineage_tags_are_unique_across_the_model():
    tags: dict[str, str] = {}
    for path in sorted(MODEL.rglob("*.tmdl")):
        for tag in re.findall(r"^\s*lineageTag: (\S+)", path.read_text(encoding="utf-8"), re.M):
            assert tag not in tags, f"lineageTag {tag} in {path.name} and {tags[tag]}"
            tags[tag] = path.name


def test_no_comment_between_tmdl_blocks():
    """`//` at block indentation breaks opening (InvalidLineType: Other).
    `///` is a TMDL description attached to the next object and is valid."""
    for path in TABLES:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert not re.match(r"^\t{0,2}//(?!/)", line), f"{path.name}:{i}: // comment outside a DAX body"


def test_every_table_file_is_referenced_by_the_model():
    model = (MODEL / "model.tmdl").read_text(encoding="utf-8")
    for path in TABLES:
        name = path.stem
        ref = f"ref table '{name}'" if " " in name else f"ref table {name}"
        assert ref in model, f"{name} is not referenced in model.tmdl"


def _tmdl_type(t: pa.DataType) -> str:
    if pa.types.is_integer(t):
        return "int64"
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return "double"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    if pa.types.is_timestamp(t) or pa.types.is_date(t):
        return "dateTime"
    if pa.types.is_boolean(t):
        return "boolean"
    return str(t)


@pytest.mark.parametrize("path", TABLES, ids=lambda p: p.stem)
def test_declared_types_match_the_parquet(path: Path):
    spec = parse_table(path)
    if not spec["file"]:
        pytest.skip("not loaded from Parquet")
    folder = lake.GOLD_MODEL_OUT if spec["folder"] == "model_out" else lake.GOLD_STAR
    parquet = folder / spec["file"]
    if not parquet.exists():
        pytest.skip(f"{parquet} not on disk")
    schema = pq.read_schema(parquet)
    names = set(schema.names)
    for col, dtype, src in spec["columns"]:
        if src not in names:
            assert f'"{src}"' in spec["text"], f"{path.stem}.{col}: source column {src} neither in the file nor added by the query"
            continue
        actual = _tmdl_type(schema.field(src).type)
        assert actual == dtype, f"{path.stem}.{col}: model declares {dtype}, {parquet.name} holds {schema.field(src).type}"


@pytest.mark.parametrize("path", TABLES, ids=lambda p: p.stem)
def test_partition_reads_the_parquet_named_after_its_table(path: Path):
    """A copy-pasted partition that still reads another table's file loads the
    wrong data silently; the type test cannot see it when the columns overlap."""
    spec = parse_table(path)
    if spec["file"]:
        assert spec["file"] == f"{path.stem}.parquet", f"{path.stem} reads {spec['file']}"


def test_every_model_out_table_the_report_needs_is_in_the_model():
    names = {p.stem for p in TABLES}
    for needed in (
        "tail_risk_country", "tail_risk_levels", "tail_risk_threshold_stability", "tail_risk_gof",
        "attribution_coefficients", "attribution_summary", "attribution_robustness",
        "attribution_exclusions", "q1_criterion", "q1_importance", "q1_metrics",
    ):
        assert needed in names, f"{needed} is not in the semantic model"
    summary = next(p for p in TABLES if p.stem == "attribution_summary").read_text(encoding="utf-8")
    assert "r2_within_total" not in summary and "r2_oos_vs_country_mean" in summary
