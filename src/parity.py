"""Canonical form of a gold table for value-for-value comparison.

Used by the v1 -> v2 parity test and by the dbt-versus-Python equivalence
test. It removes representations that carry no information for Power BI
(datetime resolution, nullable-int versus float with NaN, object versus
string, row order) and keeps everything else.
"""
from __future__ import annotations

import pandas as pd


def canonical(df: pd.DataFrame, keys: tuple[str, ...], columns: list[str]) -> pd.DataFrame:
    out = df[columns].sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.astype("datetime64[ns]")
        elif pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("bool")
        elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            out[col] = s.astype("float64")
        else:
            out[col] = s.astype("object").where(s.notna(), None)
    return out
