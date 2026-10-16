"""The common 0.25 degree grid (ADR 0003): the FWI lattice, one integer id
per cell, and the mesh of cells whose centre lies in a tracked country.

Every table of the feature store joins on `cell`, so the id must be a pure
function of the cell centre and identical wherever it is computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRID = 0.25
_LON_OFFSET, _LAT_OFFSET = 180.0, 90.0
_ROW = 10_000  # cells per lattice row in the id space (360 / 0.25 = 1440 < 10000)


def snap(values, step: float = GRID) -> np.ndarray:
    """Round coordinates to the lattice so grids from different sources join exactly."""
    return np.round(np.asarray(values, dtype="float64") / step) * step


def cell_id(lon, lat) -> np.ndarray:
    """Integer id of the cell whose centre is (lon, lat); inputs are snapped first."""
    ix = np.round((snap(lon) + _LON_OFFSET) / GRID).astype("int64")
    iy = np.round((snap(lat) + _LAT_OFFSET) / GRID).astype("int64")
    return iy * _ROW + ix


def cell_centre(cell) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of cell_id: (lon, lat) of the cell centre."""
    c = np.asarray(cell, dtype="int64")
    iy, ix = np.divmod(c, _ROW)
    return ix * GRID - _LON_OFFSET, iy * GRID - _LAT_OFFSET


def neighbours(cell, radius: int = 1) -> np.ndarray:
    """Ids of the (2r+1)^2 cells around each input cell, input included."""
    c = np.asarray(cell, dtype="int64")[:, None]
    d = np.arange(-radius, radius + 1)
    offsets = (d[:, None] * _ROW + d[None, :]).ravel()
    return (c + offsets[None, :]).ravel()


def mesh_from_lattice(lons, lats, countries: list[str] | None = None) -> pd.DataFrame:
    """Cells of a regular lattice whose centre falls in a tracked country.

    Columns: cell, lon, lat, country_key. Country attribution goes through
    src.ingest.geo_ref, the single entry point to the country polygons, so the
    feature store attributes the same cells to the same countries as the v1
    climate tables.
    """
    from src.ingest import geo_ref

    mask = geo_ref.grid_mask(np.sort(np.unique(snap(lons))), np.sort(np.unique(snap(lats))))
    if countries is not None:
        mask = mask[mask["country_key"].isin(countries)]
    out = pd.DataFrame({"lon": snap(mask["lon"]), "lat": snap(mask["lat"]), "country_key": mask["country_key"].to_numpy()})
    out.insert(0, "cell", cell_id(out["lon"], out["lat"]))
    return out.drop_duplicates("cell").sort_values("cell").reset_index(drop=True)
