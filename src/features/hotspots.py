"""Ignition label (b) from VIIRS detections: the first day of a hotspot
cluster. Generalises notebooks/spike_q1/s05_build_dataset.py::labels_viirs.

A detection day is clustered with DBSCAN on the sphere (eps in km, one
detection is enough to make a cluster); a cluster is NEW if no detection of
the previous day lies within eps of any of its points. New clusters are
counted per cell and day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import grid

EARTH_KM = 6371.0088


def filter_detections(raw: pd.DataFrame) -> pd.DataFrame:
    """Science-quality vegetation fires: type 0 when present, confidence not low."""
    df = raw
    if "type" in df.columns:
        df = df[df["type"] == 0]
    if "confidence" in df.columns:
        df = df[df["confidence"].astype(str) != "l"]
    return df.dropna(subset=["latitude", "longitude", "acq_date"])


def new_cluster_starts(raw: pd.DataFrame, eps_km: float = 2.0) -> pd.DataFrame:
    """(date, lat, lon) of each new cluster, lat/lon being the cluster centroid."""
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import BallTree

    det = filter_detections(raw).copy()
    det["date"] = pd.to_datetime(det["acq_date"]).dt.normalize()
    eps = eps_km / EARTH_KM
    rows = []
    prev_tree, prev_day = None, None
    for day, g in det.groupby("date", sort=True):
        pts = np.radians(g[["latitude", "longitude"]].to_numpy(dtype="float64"))
        lab = DBSCAN(eps=eps, min_samples=1, metric="haversine", algorithm="ball_tree").fit_predict(pts)
        if prev_tree is not None and (day - prev_day).days == 1:
            near = prev_tree.query_radius(pts, r=eps, count_only=True) > 0
        else:
            near = np.zeros(len(pts), dtype=bool)
        d = pd.DataFrame({"lab": lab, "near": near, "lat": g["latitude"].to_numpy(), "lon": g["longitude"].to_numpy()})
        cl = d.groupby("lab").agg(near=("near", "any"), lat=("lat", "mean"), lon=("lon", "mean"))
        new = cl[~cl["near"]]
        rows.append(pd.DataFrame({"date": day, "lat": new["lat"].to_numpy(), "lon": new["lon"].to_numpy()}))
        prev_tree, prev_day = BallTree(pts, metric="haversine"), day
    if not rows:
        return pd.DataFrame(columns=["date", "lat", "lon"])
    return pd.concat(rows, ignore_index=True)


def cluster_labels(raw: pd.DataFrame, cells: pd.DataFrame, eps_km: float = 2.0) -> pd.DataFrame:
    """(cell, date, n_start): new hotspot clusters per cell and day, cells of the mesh only."""
    starts = new_cluster_starts(raw, eps_km)
    if starts.empty:
        return pd.DataFrame(columns=["cell", "date", "n_start"])
    starts["cell"] = grid.cell_id(starts["lon"], starts["lat"])
    starts = starts[starts["cell"].isin(cells["cell"])]
    return starts.groupby(["cell", "date"]).size().rename("n_start").reset_index()
