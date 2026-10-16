import pandas as pd

from src import grid
from src.features import hotspots


def _cells():
    df = pd.DataFrame({"lon": [-4.0, -3.75], "lat": [40.0, 40.0], "country_key": "ES"})
    df.insert(0, "cell", grid.cell_id(df["lon"], df["lat"]))
    return df


def test_new_clusters_are_counted_once_and_chained_days_are_not_new():
    # day 1: two detections 500 m apart (one cluster) near cell A, plus one far detection in cell B
    # day 2: the same fire moved 800 m (chained, not new) and a brand-new fire 30 km away in cell A
    # day 4: a detection after a one-day gap: new again
    raw = pd.DataFrame({
        "latitude":  [40.00, 40.0045, 40.0,  40.007, 40.27,  40.0],
        "longitude": [-4.00, -4.0000, -3.75, -4.000, -4.00,  -4.0],
        "acq_date":  ["2023-07-01", "2023-07-01", "2023-07-01", "2023-07-02", "2023-07-02", "2023-07-04"],
        "confidence": ["n", "h", "n", "n", "h", "n"],
        "type": [0, 0, 0, 0, 0, 0],
        "frp": [5.0] * 6,
    })
    starts = hotspots.new_cluster_starts(raw, eps_km=2.0)
    per_day = starts.groupby("date").size()
    assert per_day.loc["2023-07-01"] == 2
    assert per_day.loc["2023-07-02"] == 1
    assert per_day.loc["2023-07-04"] == 1
    labels = hotspots.cluster_labels(raw, _cells())
    a = grid.cell_id([-4.0], [40.0])[0]
    assert labels.set_index(["cell", "date"]).loc[(a, pd.Timestamp("2023-07-01")), "n_start"] == 1
    assert int(labels["n_start"].sum()) == 3  # the 40.27 one lies outside the two mesh cells


def test_low_confidence_and_non_vegetation_are_dropped():
    raw = pd.DataFrame({
        "latitude": [40.0, 40.1], "longitude": [-4.0, -4.1],
        "acq_date": ["2023-07-01", "2023-07-01"], "confidence": ["l", "n"], "type": [0, 2], "frp": [1.0, 1.0],
    })
    assert hotspots.filter_detections(raw).empty
