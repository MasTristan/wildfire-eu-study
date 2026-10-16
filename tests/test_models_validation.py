import numpy as np

from src.models import validation as v


def test_leave_one_year_out_and_block_folds_partition_rows():
    years = np.repeat([2019, 2020, 2021], 10)
    seen = []
    for train, test, held in v.leave_one_year_out(years):
        assert set(years[test]) == {held} and held not in set(years[train])
        seen.append(held)
    assert seen == [2019, 2020, 2021]
    lon = np.random.default_rng(0).uniform(-10, 30, 500)
    lat = np.random.default_rng(1).uniform(35, 60, 500)
    blocks = v.spatial_blocks(lon, lat, 2.0)
    covered = np.zeros(500, dtype=int)
    for train, test, _ in v.block_folds(blocks, k=5):
        assert not set(blocks[train]) & set(blocks[test])  # a block is never split
        covered[test] += 1
    assert (covered == 1).all()


def test_metrics_and_skill():
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=5000) < 0.05).astype(int)
    good = np.clip(0.05 + 0.3 * y + rng.normal(0, 0.05, 5000), 0.001, 0.999)
    base = np.full(5000, 0.05)
    m = v.metrics(y, good)
    assert 0 < m["base_rate"] < 0.1 and m["pr_auc"] > 0.5
    assert v.brier_skill(y, good, base) > 0
    rel = v.reliability(y, good)
    assert {"mean_pred", "obs_rate", "n"} <= set(rel.columns)


def test_block_bootstrap_delta_favours_the_better_model():
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=4000) < 0.05).astype(int)
    p_a = np.full(4000, 0.05)
    p_b = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.05, 4000), 0.001, 0.999)
    groups = np.arange(4000) // 50
    d = v.block_bootstrap_delta(y, p_a, p_b, groups, reps=100).set_index("quantity")
    assert d.loc["brier_rel_improvement", "p2.5"] > 0
    assert d.loc["pr_auc_gain", "p2.5"] > 0
