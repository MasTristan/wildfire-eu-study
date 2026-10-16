import numpy as np

from src import grid


def test_cell_id_roundtrip_and_snap():
    lon, lat = np.array([-4.0, 23.51, 180.0 - 0.25]), np.array([40.0, 38.26, -90.0])
    cell = grid.cell_id(lon, lat)
    back_lon, back_lat = grid.cell_centre(cell)
    assert np.allclose(back_lon, grid.snap(lon)) and np.allclose(back_lat, grid.snap(lat))
    assert len(set(cell.tolist())) == 3


def test_cell_ids_are_unique_over_the_globe():
    lons = np.arange(-180, 180, grid.GRID)
    lats = np.arange(-90, 90 + grid.GRID, grid.GRID)
    lon_g, lat_g = np.meshgrid(lons, lats)
    ids = grid.cell_id(lon_g.ravel(), lat_g.ravel())
    assert len(np.unique(ids)) == ids.size


def test_neighbours_are_the_3x3_block():
    c = grid.cell_id([0.0], [0.0])
    nb = grid.neighbours(c)
    lon, lat = grid.cell_centre(nb)
    assert len(nb) == 9
    assert set(np.round(lon, 2)) == {-0.25, 0.0, 0.25} and set(np.round(lat, 2)) == {-0.25, 0.0, 0.25}
