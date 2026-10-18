import os
import subprocess
import sys

import batman
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sherlockpipe.exomoon import ExoMoonLeastSquares


def _planet_params():
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = 20.0
    params.rp = 0.08
    params.a = 12.0
    params.inc = 89.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.2, 0.1]
    params.limb_dark = 'quadratic'
    return params


def _emls(object_dir, time, flux, errors):
    return ExoMoonLeastSquares(
        str(object_dir), 1, 1.0, 1.0, [0.2, 0.1], 8.72, 0.8, 100.0, 20.0,
        0.0, 0.30, 12.0 / 215.032, 89.0, 0.5, 0.0, 0.3, 90.0, 0.0,
        0.3, 0.6, [0.0], time, flux, errors, 5.5, min_cadences_per_t0=3,
        skip_fold_fit=True, fit_rp_per_transit=False,
    )


def test_plot_false_writes_nothing_and_retains_numeric_diagnostics(tmp_path):
    time = np.arange(-0.65, 0.651, 0.01)
    errors = np.full_like(time, 2e-4)
    params = _planet_params()
    flux = batman.TransitModel(params, time).light_curve(params)
    originals = tuple(value.copy() for value in (time, flux, errors))
    plt.close('all')

    emls = _emls(tmp_path, time, flux, errors)
    result = emls.subtract_planet_transit(time, flux, errors, plot=False)

    assert not plt.get_fignums()
    assert list(tmp_path.iterdir()) == []
    assert emls.epoch_diagnostics[0]['accepted']
    assert np.isfinite(emls.epoch_diagnostics[0]['weighted_residual_scatter'])
    assert len(result[0]) == len(result[1]) == len(result[2])
    for actual, expected in zip((time, flux, errors), originals):
        np.testing.assert_array_equal(actual, expected)


def test_diagnostic_sorting_returns_a_detached_ranked_copy():
    rows = np.array([[5.0, 1.0, -1.0], [2.0, 2.0, -3.0], [3.0, 3.0, -2.0]])
    before = rows.copy()

    ranked = ExoMoonLeastSquares.sorted_diagnostic_copy(rows, 2)

    np.testing.assert_array_equal(rows, before)
    np.testing.assert_array_equal(ranked[:, 0], [2.0, 3.0, 5.0])
    ranked[0, 0] = -99.0
    np.testing.assert_array_equal(rows, before)
    with pytest.raises(ValueError, match='aligned'):
        ExoMoonLeastSquares.sorted_diagnostic_copy(rows, 3)


def test_catalog_geometry_constructs_koi_bounds_without_catalog_io():
    geometry = ExoMoonLeastSquares.catalog_planet_geometry(
        'kois', star_mass=1.0, star_radius=1.0, planet_period=30.0,
        planet_radius=8.0, planet_impact_param=0.2,
    )

    assert geometry['planet_semimajor_axis'] > 0
    assert 0 < geometry['planet_inc'] <= 90
    assert geometry['planet_inc_err'] == 0.5
    assert geometry['planet_impact_param'] == pytest.approx(0.2)
    assert geometry['min_radius'] == pytest.approx(0.4)
    assert geometry['max_radius'] == pytest.approx(4.0)
    with pytest.raises(ValueError, match='Catalog must'):
        ExoMoonLeastSquares.catalog_planet_geometry(
            'unknown', 1.0, 1.0, 30.0, 8.0, 0.2)


def test_basic_emls_import_does_not_load_optional_pandora():
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    script = '''
import importlib.abc
import sys

class BlockPandora(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pandoramoon":
            raise ImportError("Pandora must remain optional for basic EMLS imports")
        return None

sys.meta_path.insert(0, BlockPandora())
import sherlockpipe.exomoon
assert "pandoramoon" not in sys.modules
'''
    environment = os.environ.copy()
    environment['PYTHONPATH'] = source_root
    subprocess.run([sys.executable, '-c', script], check=True, env=environment)
