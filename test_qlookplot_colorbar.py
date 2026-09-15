"""Regression tests for qlookplot frequency colorbar scaling."""

import unittest

import numpy as np
from matplotlib import colors

from suncasa.utils.qlookplot import (
    _resolve_radio_contour_levels,
    _uses_selected_frequency_range,
    get_colorbar_params,
    get_normalization,
)


class GetColorbarParamsTest(unittest.TestCase):
    def setUp(self):
        self.fbounds = {
            "cfreqs": np.array([1.10, 1.20, 1.40, 1.50]),
            "bounds_lo": np.array([1.10, 1.20, 1.40, 1.50]),
            "bounds_hi": np.array([1.10, 1.20, 1.40, 1.50]),
            "bounds_all": np.array([1.0, 2.0, 18.0]),
        }

    def test_channel_mode_uses_selected_frequency_range(self):
        ticks, bounds, fmax, fmin, freqmask = get_colorbar_params(
            self.fbounds, use_selected_range=True
        )

        self.assertEqual(fmin, 1.10)
        self.assertEqual(fmax, 1.50)
        self.assertEqual(bounds[0], fmin)
        self.assertEqual(bounds[-1], fmax)
        self.assertLessEqual(len(ticks), 4)
        self.assertNotIn(fmin, ticks)
        self.assertNotIn(fmax, ticks)
        self.assertEqual(freqmask, [])

    def test_spw_mode_keeps_full_instrument_range(self):
        _, _, fmax, fmin, _ = get_colorbar_params(self.fbounds)

        self.assertEqual(fmin, 1.0)
        self.assertEqual(fmax, 18.0)

    def test_movie_channel_labels_enable_selected_frequency_range(self):
        self.assertTrue(_uses_selected_frequency_range(["00ch000", "01ch021"]))
        self.assertTrue(_uses_selected_frequency_range(["0:0", "1:21"]))
        self.assertFalse(_uses_selected_frequency_range(["0", "1"]))


class GetNormalizationTest(unittest.TestCase):
    def test_existing_normalization_is_preserved_for_movie_plotting(self):
        configured = colors.LogNorm(vmin=0.1, vmax=100.0)

        result = get_normalization(1.0, 2.0, configured)

        self.assertIs(result, configured)


class RadioContourLevelsTest(unittest.TestCase):
    def test_imin_imax_override_relative_levels(self):
        levels = _resolve_radio_contour_levels(
            0,
            np.array([1.0e7, 2.0e8]),
            3,
            [0.98, 1.0],
            None,
            1.0e8,
            1.0e9,
        )

        np.testing.assert_allclose(levels, [1.0e8, 5.5e8, 1.0e9])

    def test_clevelsfix_override_imin_imax(self):
        levels = _resolve_radio_contour_levels(
            1,
            np.array([1.0e7, 2.0e8]),
            3,
            [0.98, 1.0],
            [np.array([1.0e8, 2.0e8]), np.array([3.0e8, 4.0e8])],
            1.0e8,
            1.0e9,
        )

        np.testing.assert_allclose(levels, [3.0e8, 4.0e8])

    def test_shared_clevelsfix_applies_to_every_frequency(self):
        levels = _resolve_radio_contour_levels(
            4,
            np.array([1.0e7, 2.0e8]),
            3,
            [0.98, 1.0],
            [1.0e8, 2.0e8],
            1.0e8,
            1.0e9,
        )

        np.testing.assert_allclose(levels, [1.0e8, 2.0e8])


if __name__ == "__main__":
    unittest.main()
