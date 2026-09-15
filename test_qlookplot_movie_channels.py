"""Regression tests for per-channel qlook movie imaging."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from matplotlib import pyplot as plt
from astropy.time import Time

from suncasa.utils import qlookplot as ql


class SelectionLabelTest(unittest.TestCase):
    def test_channel_selection_has_sortable_filesystem_safe_label(self):
        self.assertEqual(ql.format_imaging_selection_label("0:0"), "00ch000")
        self.assertEqual(ql.format_imaging_selection_label("1:21"), "01ch021")
        self.assertEqual(ql.format_imaging_selection_label("0~1"), "00~01")


class MovieChannelMetadataTest(unittest.TestCase):
    def test_mk_qlook_image_accepts_channel_selections(self):
        selections = ["0:{}".format(channel) for channel in range(30)] + [
            "1:{}".format(channel) for channel in range(22)
        ]
        bandinfo = {
            "cfreqs": np.linspace(1.10, 1.73, len(selections)),
            "bounds_lo": np.linspace(1.09, 1.72, len(selections)),
            "bounds_hi": np.linspace(1.11, 1.74, len(selections)),
        }
        restoring_beams = np.linspace(63.6, 40.4, len(selections))
        metadata = mock.Mock()
        metadata.observatorynames.return_value = ["EOVSA"]
        ms = mock.Mock()
        ms.metadata.return_value = metadata
        ms.getspectralwindowinfo.return_value = {
            str(spw): {} for spw in range(50)
        }

        def fake_ptclean(**kwargs):
            return {
                "Succeeded": [True],
                "BeginTime": ["2025-03-28T15:50:01"],
                "EndTime": ["2025-03-28T15:50:02"],
                "ImageName": [kwargs["spw"] + ".fits"],
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            ql, "ms", ms
        ), mock.patch.object(
            ql.mstools, "get_bandinfo", return_value=bandinfo
        ) as get_bandinfo, mock.patch.object(
            ql.mstools, "get_bmsize", return_value=restoring_beams
        ) as get_bmsize, mock.patch.object(
            ql, "ptclean", side_effect=fake_ptclean
        ) as ptclean:
            imres, outfits = ql.mk_qlook_image(
                "input.ms",
                imagedir=directory,
                spws=selections,
                restoringbeam=[""],
                wrapfits=False,
                c_external=False,
            )

        get_bandinfo.assert_called_once_with(
            "input.ms", spw=selections, returnbdinfo=True
        )
        get_bmsize.assert_called_once_with(
            bandinfo["cfreqs"], refbmsize=70.0, reffreq=1.0, minbmsize=4.0
        )
        self.assertEqual(len(imres["Spw"]), 52)
        self.assertEqual(imres["Spw"][0], "00ch000")
        self.assertEqual(imres["Spw"][-1], "01ch021")
        self.assertEqual(imres["Freq"][0], [1.09, 1.11])
        self.assertEqual(imres["Freq"][-1], [1.72, 1.74])
        self.assertEqual(outfits, [])
        self.assertEqual([call.kwargs["spw"] for call in ptclean.call_args_list], selections)
        self.assertEqual(
            [call.kwargs["restoringbeam"] for call in ptclean.call_args_list],
            [["{:.1f}arcsec".format(beam)] for beam in restoring_beams],
        )

    def test_scalar_aia_download_uses_requested_passband(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            ql, "download_single_jp2", return_value="aia131.jp2"
        ) as download_single_jp2:
            downloaded = ql.download_aia_data(
                Time("2025-03-28T15:50:01"), wavelengths=131, outdir=directory
            )

        data_sources = download_single_jp2.call_args.args[3]
        self.assertEqual(data_sources[131], ql.DataSource.AIA_131)
        self.assertEqual(downloaded, ["aia131.jp2"])

    def test_read_imres_preserves_channel_order(self):
        raw = {
            "Succeeded": [True] * 4,
            "BeginTime": [
                "2025-03-28T15:50:01",
                "2025-03-28T15:50:02",
                "2025-03-28T15:50:01",
                "2025-03-28T15:50:02",
            ],
            "EndTime": [
                "2025-03-28T15:50:02",
                "2025-03-28T15:50:03",
                "2025-03-28T15:50:02",
                "2025-03-28T15:50:03",
            ],
            "ImageName": ["a0.fits", "a1.fits", "b0.fits", "b1.fits"],
            "Spw": ["00ch000", "00ch000", "01ch021", "01ch021"],
            "Vis": ["input.ms"] * 4,
            "Freq": [[1.09, 1.11], [1.09, 1.11], [1.72, 1.74], [1.72, 1.74]],
            "Obs": ["EOVSA"] * 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            imresfile = Path(directory) / "input.imres.npz"
            np.savez(imresfile, imres=raw)
            result = ql.read_imres(imresfile)

        self.assertEqual(result["spws"], ["00ch000", "01ch021"])
        self.assertEqual(
            result["images"].tolist(),
            [["a0.fits", "b0.fits"], ["a1.fits", "b1.fits"]],
        )


class MovieTimeSpanTest(unittest.TestCase):
    def test_rectangle_span_is_moved_to_current_movie_frame(self):
        figure, axis = plt.subplots()
        span = axis.axvspan(1.0, 2.0)

        ql._update_time_span(span, 3.0, 5.0)

        self.assertEqual(span.get_x(), 3.0)
        self.assertEqual(span.get_width(), 2.0)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
