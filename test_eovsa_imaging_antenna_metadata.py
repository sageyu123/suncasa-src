from astropy.io import fits
import json
import numpy as np

from suncasa.eovsa import eovsa_synoptic_imaging_pipeline_wsclean as pipeline


def test_imaging_fits_records_exact_unflagged_solar_antennas(tmp_path):
    fitsfile = tmp_path / "daily.fits"
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU()
    image.header["CDELT1"] = 2.0
    fits.HDUList([primary, image]).writeto(fitsfile)

    pipeline._set_imaging_antenna_keywords(
        str(fitsfile),
        n_ant_img=4,
        n_ant_total=15,
        antenna_ids=[1, 2, 7, 15],
    )

    with fits.open(fitsfile) as hdul:
        header = hdul[1].header
        assert header["NANTIMG"] == 4
        assert header["NANTTOT"] == 15
        assert header["ANTUSED"] == "1,2,7,15"


def test_imaging_antenna_ids_are_one_based_and_require_unflagged_data(monkeypatch):
    class FakeSubtable:
        def nrows(self):
            return 3

        def getcol(self, name, startrow, nread):
            values = {
                "ANTENNA1": np.array([0, 1, 14]),
                "ANTENNA2": np.array([1, 2, 0]),
                "FLAG": np.array([[[False, True, False]]]),
            }
            return values[name][..., startrow:startrow + nread]

        def close(self):
            return None

    class FakeTable:
        def open(self, _path):
            return None

        def query(self, _expression):
            return FakeSubtable()

        def close(self):
            return None

    monkeypatch.setattr(pipeline, "tb", FakeTable())
    monkeypatch.setattr(
        pipeline,
        "_data_description_ids_for_spws",
        lambda *_args: [0],
    )

    assert pipeline.unflagged_solar_antenna_ids("daily.ms", "0", 15) == [1, 2, 15]


def test_imaging_product_publishes_machine_readable_antenna_manifest(tmp_path):
    fitsfile = tmp_path / "daily.fits"
    fitsfile.write_bytes(b"product")

    manifest_path = pipeline._write_imaging_antenna_manifest(
        str(fitsfile),
        n_ant_total=15,
        antenna_ids=[1, 2, 7, 15],
    )

    assert manifest_path == str(fitsfile) + ".antennas.json"
    assert json.loads((tmp_path / "daily.fits.antennas.json").read_text()) == {
        "schema_version": 1,
        "source": "unflagged_imaging_input",
        "expected": 15,
        "used": [1, 2, 7, 15],
    }


def test_merged_daily_product_unions_interval_antenna_manifests(tmp_path):
    interval_a = tmp_path / "interval-a.fits"
    interval_b = tmp_path / "interval-b.fits"
    daily = tmp_path / "daily.fits"
    interval_a.write_bytes(b"interval-a")
    interval_b.write_bytes(b"interval-b")
    daily.write_bytes(b"daily")
    pipeline._write_imaging_antenna_manifest(
        str(interval_a),
        n_ant_total=15,
        antenna_ids=[1, 3, 15],
    )
    pipeline._write_imaging_antenna_manifest(
        str(interval_b),
        n_ant_total=15,
        antenna_ids=[1, 9, 15],
    )

    manifest_path = pipeline._promote_imaging_antenna_manifest(
        [str(interval_a), str(interval_b)], str(daily)
    )

    assert manifest_path == str(daily) + ".antennas.json"
    assert json.loads((tmp_path / "daily.fits.antennas.json").read_text()) == {
        "schema_version": 1,
        "source": "unflagged_imaging_input",
        "expected": 15,
        "used": [1, 3, 9, 15],
    }


def test_current_run_merge_inputs_exclude_stale_products(tmp_path):
    stale = tmp_path / "eovsa.synoptic.20260901T144840Z.s02-04.tb.fits"
    current = tmp_path / "eovsa.synoptic.20260901T144929Z.s02-04.tb.fits"
    stale.write_bytes(b"stale")
    current.write_bytes(b"current")

    selected = pipeline._synoptic_merge_inputs(
        1,
        {1: [str(current)]},
        str(tmp_path),
        "20260901",
        "02-04",
        merge_existing=False,
    )

    assert selected == [str(current)]


def test_merge_existing_mode_uses_matching_interval_products(tmp_path):
    first = tmp_path / "eovsa.synoptic.20260901T144840Z.s02-04.tb.fits"
    second = tmp_path / "eovsa.synoptic.20260902T010130Z.s02-04.tb.fits"
    unrelated = tmp_path / "eovsa.synoptic.20260901T144840Z.s05-10.tb.fits"
    for path in (first, second, unrelated):
        path.write_bytes(b"product")

    selected = pipeline._synoptic_merge_inputs(
        1,
        {},
        str(tmp_path),
        "20260901",
        "02-04",
        merge_existing=True,
    )

    assert selected == [str(first), str(second)]
