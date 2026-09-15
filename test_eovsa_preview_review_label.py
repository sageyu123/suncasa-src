"""Focused tests for review labels on EOVSA browser previews."""

import json
import sys
import types
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


try:
    from eovsapy import spw_config as _spw_config
except ImportError:
    try:
        import eovsapy as _eovsapy
    except ImportError:
        _eovsapy = types.ModuleType("eovsapy")
    _spw_config = types.ModuleType("eovsapy.spw_config")
    _spw_config.SPWS_34BAND = []
    _spw_config.SPWS_52BAND = []
    _spw_config.SPWS_52BAND_ALT = []
    _spw_config.SPW_EPOCH_SPLIT_DATE = "2021-01-01"
    _eovsapy.spw_config = _spw_config
    sys.modules["eovsapy"] = _eovsapy
    sys.modules["eovsapy.spw_config"] = _spw_config

from suncasa.eovsa import eovsa_pltQlookImage as qlook


def test_product_version_selectors_are_canonical():
    canonical = (
        "v1.0", "v2.0", "v2.0_alt", "v2.1", "v2.1_alt", "legacy_v2.0",
    )
    for version in canonical:
        assert qlook.normalize_product_version(version) == version

    for retired in ("v3.0", "v3.0_alt", "v3.1", "v3.1_alt"):
        with pytest.raises(ValueError):
            qlook.normalize_product_version(retired)


def test_product_and_status_paths_do_not_read_retired_v3_directories(tmp_path, monkeypatch):
    dateobj = datetime(2026, 7, 22)
    filename = "eovsa.synoptic_daily.20260722T200000Z.s00-01.tb.disk.fits"
    date_dir = tmp_path / "2026" / "07" / "22"
    (date_dir / "v3.0").mkdir(parents=True)
    (date_dir / "v3.0" / filename).write_bytes(b"retired")
    (date_dir / "eovsa.synoptic_pipeline_status.20260722.v3.0.json").write_text("{}")
    monkeypatch.setattr(qlook, "imgfitsdir", str(tmp_path))

    assert qlook.synoptic_product_path(dateobj, filename, version="v2.0") == str(
        date_dir / "v2.0" / filename
    )
    assert qlook.synoptic_pipeline_status_path(dateobj, version="v2.0") == str(
        date_dir / "eovsa.synoptic_pipeline_status.20260722.v2.0.json"
    )


def test_canonical_v2_filenames_keep_legacy_filename_explicit():
    dateobj = datetime(2026, 7, 22)
    assert qlook.synoptic_daily_product_filename(
        dateobj, "00-01", version="legacy_v2.0"
    ) == "eovsa.synoptic_daily.20260722T200000_UTC.s00-01.tb.fits"
    for version in ("v2.0", "v2.0_alt", "v2.1", "v2.1_alt"):
        assert qlook.synoptic_daily_product_filename(
            dateobj, "00-01", version=version
        ) == "eovsa.synoptic_daily.20260722T200000Z.s00-01.tb.disk.fits"


def test_fitsutils_does_not_fall_back_to_retired_v3_directory(tmp_path, monkeypatch):
    from suncasa.eovsa import eovsa_fitsutils as fitsutils

    dateobj = datetime(2026, 7, 22)
    date_dir = tmp_path / "2026" / "07" / "22"
    (date_dir / "v3.1").mkdir(parents=True)
    monkeypatch.setattr(fitsutils, "imgfitsdir", str(tmp_path))

    assert fitsutils.synoptic_product_dir(
        dateobj, version="v2.1", fallback=False
    ) == str(date_dir / "v2.1")
    assert fitsutils.synoptic_product_dir(
        dateobj, version="v2.1", fallback=True
    ) == str(date_dir)
    for retired in ("v3.0", "v3.0_alt", "v3.1", "v3.1_alt"):
        with pytest.raises(ValueError):
            fitsutils.normalize_product_version(retired)


def _write_status(tmp_path, status):
    statusfile = tmp_path / "2026" / "07" / "22" / "eovsa.synoptic_pipeline_status.20260722.v2.0.json"
    statusfile.parent.mkdir(parents=True)
    statusfile.write_text(json.dumps(status))


def test_review_label_uses_s00_state(monkeypatch, tmp_path):
    monkeypatch.setattr(qlook, "imgfitsdir", str(tmp_path))
    _write_status(tmp_path, {
        "state": "imaging_review_required",
        "s00_qa_state": "FAIL_COVERAGE",
        "s00_qa_reason": "fail_coverage",
        "s00_qa_requires_review": True,
    })

    assert qlook.eovsa_preview_review_label(
        datetime(2026, 7, 22), version="v2.0"
    ) == "REVIEW: FAIL_COVERAGE"


def test_review_label_includes_advisory_review_state(monkeypatch, tmp_path):
    monkeypatch.setattr(qlook, "imgfitsdir", str(tmp_path))
    _write_status(tmp_path, {
        "state": "success",
        "s00_qa_state": "REVIEW_PSF_RESIDUAL",
        "s00_qa_reason": "review_psf_residual",
        "s00_qa_warning": True,
        "s00_qa_requires_review": False,
    })

    assert qlook.eovsa_preview_review_label(
        datetime(2026, 7, 22), version="v2.0"
    ) == "REVIEW: REVIEW_PSF_RESIDUAL"


@pytest.mark.parametrize(
    "status",
    [
        {"state": "success", "s00_qa_state": "PASS_BASE_MASK"},
        {"state": "failed", "error": "imaging failed"},
    ],
)
def test_review_label_is_absent_without_review_metadata(monkeypatch, tmp_path, status):
    monkeypatch.setattr(qlook, "imgfitsdir", str(tmp_path))
    _write_status(tmp_path, status)

    assert qlook.eovsa_preview_review_label(
        datetime(2026, 7, 22), version="v2.0"
    ) == ""


def test_review_label_falls_back_to_reason_and_missing_file_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(qlook, "imgfitsdir", str(tmp_path))
    _write_status(tmp_path, {
        "state": "success",
        "s00_qa_requires_review": True,
        "s00_qa_reason": "s00_fits_read_failed: missing image HDU",
    })

    assert qlook.eovsa_preview_review_label(
        datetime(2026, 7, 22), version="v2.0"
    ) == "REVIEW: S00_FITS_READ_FAILED"
    assert qlook.eovsa_preview_review_label(
        datetime(2026, 7, 23), version="v2.0"
    ) == ""


def test_rendered_preview_contains_review_text(monkeypatch, tmp_path):
    class DummyMap:
        meta = {"CRVAL3": 1.0e9}
        date = datetime(2026, 7, 22)
        data = np.ones((2, 2))

    class DummySunmap:
        def imshow(self, axes, **kwargs):
            axes.imshow(DummyMap.data)

        def draw_limb(self, **kwargs):
            pass

        def draw_grid(self, **kwargs):
            pass

    monkeypatch.setattr(qlook.smap, "Map", lambda _: DummyMap())
    monkeypatch.setattr(qlook.pmX, "Sunmap", lambda _: DummySunmap())
    fig, ax = plt.subplots()
    try:
        qlook._render_eovsa_band_frame(
            "dummy.fits", ax, plt.get_cmap("viridis"), -1.0, 1.0,
            {}, str(tmp_path), lambda _: "unused.jpg", fig,
            review_label="REVIEW: FAIL_COVERAGE",
        )
        assert "REVIEW: FAIL_COVERAGE" in [text.get_text() for text in ax.texts]
    finally:
        plt.close(fig)


def test_v2_entrypoint_refreshes_existing_preview_for_review(monkeypatch, tmp_path):
    dateobj = datetime(2026, 7, 22)
    monkeypatch.setattr(qlook, "imgfitsdir", str(tmp_path / "fits"))
    monkeypatch.setattr(qlook, "pltfigdir", str(tmp_path / "browser"))
    _write_status(tmp_path / "fits", {
        "state": "imaging_review_required",
        "s00_qa_state": "FAIL_COVERAGE",
        "s00_qa_requires_review": True,
    })
    output_dir = qlook.synoptic_preview_dir(dateobj, version="v2.0", create=True)
    (Path(output_dir) / "t_eovsa_bd01.jpg").write_bytes(b"old")
    product = tmp_path / "product.fits"
    product.write_bytes(b"placeholder")
    monkeypatch.setattr(qlook, "synoptic_product_path", lambda *args, **kwargs: str(product))
    rendered = []
    monkeypatch.setattr(
        qlook,
        "_render_eovsa_band_frame",
        lambda *args, **kwargs: rendered.append(kwargs["review_label"]),
    )
    fig, ax = plt.subplots()
    try:
        qlook.pltEovsaQlookImage_v3(
            dateobj.strftime("%Y-%m-%d"), ["0~1"], [1.0], [-1.0], {"t": 32},
            fig=fig, ax=ax, overwrite=False, include_fine_bands=False,
        )
    finally:
        plt.close(fig)

    assert rendered == ["REVIEW: FAIL_COVERAGE"]
