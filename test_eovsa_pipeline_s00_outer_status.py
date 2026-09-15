import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


_TEST_ROOT = Path(tempfile.gettempdir()) / "suncasa-eovsa-s00-status-tests"
for _name in (
    "EOVSAUDBMS",
    "EOVSAUDBMSSCL",
    "EOVSAUDBMSSLFCALED",
    "EOVSAUDB",
    "EOVSACAL",
    "EOVSASLFCAL",
    "EOVSAQLOOKFITS",
    "EOVSAQLOOKFIG",
    "EOVSASYNOPTICFIG",
    "EOVSA_WORKDIR",
):
    os.environ.setdefault(_name, str(_TEST_ROOT / _name.lower()))


sys.modules.setdefault(
    "suncasa.eovsa.eovsa_diskmodel",
    types.ModuleType("suncasa.eovsa.eovsa_diskmodel"),
)


eovsa_pipeline = importlib.import_module("suncasa.eovsa.eovsa_pipeline")


def _write_s00_fits(path, qa_state=None):
    header = fits.Header()
    header["CDELT1"] = 2.0
    if qa_state is not None:
        header["QASTATE"] = qa_state
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.zeros((4, 4)), header=header),
    ]).writeto(path)


@pytest.mark.parametrize(
    ("qa_state", "warning", "requires_review"),
    [
        ("PASS_BASE_MASK", False, False),
        ("PASS_EXPANDED_MASK", False, False),
        ("REVIEW_PSF_RESIDUAL", True, False),
        ("FAIL_COVERAGE", False, True),
        ("FAIL_MOTION", False, True),
        ("FAIL_PSF", False, True),
        ("FAIL_AMBIGUOUS_SOURCE", False, True),
        ("PASS_FAKE", False, True),
        ("FUTURE_STATE", False, True),
    ],
)
def test_s00_qastate_is_advisory_to_outer_state(
    tmp_path, qa_state, warning, requires_review
):
    s00_fits = tmp_path / "eovsa.synoptic_daily.20260722T200000Z.s00-01.tb.disk.fits"
    _write_s00_fits(s00_fits, qa_state)

    summary = eovsa_pipeline.summarize_s00_imaging_qa([str(s00_fits)])

    assert summary["s00_qa_state"] == qa_state
    assert summary["s00_qa_warning"] is warning
    assert summary["s00_qa_requires_review"] is requires_review
    state, status_fields = eovsa_pipeline.apply_s00_qa_to_pipeline_state(
        "success", [str(s00_fits)]
    )
    assert state == "success"
    assert status_fields.get("imaging_base_state") == (
        "success" if requires_review else None
    )


@pytest.mark.parametrize(
    "base_state",
    ["success", eovsa_pipeline.PROVISIONAL_SUCCESS_STATE],
)
def test_review_psf_residual_does_not_block_complete_run(tmp_path, base_state):
    s00_fits = tmp_path / "eovsa.synoptic_daily.20260722T200000Z.s00-01.tb.disk.fits"
    _write_s00_fits(s00_fits, "REVIEW_PSF_RESIDUAL")

    state, summary = eovsa_pipeline.apply_s00_qa_to_pipeline_state(
        base_state, [str(s00_fits)], require_qa=True
    )

    assert state == base_state
    assert summary["s00_qa_state"] == "REVIEW_PSF_RESIDUAL"
    assert summary["s00_qa_warning"] is True
    assert summary["s00_qa_requires_review"] is False
    assert summary["s00_qa_reason"] == "review_psf_residual"
    assert "imaging_base_state" not in summary


def test_unstamped_historical_s00_remains_backward_compatible(tmp_path):
    s00_fits = tmp_path / "eovsa.synoptic_daily.20220330T200000Z.s00-01.tb.disk.fits"
    _write_s00_fits(s00_fits)

    summary = eovsa_pipeline.summarize_s00_imaging_qa([str(s00_fits)])

    assert summary["s00_qa_state"] is None
    assert summary["s00_qa_warning"] is False
    assert summary["s00_qa_requires_review"] is False
    assert summary["s00_qa_reason"] == "qastate_not_present"


def test_unstamped_fresh_s00_requires_review_without_blocking(tmp_path):
    s00_fits = tmp_path / "eovsa.synoptic_daily.20260722T200000Z.s00-01.tb.disk.fits"
    _write_s00_fits(s00_fits)

    state, summary = eovsa_pipeline.apply_s00_qa_to_pipeline_state(
        "success", [str(s00_fits)], require_qa=True
    )

    assert state == "success"
    assert summary["s00_qa_state"] is None
    assert summary["s00_qa_warning"] is False
    assert summary["s00_qa_requires_review"] is True
    assert summary["s00_qa_reason"] == "qastate_not_present"
    assert summary["imaging_base_state"] == "success"


def test_unreadable_s00_requires_review(tmp_path):
    s00_fits = tmp_path / "eovsa.synoptic_daily.20260722T200000Z.s00-01.tb.disk.fits"
    s00_fits.write_text("not a FITS file")

    summary = eovsa_pipeline.summarize_s00_imaging_qa([str(s00_fits)])

    assert summary["s00_qa_state"] is None
    assert summary["s00_qa_warning"] is False
    assert summary["s00_qa_requires_review"] is True
    assert summary["s00_qa_reason"].startswith("s00_fits_read_failed:")


def test_partial_state_is_not_hidden_by_imaging_review(tmp_path):
    s00_fits = tmp_path / "eovsa.synoptic_daily.20260722T200000Z.s00-01.tb.disk.fits"
    _write_s00_fits(s00_fits, "FAIL_AMBIGUOUS_SOURCE")

    state, summary = eovsa_pipeline.apply_s00_qa_to_pipeline_state(
        "partial", [str(s00_fits)], require_qa=True
    )

    assert state == "partial"
    assert summary["s00_qa_warning"] is False
    assert summary["s00_qa_requires_review"] is True
    assert summary["imaging_base_state"] == "partial"
