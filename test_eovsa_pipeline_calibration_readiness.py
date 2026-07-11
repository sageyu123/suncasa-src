import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
from astropy.time import Time


_TEST_ROOT = Path(tempfile.gettempdir()) / "suncasa-eovsa-pipeline-tests"
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
    "EOVSAWORKDIR",
):
    os.environ.setdefault(_name, str(_TEST_ROOT / _name.lower()))


sys.modules.setdefault(
    "suncasa.eovsa.eovsa_diskmodel",
    types.ModuleType("suncasa.eovsa.eovsa_diskmodel"),
)


eovsa_pipeline = importlib.import_module("suncasa.eovsa.eovsa_pipeline")


def _provenance_record(vis, refcal_date="2026-07-08"):
    return {
        "vis": vis,
        "lookup_time_utc": "2026-07-10 07:00:00.000",
        "source": "sql_legacy_type8",
        "mode": "legacy",
        "sql_record_time_utc": f"{refcal_date} 07:12:00.000",
        "refcal_time_utc": f"{refcal_date} 12:52:53.000",
        "refcal_date_utc": refcal_date,
        "applied": True,
    }


def _run_smart_pipeline_case(
    monkeypatch,
    tmp_path,
    *,
    initial_complete,
    previous_status,
    readiness,
    applied_refcal_date,
    warning_update_count=None,
    fine_spectral_imaging=False,
):
    incomplete = {
        "statusfile": str(tmp_path / "status.json"),
        "fits_complete": False,
        "fits_count": 0,
        "fits_expected_count": 1,
        "existing_fitsfiles": [],
    }
    complete = {
        **incomplete,
        "fits_complete": True,
        "fits_count": 1,
        "existing_fitsfiles": [str(tmp_path / "band.fits")],
    }
    initial = complete if initial_complete else incomplete
    summaries = iter((initial, complete))
    captured = {
        "status_writes": [],
        "warning_dates": [],
        "provenance_lists": [],
        "overwrite_values": [],
    }

    monkeypatch.setattr(eovsa_pipeline, "workdir_default", str(tmp_path))
    monkeypatch.setattr(eovsa_pipeline, "udbmsslfcaleddir", str(tmp_path / "selfcal"))
    monkeypatch.setattr(eovsa_pipeline.os, "chdir", lambda _: None)
    monkeypatch.setattr(
        eovsa_pipeline,
        "summarize_synoptic_outputs",
        lambda *args, **kwargs: next(summaries),
    )
    monkeypatch.setattr(eovsa_pipeline, "read_pipeline_status", lambda _: previous_status)
    monkeypatch.setattr(eovsa_pipeline, "get_calibration_readiness", lambda _: readiness)

    def fake_calib_pipeline(*args, refcal_provenance=None, **kwargs):
        captured["provenance_lists"].append(refcal_provenance)
        captured["overwrite_values"].append(kwargs["overwrite"])
        refcal_provenance.append(
            _provenance_record(
                "UDB20260709135125.ms",
                refcal_date=applied_refcal_date,
            )
        )
        return {"imaged": True}

    monkeypatch.setattr(eovsa_pipeline, "calib_pipeline", fake_calib_pipeline)
    monkeypatch.setattr(
        eovsa_pipeline,
        "set_synoptic_calibration_warning",
        lambda fitsfiles, calibration_date=None: (
            captured["warning_dates"].append(calibration_date)
            or (len(fitsfiles) if warning_update_count is None else warning_update_count)
        ),
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "write_pipeline_status",
        lambda statusfile, state, **extra: (
            captured["status_writes"].append((state, extra)) or extra
        ),
    )

    captured["result"] = eovsa_pipeline.pipeline(
        year=2026,
        month=7,
        day=9,
        ndays=1,
        clearcache=False,
        doimport=True,
        version="v3.0",
        smart_cal_check=True,
        fine_spectral_imaging=fine_spectral_imaging,
    )
    return captured


def test_prior_day_refcal_does_not_count_as_same_day_ready(monkeypatch):
    prior_day_locator = Time("2026-07-08 07:12:00")
    prior_day_observation = Time("2026-07-08 12:52:53")

    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcalX",
        lambda _: {
            "timestamp": prior_day_locator,
            "t_bg": prior_day_observation,
        },
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bphsbdX",
        lambda _: {
            "timestamp": prior_day_locator,
            "t_refcal": prior_day_observation,
        },
    )
    monkeypatch.setattr(eovsa_pipeline, "sql2phacalX", lambda *args, **kwargs: [])

    readiness = eovsa_pipeline.get_calibration_readiness(Time("2026-07-09 20:00:00"))

    assert readiness["ready"] is False
    assert readiness["reason"] == "stale_refcal"
    assert readiness["refcal_timestamp_utc"] == prior_day_observation.iso


def test_stale_bphsbd_does_not_override_same_day_type8_refcal(monkeypatch):
    same_day_locator = Time("2026-07-09 07:12:00")
    same_day_observation = Time("2026-07-09 12:52:53")
    prior_day_locator = Time("2026-07-08 07:12:00")
    prior_day_observation = Time("2026-07-08 12:52:53")

    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcalX",
        lambda _: {
            "timestamp": same_day_locator,
            "t_bg": same_day_observation,
        },
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bphsbdX",
        lambda _: {
            "timestamp": prior_day_locator,
            "t_refcal": prior_day_observation,
        },
    )
    monkeypatch.setattr(eovsa_pipeline, "sql2phacalX", lambda *args, **kwargs: [])

    readiness = eovsa_pipeline.get_calibration_readiness(Time("2026-07-09 20:00:00"))

    assert readiness["ready"] is True
    assert readiness["reason"] == "ready_without_phacal"
    assert readiness["refcal_timestamp_utc"] == same_day_observation.iso


def test_same_day_bphsbd_is_ready_without_phacal(monkeypatch):
    same_day_locator = Time("2026-07-09 07:12:00")
    same_day_observation = Time("2026-07-09 12:52:53")

    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcalX",
        lambda _: {
            "timestamp": same_day_locator,
            "t_bg": same_day_observation,
        },
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bphsbdX",
        lambda _: {
            "timestamp": same_day_locator,
            "t_refcal": same_day_observation,
        },
    )
    monkeypatch.setattr(eovsa_pipeline, "sql2phacalX", lambda *args, **kwargs: [])

    readiness = eovsa_pipeline.get_calibration_readiness(Time("2026-07-09 20:00:00"))

    assert readiness["ready"] is True
    assert readiness["reason"] == "ready_without_phacal"
    assert readiness["phacal_warning"] == "missing_phacal"
    assert readiness["refcal_timestamp_utc"] == same_day_observation.iso


def test_same_day_locator_with_out_of_day_refcal_is_stale(monkeypatch):
    same_day_locator = Time("2026-07-05 07:12:00")
    out_of_day_observation = Time("2026-07-06 12:50:53")

    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcalX",
        lambda _: {
            "timestamp": same_day_locator,
            "t_bg": out_of_day_observation,
        },
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bphsbdX",
        lambda _: {
            "timestamp": same_day_locator,
            "t_refcal": out_of_day_observation,
        },
    )
    monkeypatch.setattr(eovsa_pipeline, "sql2phacalX", lambda *args, **kwargs: [])

    readiness = eovsa_pipeline.get_calibration_readiness(Time("2026-07-05 20:00:00"))

    assert readiness["ready"] is False
    assert readiness["reason"] == "stale_refcal"
    assert readiness["refcal_timestamp_utc"] == out_of_day_observation.iso


def test_fallback_lookup_is_inside_the_ready_prior_day(monkeypatch):
    monkeypatch.setattr(
        eovsa_pipeline,
        "get_calibration_readiness",
        lambda _: {"ready": True, "reason": "ready_without_phacal"},
    )

    fallback = eovsa_pipeline.find_previous_ready_calibration(
        Time("2026-07-09 20:00:00"),
        max_lookback_days=1,
    )

    prior_day = Time("2026-07-08 20:00:00")
    btime, etime = eovsa_pipeline.get_local_day_bounds(prior_day)
    lookup_time = Time(fallback["lookup_time_utc"])

    assert btime.mjd < lookup_time.mjd < etime.mjd
    assert lookup_time.mjd > Time("2026-07-08 07:12:00").mjd
    assert lookup_time.iso == Time(etime.mjd - 1.0 / 86400.0, format="mjd").iso


def test_prior_day_applied_provenance_requires_provisional_replacement():
    records = [
        _provenance_record("UDB20260709135125.ms"),
        _provenance_record("UDB20260709153026.ms"),
    ]

    summary = eovsa_pipeline.summarize_applied_refcal_provenance(records, "2026-07-09")

    assert summary == {
        "record_count": 2,
        "source": "sql_legacy_type8",
        "mode": "legacy",
        "sql_record_time_utc": "2026-07-08 07:12:00.000",
        "refcal_time_utc": "2026-07-08 12:52:53.000",
        "refcal_date_utc": "2026-07-08",
        "same_day": False,
    }


def test_same_day_applied_provenance_is_ordinary():
    summary = eovsa_pipeline.summarize_applied_refcal_provenance(
        [_provenance_record("UDB20260709135125.ms", refcal_date="2026-07-09")],
        "2026-07-09",
    )

    assert summary["refcal_date_utc"] == "2026-07-09"
    assert summary["same_day"] is True


def test_mixed_applied_provenance_fails_closed():
    records = [
        _provenance_record("UDB20260709135125.ms"),
        _provenance_record("UDB20260709153026.ms", refcal_date="2026-07-09"),
    ]

    with pytest.raises(ValueError, match="inconsistent applied refcal provenance"):
        eovsa_pipeline.summarize_applied_refcal_provenance(records, "2026-07-09")


def test_missing_applied_provenance_fails_closed():
    with pytest.raises(ValueError, match="missing applied refcal provenance"):
        eovsa_pipeline.classify_synoptic_refcal_provenance(
            [],
            "2026-07-09",
            fits_complete=True,
        )


def test_after_midnight_refcal_inside_local_day_is_same_day():
    record = _provenance_record("UDB20260709135125.ms", refcal_date="2026-07-10")
    record["sql_record_time_utc"] = "2026-07-09 07:12:00.000"
    record["refcal_time_utc"] = "2026-07-10 02:00:00.000"

    summary = eovsa_pipeline.summarize_applied_refcal_provenance(
        [record],
        "2026-07-09",
    )

    assert summary["refcal_date_utc"] == "2026-07-10"
    assert summary["same_day"] is True


def test_prior_day_provenance_classifies_complete_product_as_provisional():
    classification = eovsa_pipeline.classify_synoptic_refcal_provenance(
        [_provenance_record("UDB20260709135125.ms")],
        "2026-07-09",
        fits_complete=True,
    )

    assert classification["state"] == eovsa_pipeline.PROVISIONAL_SUCCESS_STATE
    assert classification["warning_calibration_date"] == "2026-07-08"
    assert classification["using_fallback_calibration"] is True
    assert classification["needs_same_day_calibration_rerun"] is True
    assert classification["refcal_timestamp_utc"] == "2026-07-08 12:52:53.000"
    assert classification["calibration_ready"] is False
    assert classification["same_day_calibration_ready"] is False
    assert classification["calibration_reason"] == "applied_fallback_refcal"


def test_same_day_rerun_classifies_complete_product_as_success():
    classification = eovsa_pipeline.classify_synoptic_refcal_provenance(
        [_provenance_record("UDB20260709135125.ms", refcal_date="2026-07-09")],
        "2026-07-09",
        fits_complete=True,
    )

    assert classification["state"] == "success"
    assert classification["warning_calibration_date"] is None
    assert classification["using_fallback_calibration"] is False
    assert classification["needs_same_day_calibration_rerun"] is False
    assert classification["calibration_ready"] is True
    assert classification["same_day_calibration_ready"] is True


def test_pipeline_prefers_applied_prior_day_provenance_over_ready_prediction(
    monkeypatch,
    tmp_path,
):
    captured = _run_smart_pipeline_case(
        monkeypatch,
        tmp_path,
        initial_complete=False,
        previous_status={},
        readiness={
            "ready": True,
            "reason": "ready_without_phacal",
            "deadline_expired": False,
            "refcal_timestamp_utc": "2026-07-08 12:52:53.000",
        },
        applied_refcal_date="2026-07-08",
    )

    assert captured["result"] == {"failed_dates": []}
    assert len(captured["provenance_lists"]) == 1
    assert isinstance(captured["provenance_lists"][0], list)
    assert captured["warning_dates"] == ["2026-07-08"]
    final_state, final_status = captured["status_writes"][-1]
    assert final_state == eovsa_pipeline.PROVISIONAL_SUCCESS_STATE
    assert final_status["using_fallback_calibration"] is True
    assert final_status["same_day_calibration_ready"] is False
    assert final_status["needs_same_day_calibration_rerun"] is True
    assert final_status["applied_refcal_date_utc"] == "2026-07-08"


def test_pipeline_reruns_provisional_product_with_same_day_provenance(
    monkeypatch,
    tmp_path,
):
    captured = _run_smart_pipeline_case(
        monkeypatch,
        tmp_path,
        initial_complete=True,
        previous_status={
            "state": eovsa_pipeline.PROVISIONAL_SUCCESS_STATE,
            "message": "Waiting for same-day calibration.",
            "using_fallback_calibration": True,
        },
        readiness={
            "ready": True,
            "reason": "ready_without_phacal",
            "deadline_expired": False,
            "refcal_timestamp_utc": "2026-07-09 12:52:53.000",
        },
        applied_refcal_date="2026-07-09",
    )

    assert captured["result"] == {"failed_dates": []}
    assert captured["overwrite_values"] == [True]
    assert captured["warning_dates"] == [None]
    final_state, final_status = captured["status_writes"][-1]
    assert final_state == "success"
    assert final_status["using_fallback_calibration"] is False
    assert final_status["same_day_calibration_ready"] is True
    assert final_status["needs_same_day_calibration_rerun"] is False
    assert final_status["applied_refcal_date_utc"] == "2026-07-09"


def test_fine_request_bypasses_standard_product_completion(monkeypatch, tmp_path):
    captured = _run_smart_pipeline_case(
        monkeypatch,
        tmp_path,
        initial_complete=True,
        previous_status={"state": "success"},
        readiness={
            "ready": True,
            "reason": "ready_without_phacal",
            "deadline_expired": False,
            "refcal_timestamp_utc": "2026-07-09 12:52:53.000",
        },
        applied_refcal_date="2026-07-09",
        fine_spectral_imaging=True,
    )

    assert captured["result"] == {"failed_dates": []}
    assert len(captured["provenance_lists"]) == 1


def test_parentless_fine_only_request_is_rejected_before_pipeline_work():
    with pytest.raises(ValueError, match="fine_spectral_only is no longer safe"):
        eovsa_pipeline.pipeline(
            year=2026,
            month=7,
            day=9,
            fine_spectral_only=True,
        )


def test_pipeline_fails_when_not_all_fits_receive_calibration_warning_metadata(
    monkeypatch,
    tmp_path,
):
    captured = _run_smart_pipeline_case(
        monkeypatch,
        tmp_path,
        initial_complete=False,
        previous_status={},
        readiness={
            "ready": True,
            "reason": "ready_without_phacal",
            "deadline_expired": False,
            "refcal_timestamp_utc": "2026-07-09 12:52:53.000",
        },
        applied_refcal_date="2026-07-09",
        warning_update_count=0,
    )

    assert captured["result"] == {"failed_dates": ["2026-07-09"]}
    states = [state for state, _ in captured["status_writes"]]
    assert states[-1] == "failed"
    assert "success" not in states
    assert eovsa_pipeline.PROVISIONAL_SUCCESS_STATE not in states
    final_status = captured["status_writes"][-1][1]
    assert final_status["calibration_warning_fits_count"] == 0
    assert final_status["calibration_warning_fits_expected_count"] == 1
    assert "updated calibration warning metadata for 0 of 1 FITS files" in final_status["error"]
