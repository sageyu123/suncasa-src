import ast
import importlib
import inspect
import os
import sys
import tempfile
import types
from pathlib import Path

from astropy.time import Time


_TEST_ROOT = Path(tempfile.gettempdir()) / "suncasa-eovsa-output-completeness-tests"
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


FINE_SPWS_52BAND = [
    "0~1", "2", "3", "4", "5~6", "7~8", "9~10",
    "11~12", "13~14", "15~16", "17~18", "19~20",
    "21~22", "23~24", "25~26", "27~28", "29~30",
    "31~33", "34~35", "36~37", "38~39", "40~41", "42~43",
    "44~49",
]


def test_normal_mode_keeps_the_seven_default_groups():
    expected = eovsa_pipeline.get_expected_synoptic_spws(Time("2026-07-05 20:00"))

    assert expected == [
        "0~1", "2~4", "5~10", "11~20", "21~30", "31~43", "44~49"
    ]


def test_standalone_fine_plan_expects_valid_requested_groups_only():
    expected = eovsa_pipeline.get_expected_synoptic_spws(
        Time("2026-07-05 20:00"),
        custom_spws=FINE_SPWS_52BAND,
        fine_spectral_imaging=True,
    )

    assert len(expected) == 22
    assert "2~2" not in expected
    assert "3~3" not in expected
    assert expected[:3] == ["0~1", "4~4", "5~6"]
    assert expected[-1] == "44~49"


def test_standalone_custom_groups_are_normalized_and_deduplicated():
    expected = eovsa_pipeline.get_expected_synoptic_spws(
        Time("2026-07-05 20:00"),
        custom_spws=["5~6", "7~8", "5~6"],
    )

    assert expected == ["5~6", "7~8"]


def test_bootstrap_expects_parents_plus_valid_strict_children():
    expected = eovsa_pipeline.get_expected_synoptic_spws(
        Time("2026-07-05 20:00"),
        custom_spws=[
            "0~1", "2", "3", "4", "5~6", "5~6", "10~11", "44~49"
        ],
        fine_spectral_imaging=True,
        fine_spectral_bootstrap=True,
    )

    assert expected == [
        "0~1", "2~4", "5~10", "11~20", "21~30", "31~43", "44~49",
        "4~4", "5~6",
    ]


def test_exact_bootstrap_plan_expects_seven_parents_and_twenty_children():
    expected = eovsa_pipeline.get_expected_synoptic_spws(
        Time("2026-07-05 20:00"),
        custom_spws=FINE_SPWS_52BAND,
        fine_spectral_imaging=True,
        fine_spectral_bootstrap=True,
    )

    assert len(expected) == 27
    assert expected[:7] == [
        "0~1", "2~4", "5~10", "11~20", "21~30", "31~43", "44~49"
    ]
    assert "2~2" not in expected
    assert "3~3" not in expected
    assert expected[7:10] == ["4~4", "5~6", "7~8"]


def test_summary_uses_requested_groups_without_changing_status_filename(monkeypatch):
    tim = Time("2026-07-05 20:00")
    default_status = eovsa_pipeline.get_synoptic_output_info(tim)["statusfile"]
    monkeypatch.setattr(
        eovsa_pipeline.os.path,
        "exists",
        lambda path: ".s05-06." in path,
    )

    summary = eovsa_pipeline.summarize_synoptic_outputs(
        tim,
        expected_spws=["5~6", "7~8"],
    )

    assert summary["statusfile"] == default_status
    assert summary["fits_expected_count"] == 2
    assert summary["fits_count"] == 1
    assert summary["fits_complete"] is False
    assert [Path(path).name for path in summary["fitsfiles"]] == [
        "eovsa.synoptic_daily.20260705T200000Z.s05-06.tb.disk.fits",
        "eovsa.synoptic_daily.20260705T200000Z.s07-08.tb.disk.fits",
    ]


def test_pipeline_threads_expected_groups_to_every_summary_call():
    tree = ast.parse(inspect.getsource(eovsa_pipeline.pipeline))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "summarize_synoptic_outputs"
    ]

    assert calls
    assert all(
        any(keyword.arg == "expected_spws" for keyword in call.keywords)
        for call in calls
    )


def test_incomplete_explicit_custom_run_returns_failure(monkeypatch, tmp_path):
    summary_calls = []

    def incomplete_summary(*args, expected_spws=None, **kwargs):
        summary_calls.append(expected_spws)
        return {
            "statusfile": str(tmp_path / "status.json"),
            "fits_complete": False,
            "fits_count": 0,
            "fits_expected_count": len(expected_spws),
            "existing_fitsfiles": [],
        }

    monkeypatch.setattr(eovsa_pipeline, "workdir_default", str(tmp_path))
    monkeypatch.setattr(eovsa_pipeline.os, "chdir", lambda _: None)
    monkeypatch.setattr(eovsa_pipeline, "summarize_synoptic_outputs", incomplete_summary)
    monkeypatch.setattr(eovsa_pipeline, "calib_pipeline", lambda *args, **kwargs: {"imaged": True})

    result = eovsa_pipeline.pipeline(
        year=2026,
        month=7,
        day=5,
        ndays=1,
        clearcache=False,
        doimport=False,
        version="v3.0",
        debugging=True,
        smart_cal_check=False,
        custom_spws=["5~6", "7~8"],
    )

    assert result == {"failed_dates": ["2026-07-05"]}
    assert summary_calls == [["5~6", "7~8"], ["5~6", "7~8"]]


def test_smart_status_partial_explicit_run_returns_failure(monkeypatch, tmp_path):
    status_writes = []

    def incomplete_summary(*args, expected_spws=None, **kwargs):
        return {
            "statusfile": str(tmp_path / "status.json"),
            "fits_complete": False,
            "fits_count": 0,
            "fits_expected_count": len(expected_spws),
            "existing_fitsfiles": [],
        }

    def fake_calib_pipeline(*args, refcal_provenance=None, **kwargs):
        refcal_provenance.append({
            "vis": "UDB20260705135125.ms",
            "lookup_time_utc": "2026-07-05 07:00:00.000",
            "source": "sql_legacy_type8",
            "mode": "legacy",
            "sql_record_time_utc": "2026-07-05 07:12:00.000",
            "refcal_time_utc": "2026-07-05 12:52:53.000",
            "refcal_date_utc": "2026-07-05",
            "applied": True,
        })
        return {"imaged": True}

    monkeypatch.setattr(eovsa_pipeline, "workdir_default", str(tmp_path))
    monkeypatch.setattr(eovsa_pipeline, "udbmsslfcaleddir", str(tmp_path / "selfcal"))
    monkeypatch.setattr(eovsa_pipeline.os, "chdir", lambda _: None)
    monkeypatch.setattr(eovsa_pipeline, "summarize_synoptic_outputs", incomplete_summary)
    monkeypatch.setattr(eovsa_pipeline, "read_pipeline_status", lambda _: {})
    monkeypatch.setattr(
        eovsa_pipeline,
        "get_calibration_readiness",
        lambda _: {
            "ready": True,
            "reason": "ready_without_phacal",
            "deadline_expired": False,
            "refcal_timestamp_utc": "2026-07-05 12:52:53.000",
        },
    )
    monkeypatch.setattr(eovsa_pipeline, "calib_pipeline", fake_calib_pipeline)
    monkeypatch.setattr(
        eovsa_pipeline,
        "set_synoptic_calibration_warning",
        lambda fitsfiles, calibration_date=None: len(fitsfiles),
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "write_pipeline_status",
        lambda statusfile, state, **extra: status_writes.append((state, extra)),
    )

    result = eovsa_pipeline.pipeline(
        year=2026,
        month=7,
        day=5,
        ndays=1,
        clearcache=False,
        doimport=False,
        version="v3.0",
        debugging=True,
        smart_cal_check=True,
        custom_spws=["5~6", "7~8"],
    )

    assert result == {"failed_dates": ["2026-07-05"]}
    assert status_writes[-1][0] == "partial"
    assert status_writes[-1][1]["fits_expected_count"] == 2
