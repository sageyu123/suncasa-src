import importlib
import inspect
import os
import sys
import tarfile
import tempfile
import types
from pathlib import Path

import numpy as np
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
    "EOVSA_WORKDIR",
):
    os.environ.setdefault(_name, str(_TEST_ROOT / _name.lower()))


sys.modules.setdefault(
    "suncasa.eovsa.eovsa_diskmodel",
    types.ModuleType("suncasa.eovsa.eovsa_diskmodel"),
)


eovsa_pipeline = importlib.import_module("suncasa.eovsa.eovsa_pipeline")


@pytest.fixture(autouse=True)
def _empty_sql_type16_day(monkeypatch):
    """Keep historical readiness tests independent of the live SQL service."""

    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bpsX",
        lambda *_args, **_kwargs: [],
    )


def test_path_config_does_not_create_default_directories(monkeypatch, tmp_path):
    """Resolving fallback paths must not create a local data-tree mirror."""

    for name in (
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
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    config = eovsa_pipeline.Path_config(base_dir=".")

    assert config.udbdir == "./data1/eovsa/fits/UDB/"
    assert config.qlookfigdir == "./common/webplots/qlookimg_10m/"
    assert list(tmp_path.iterdir()) == []


def test_end_time_cutoff_stages_and_flags_only_copies(monkeypatch, tmp_path):
    source = tmp_path / "UDB20260710135125.ms"
    source.mkdir()
    (source / "table.dat").write_text("raw")
    cleared = []
    flagged = []

    monkeypatch.setattr(
        eovsa_pipeline,
        "clearcal",
        lambda **kwargs: cleared.append(kwargs),
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "flagdata",
        lambda **kwargs: flagged.append(kwargs),
    )

    staged = eovsa_pipeline.stage_time_cutoff_inputs(
        [str(source)], str(tmp_path / "work"), "2026-07-10T21:00:00"
    )

    assert source.exists()
    assert (source / "table.dat").read_text() == "raw"
    assert len(staged) == 1
    assert Path(staged[0]).exists()
    assert cleared == [{"vis": staged[0], "addmodel": False}]
    assert flagged == [{
        "vis": staged[0],
        "mode": "manual",
        "timerange": "2026/07/10/21:00:00~2026/07/11/21:00:00",
        "flagbackup": False,
    }]


def test_standard_calibration_stages_mutable_inputs_on_workdir(tmp_path):
    source = tmp_path / "raid" / "UDB20260710135125.ms"
    source.mkdir(parents=True)
    marker = source / "table.dat"
    marker.write_text("durable")

    staged = eovsa_pipeline.stage_calibration_inputs(
        [str(source)],
        str(tmp_path / "scratch"),
    )

    assert marker.read_text() == "durable"
    assert len(staged) == 1
    assert Path(staged[0]).is_dir()
    assert os.path.commonpath((staged[0], str(tmp_path / "scratch"))) == str(
        tmp_path / "scratch"
    )
    (Path(staged[0]) / "table.dat").write_text("calibration mutation")
    assert marker.read_text() == "durable"


def test_calibration_does_not_recopy_inputs_already_on_workdir(tmp_path):
    source = tmp_path / "scratch" / "imported_ms" / "UDB20260710135125.ms"
    source.mkdir(parents=True)

    staged = eovsa_pipeline.stage_calibration_inputs(
        [str(source)],
        str(tmp_path / "scratch"),
    )

    assert staged == [str(source)]


def test_observing_state_flags_preserve_imports_inside_workdir(tmp_path):
    workdir = tmp_path / 'scratch'
    source = workdir / 'imported_ms' / 'UDB20260905142224.ms'
    source.mkdir(parents=True)
    (source / 'table.dat').write_bytes(b'original import')
    staged = eovsa_pipeline.stage_calibration_inputs(
        [str(source)], str(workdir), preserve_imports=True)
    (Path(staged[0]) / 'table.dat').write_bytes(b'flagged working copy')
    assert (source / 'table.dat').read_bytes() == b'original import'
    with pytest.raises(ValueError, match='preserved input'):
        eovsa_pipeline.stage_calibration_inputs(
            staged, str(workdir), preserve_imports=True)
    assert (Path(staged[0]) / 'table.dat').read_bytes() == b'flagged working copy'


def test_archived_import_is_unpacked_only_into_mutable_stage(tmp_path):
    source = tmp_path / 'source' / 'UDB20260905142224.ms'
    source.mkdir(parents=True)
    (source / 'table.dat').write_bytes(b'original')
    archive = tmp_path / 'UDB20260905142224.ms.tar.gz'
    with tarfile.open(archive, 'w:gz') as out:
        out.add(source, arcname=source.name)
    archive_bytes = archive.read_bytes()
    staged = eovsa_pipeline.stage_calibration_inputs(
        [str(archive)], str(tmp_path / 'scratch'), preserve_imports=True)
    assert Path(staged[0]).name == source.name
    (Path(staged[0]) / 'table.dat').write_bytes(b'flagged')
    assert archive.read_bytes() == archive_bytes
    assert (source / 'table.dat').read_bytes() == b'original'


def test_new_scan_ms_import_is_written_to_processing_workdir(monkeypatch, tmp_path):
    durable_ms = tmp_path / "raid" / "UDBms"
    raw_udb = tmp_path / "raid" / "UDB"
    scratch_import = tmp_path / "scratch" / "imported_ms"
    scan_name = "UDB20260710120000"
    imported_prefixes = []

    monkeypatch.setattr(eovsa_pipeline, "udbmsdir", str(durable_ms) + "/")
    monkeypatch.setattr(eovsa_pipeline, "udbdir", str(raw_udb) + "/")
    monkeypatch.setattr(
        eovsa_pipeline,
        "findfiles",
        lambda *_args, **_kwargs: {
            "scanlist": [str(raw_udb / "2026" / scan_name)],
            "tstlist": [],
            "tedlist": [],
        },
    )

    def fake_importeovsa(*_args, **kwargs):
        prefix = kwargs["visprefix"]
        imported_prefixes.append(prefix)
        Path(prefix + scan_name + ".ms").mkdir(parents=True)

    monkeypatch.setattr(eovsa_pipeline, "importeovsa", fake_importeovsa)

    result = eovsa_pipeline.trange2ms(
        trange=Time(["2026-07-10 12:00:00", "2026-07-10 13:00:00"]),
        doimport=True,
        import_outpath=str(scratch_import),
    )

    assert imported_prefixes == [str(scratch_import) + "/"]
    assert result["ms"] == [str(scratch_import / (scan_name + ".ms"))]
    assert not (durable_ms / "202607" / (scan_name + ".ms")).exists()

    # A later calibration attempt must discover this same cached import
    # before deciding whether importeovsa is necessary.
    repeated = eovsa_pipeline.trange2ms(
        trange=Time(["2026-07-10 12:00:00", "2026-07-10 13:00:00"]),
        doimport=True, overwrite=True, prefer_scan_ms=True,
        import_outpath=str(scratch_import),
    )
    assert imported_prefixes == [str(scratch_import) + "/"]
    assert repeated['ms'] == result['ms']

    new_scan = 'UDB20260710123000'
    monkeypatch.setattr(eovsa_pipeline, 'findfiles', lambda *_args, **_kwargs: {
        'scanlist': [scan_name, new_scan], 'tstlist': [], 'tedlist': []})
    imported_sources = []

    def import_only_new_scan(**kwargs):
        imported_sources.extend(kwargs['idbfiles'])
        Path(kwargs['visprefix'] + new_scan + '.ms').mkdir(parents=True)

    monkeypatch.setattr(eovsa_pipeline, 'importeovsa', import_only_new_scan)
    expanded = eovsa_pipeline.trange2ms(
        trange=Time(['2026-07-10 12:00:00', '2026-07-10 13:00:00']),
        doimport=True, overwrite=True, prefer_scan_ms=True,
        import_outpath=str(scratch_import))
    assert imported_sources == [str(raw_udb / '2026' / new_scan)]
    assert len(expanded['ms']) == 2


def test_end_time_cutoff_skips_scans_without_post_cutoff_data(monkeypatch, tmp_path):
    source = tmp_path / "UDB20260710135125.ms"
    source.mkdir()

    monkeypatch.setattr(eovsa_pipeline, "clearcal", lambda **kwargs: None)

    def no_selected_rows(**kwargs):
        raise RuntimeError("MSSelectionNullSelection : The selected table has zero rows.")

    monkeypatch.setattr(eovsa_pipeline, "flagdata", no_selected_rows)

    staged = eovsa_pipeline.stage_time_cutoff_inputs(
        [str(source)], str(tmp_path / "work"), "2026-07-10T21:00:00"
    )

    assert len(staged) == 1
    assert Path(staged[0]).exists()


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
    overwrite=False,
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
    monkeypatch.setattr(
        eovsa_pipeline,
        "get_calibration_readiness",
        lambda *_args, **_kwargs: readiness,
    )

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
        version="v2.0",
        smart_cal_check=True,
        fine_spectral_imaging=fine_spectral_imaging,
        overwrite=overwrite,
    )
    return captured


def test_imaging_pipeline_defaults_to_complete_bph_sbd_bps_sql_family():
    """Public imaging entry points must default to the fail-closed SQL family."""

    assert (
        inspect.signature(eovsa_pipeline.calib_pipeline)
        .parameters["refcal_sql_mode"]
        .default
        == "bph_sbd"
    )
    assert (
        inspect.signature(eovsa_pipeline.pipeline)
        .parameters["refcal_sql_mode"]
        .default
        == "bph_sbd"
    )


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
        lambda *_args, **_kwargs: {
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
        lambda *_args, **_kwargs: {
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
        lambda *_args, **_kwargs: {
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
        lambda *_args, **_kwargs: {
            "timestamp": same_day_locator,
            "t_refcal": out_of_day_observation,
        },
    )
    monkeypatch.setattr(eovsa_pipeline, "sql2phacalX", lambda *args, **kwargs: [])

    readiness = eovsa_pipeline.get_calibration_readiness(Time("2026-07-05 20:00:00"))

    assert readiness["ready"] is False
    assert readiness["reason"] == "stale_refcal"
    assert readiness["refcal_timestamp_utc"] == out_of_day_observation.iso


def _sql_refcal_family_records(
        locator, refcal_time, type14_digest, type16_parent_digest=None):
    """Build one compact decoded type-14/type-16 family for readiness tests."""

    shape = (2, 2, 1)
    type14 = {
        "bph_rad": np.zeros(shape, dtype=np.float64),
        "sbd_ns": np.zeros(shape, dtype=np.float64),
        "flag": np.zeros(shape, dtype=np.int32),
        "timestamp": locator,
        "t_refcal": refcal_time,
        "type14_buffer_digest": type14_digest,
    }
    empty_group = {
        "frequency_ghz": np.zeros(0, dtype=np.float64),
        "band": np.zeros(0, dtype=np.int32),
        "phase_rad": np.zeros((2, 2, 0), dtype=np.float64),
        "channel_valid": np.zeros((2, 2, 0), dtype=np.uint8),
    }
    type16 = {
        "timestamp": locator,
        "t_refcal": refcal_time,
        "t_bphsbd": locator,
        "product_digest": "b" * 64,
        "type14_digest": (
            type14_digest
            if type16_parent_digest is None
            else type16_parent_digest
        ),
        "bph_ref_frequency_ghz": np.array([1.05], dtype=np.float64),
        "bps_candidate_code": np.zeros(shape, dtype=np.uint8),
        "bps_authorized": np.zeros(shape, dtype=np.uint8),
        "bps_primary": empty_group,
        "bps_secondary": dict(empty_group),
    }
    return type14, type16


def test_bph_sbd_readiness_requires_complete_digest_matched_family(monkeypatch):
    """A same-day parent plus mismatched type 16 is not full-route ready."""

    locator = Time("2026-07-09 07:12:00")
    refcal_time = Time("2026-07-09 12:52:53")
    type14, type16 = _sql_refcal_family_records(
        locator,
        refcal_time,
        "a" * 64,
        type16_parent_digest="c" * 64,
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcalX",
        lambda *_args, **_kwargs: {
            "timestamp": locator,
            "t_bg": refcal_time,
        },
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bphsbdX",
        lambda *_args, **_kwargs: [type14],
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bpsX",
        lambda *_args, **_kwargs: [type16],
    )

    readiness = eovsa_pipeline.get_calibration_readiness(
        Time("2026-07-09 20:00:00"),
        refcal_sql_mode="bph_sbd",
    )

    assert readiness["ready"] is False
    assert readiness["reason"] == "incomplete_refcal_family"
    assert readiness["refcal_family_status"] == "partial"

    auto_readiness = eovsa_pipeline.get_calibration_readiness(
        Time("2026-07-09 20:00:00"),
        refcal_sql_mode="auto",
    )
    assert auto_readiness["ready"] is False
    assert auto_readiness["reason"] == "incomplete_refcal_family"


def test_bph_sbd_readiness_searches_past_newer_partial_family(monkeypatch):
    """An older complete same-day family remains ready after a partial retry."""

    older_locator = Time("2026-07-09 07:12:00")
    older_refcal_time = Time("2026-07-09 12:52:53")
    older_type14, older_type16 = _sql_refcal_family_records(
        older_locator,
        older_refcal_time,
        "a" * 64,
    )
    newer_locator = Time("2026-07-09 08:12:00")
    newer_type14, newer_type16 = _sql_refcal_family_records(
        newer_locator,
        Time("2026-07-09 13:52:53"),
        "d" * 64,
        type16_parent_digest="e" * 64,
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcalX",
        lambda *_args, **_kwargs: {
            "timestamp": newer_locator,
            "t_bg": Time("2026-07-09 13:52:53"),
        },
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bphsbdX",
        lambda *_args, **_kwargs: [older_type14, newer_type14],
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bpsX",
        lambda *_args, **_kwargs: [older_type16, newer_type16],
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2phacalX",
        lambda *_args, **_kwargs: [],
    )

    readiness = eovsa_pipeline.get_calibration_readiness(
        Time("2026-07-09 20:00:00"),
        refcal_sql_mode="bph_sbd",
    )

    assert readiness["ready"] is True
    assert readiness["reason"] == "ready_without_phacal"
    assert readiness["refcal_family_status"] == "complete"
    assert readiness["refcal_timestamp_utc"] == older_refcal_time.iso


def test_explicit_legacy_readiness_ignores_partial_companion_family(monkeypatch):
    """Legacy readiness remains anchored to the same-day type-8 record."""

    locator = Time("2026-07-09 07:12:00")
    refcal_time = Time("2026-07-09 12:52:53")
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcalX",
        lambda *_args, **_kwargs: {
            "timestamp": locator,
            "t_bg": refcal_time,
        },
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2refcal_bphsbdX",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy readiness must not query type 14")
        ),
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "sql2phacalX",
        lambda *_args, **_kwargs: [],
    )

    readiness = eovsa_pipeline.get_calibration_readiness(
        Time("2026-07-09 20:00:00"),
        refcal_sql_mode="legacy",
    )

    assert readiness["ready"] is True
    assert readiness["reason"] == "ready_without_phacal"
    assert readiness["refcal_family_status"] == "legacy_type8"


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


@pytest.mark.parametrize('overwrite', [False, True])
def test_explicit_overwrite_controls_completed_day_rerun(monkeypatch, tmp_path, overwrite):
    captured = _run_smart_pipeline_case(
        monkeypatch, tmp_path, initial_complete=True,
        previous_status={'state': 'success'},
        readiness={'ready': True, 'reason': 'ready_without_phacal',
                   'deadline_expired': False,
                   'refcal_timestamp_utc': '2026-07-09 12:52:53.000'},
        applied_refcal_date='2026-07-09', overwrite=overwrite)
    assert captured['result'] == {'failed_dates': []}
    assert captured['overwrite_values'] == ([True] if overwrite else [])


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


def _configure_explicit_ms_calib_test(monkeypatch, tmp_path):
    workdir = tmp_path / "work" / "20260403"
    workdir.mkdir(parents=True)
    monkeypatch.setattr(eovsa_pipeline.os, "chdir", lambda _path: None)
    monkeypatch.setattr(eovsa_pipeline, "udbmsslfcaleddir", str(tmp_path / "selfcal"))
    monkeypatch.setattr(eovsa_pipeline, "slfcaltbdir", str(tmp_path / "selfcal_tables"))
    monkeypatch.setattr(eovsa_pipeline, "synopticfigdir", str(tmp_path / "figures"))
    monkeypatch.setattr(
        eovsa_pipeline,
        "get_synoptic_product_output_dir",
        lambda *_args, **_kwargs: str(tmp_path / "images"),
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "trange2ms",
        lambda *_args, **_kwargs: pytest.fail("explicit input must bypass discovery/import"),
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        "calibeovsa",
        lambda *_args, **_kwargs: pytest.fail("explicit input must bypass calibeovsa"),
    )
    esip = importlib.import_module(
        "suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean"
    )
    captured = {}

    def fake_pipeline_run(vis, **kwargs):
        captured["vis"] = vis
        captured.update(kwargs)
        return {"imaged": True}

    monkeypatch.setattr(esip, "pipeline_run", fake_pipeline_run)
    return workdir, captured


@pytest.mark.parametrize('date', ['2026-04-03', '2026-09-05'])
def test_explicit_calibrated_ms_skips_import_and_runs_selfcal(monkeypatch, tmp_path, date):
    workdir, captured = _configure_explicit_ms_calib_test(monkeypatch, tmp_path)
    input_ms = workdir / ('UDB' + date.replace('-', '') + '.ms')
    input_ms.mkdir()

    result = eovsa_pipeline.calib_pipeline(
        Time(date + ' 20:00:00'),
        workdir=str(workdir),
        version="v2.0_alt",
        doimport=False,
        input_ms=str(input_ms),
    )

    assert result == {"imaged": True}
    assert captured["vis"] == str(input_ms)
    assert "imaging_only" not in captured
    assert captured["preserve_input"] is True
    assert captured["outputvis"].endswith('UDB' + date.replace('-', '') + '.v2.0_alt.ms')


def test_explicit_selfcal_ms_runs_imaging_only(monkeypatch, tmp_path):
    workdir, captured = _configure_explicit_ms_calib_test(monkeypatch, tmp_path)
    input_ms = workdir / "UDB20260403.v2.0.ms"
    input_ms.mkdir()

    result = eovsa_pipeline.calib_pipeline(
        Time("2026-04-03 20:00:00"),
        workdir=str(workdir),
        version="v2.0",
        doimport=False,
        imaging_only=True,
        input_ms=str(input_ms),
    )

    assert result == {"imaged": True}
    assert captured["vis"] == str(input_ms)
    assert captured["outputvis"] == ""
    assert captured["imaging_only"] is True
    assert captured["preserve_input"] is True


def test_pipeline_overwrite_preserves_explicit_workdir_input(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    date_workdir = workdir / "20260403"
    input_ms = date_workdir / "UDB20260403.v2.0.ms"
    input_ms.mkdir(parents=True)
    marker = input_ms / "table.dat"
    marker.write_text("selfcal")
    stale = date_workdir / "stale.tmp"
    stale.write_text("remove me")
    captured = {}

    monkeypatch.setattr(eovsa_pipeline, "workdir_default", str(workdir))
    monkeypatch.setattr(
        eovsa_pipeline,
        "summarize_synoptic_outputs",
        lambda *_args, **_kwargs: {
            "statusfile": str(tmp_path / "status.json"),
            "fits_complete": False,
            "fits_count": 0,
            "fits_expected_count": 7,
            "existing_fitsfiles": [],
        },
    )

    def fake_calib_pipeline(*_args, **kwargs):
        captured.update(kwargs)
        return {"imaged": True}

    monkeypatch.setattr(eovsa_pipeline, "calib_pipeline", fake_calib_pipeline)

    result = eovsa_pipeline.pipeline(
        year=2026,
        month=4,
        day=3,
        clearcache=True,
        overwrite=True,
        doimport=False,
        smart_cal_check=False,
        version="v2.0",
        imaging_only=True,
        input_ms=str(input_ms),
    )

    assert result == {"failed_dates": []}
    assert marker.read_text() == "selfcal"
    assert not stale.exists()
    assert captured["input_ms"] == str(input_ms)


def test_explicit_ms_rejects_import_and_unverifiable_smart_check(tmp_path):
    input_ms = tmp_path / "UDB20260403.ms"
    input_ms.mkdir()

    with pytest.raises(ValueError, match="mutually exclusive"):
        eovsa_pipeline.validate_reused_ms_options(str(input_ms), doimport=True)
    with pytest.raises(ValueError, match="cannot verify a reused input_ms"):
        eovsa_pipeline.validate_reused_ms_options(
            str(input_ms), smart_cal_check=True
        )


def test_explicit_ms_rejects_wrong_artifact_stage_and_date(tmp_path):
    calibrated_ms = tmp_path / "UDB20260403.ms"
    calibrated_ms.mkdir()
    selfcal_ms = tmp_path / "UDB20260403.v2.0.ms"
    selfcal_ms.mkdir()

    with pytest.raises(ValueError, match="imaging_only input_ms must be the versioned selfcal"):
        eovsa_pipeline.validate_reused_ms_options(
            str(calibrated_ms),
            version="v2.0",
            imaging_only=True,
        )
    with pytest.raises(ValueError, match="must be the unversioned calibrated daily MS"):
        eovsa_pipeline.validate_reused_ms_options(
            str(selfcal_ms),
            version="v2.0",
            imaging_only=False,
        )
    with pytest.raises(ValueError, match="does not match requested date"):
        eovsa_pipeline.normalize_reused_ms_input(
            str(calibrated_ms), expected_date=Time("2026-04-04 20:00:00")
        )


def test_wsclean_stages_explicit_ms_without_mutating_source(tmp_path):
    esip = importlib.import_module(
        "suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean"
    )
    source = tmp_path / "work" / "20260403" / "UDB20260403.ms"
    source.mkdir(parents=True)
    source_marker = source / "table.dat"
    source_marker.write_text("calibrated")

    staged = Path(esip._stage_pipeline_input(
        str(source),
        "UDB20260403",
        str(source.parent),
        preserve_input=True,
    ))
    (staged / "table.dat").write_text("pipeline mutation")

    assert staged != source
    assert source_marker.read_text() == "calibrated"
    assert (staged / "table.dat").read_text() == "pipeline mutation"


def test_wsclean_stages_explicit_selfcal_archive_without_removing_it(tmp_path):
    esip = importlib.import_module(
        "suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean"
    )
    source_ms = tmp_path / "source" / "UDB20260403.v2.0.ms"
    source_ms.mkdir(parents=True)
    (source_ms / "table.dat").write_text("selfcal")
    archive = tmp_path / "work" / "20260403" / "UDB20260403.v2.0.ms.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source_ms, arcname=source_ms.name)

    staged = Path(esip._stage_pipeline_input(
        str(archive),
        "UDB20260403.v2.0",
        str(archive.parent),
        preserve_input=True,
    ))

    assert archive.exists()
    assert staged != source_ms
    assert (staged / "table.dat").read_text() == "selfcal"


def test_outputvis_is_archived_on_scratch_then_published_atomically(tmp_path):
    esip = importlib.import_module(
        "suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean"
    )
    staged_ms = tmp_path / "scratch" / "UDB20260403.v2.0.ms"
    staged_ms.mkdir(parents=True)
    (staged_ms / "table.dat").write_text("selfcal")
    durable_ms = tmp_path / "raid" / "UDB20260403.v2.0.ms"

    durable_archive = esip._archive_and_publish_outputvis(
        str(staged_ms),
        str(durable_ms),
    )

    assert durable_archive == str(durable_ms) + ".tar.gz"
    assert Path(durable_archive).is_file()
    assert not staged_ms.exists()
    assert not Path(durable_archive + ".partial").exists()
    with tarfile.open(durable_archive, "r:gz") as tar:
        member = tar.extractfile("UDB20260403.v2.0.ms/table.dat")
        assert member.read().decode() == "selfcal"


def test_flagversions_replacement_fails_closed_when_cleanup_fails(
    monkeypatch, tmp_path
):
    esip = importlib.import_module(
        "suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean"
    )
    target = tmp_path / "UDB20250809.ms.flagversions"
    target.mkdir()

    def refuse_rmtree(*_args, **_kwargs):
        raise PermissionError("read only")

    monkeypatch.setattr(esip.shutil, "rmtree", refuse_rmtree)

    with pytest.raises(
        PermissionError, match="Unable to replace existing flagversions directory"
    ):
        esip._remove_existing_path_or_raise(
            str(target), "flagversions directory"
        )

    assert target.is_dir()


def test_wsclean_preflight_rejects_missing_executable(monkeypatch):
    esip = importlib.import_module(
        "suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean"
    )
    monkeypatch.setattr(esip, "WSCLEAN_BIN", "wsclean-not-installed")
    monkeypatch.setattr(esip.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="WSClean executable is unavailable"):
        esip._require_wsclean_available()


def test_wsclean_preflight_keeps_wrapper_command_in_sync(monkeypatch):
    esip = importlib.import_module(
        "suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean"
    )
    monkeypatch.setattr(esip, "WSCLEAN_BIN", "wsclean-test")
    monkeypatch.setattr(
        esip.shutil, "which", lambda name: "/usr/local/bin/wsclean-test"
        if name == "wsclean-test" else None
    )
    monkeypatch.setattr(esip.ww, "WSCLEAN_BIN", "stale-wsclean-command")

    assert esip._require_wsclean_available() == "wsclean-test"
    assert esip.ww.WSCLEAN_BIN == "wsclean-test"


def test_missing_corrected_data_fails_closed_before_output_assembly(monkeypatch):
    esip = importlib.import_module(
        "suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean"
    )
    monkeypatch.setattr(esip, "_ms_has_column", lambda _msfile, _column: False)

    with pytest.raises(
        RuntimeError,
        match="End-of-run self-calibrated output assembly requires CORRECTED_DATA",
    ):
        esip._require_ms_column(
            "/scratch/eovsa/workdir/20250809/UDB20250809.ms",
            "CORRECTED_DATA",
            "End-of-run self-calibrated output assembly",
        )
