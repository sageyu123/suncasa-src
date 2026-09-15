import ast
import inspect
from pathlib import Path

import pytest

from suncasa.eovsa import eovsa_synoptic_imaging_pipeline_wsclean as pipeline


_REPO_ROOT = Path(__file__).resolve().parent


def _write(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")
    return path


def test_s00_route_namespace_preserves_exact_tag_and_avoids_slug_collision():
    production = pipeline._s00_route_identity("")
    slash_tag = pipeline._s00_route_identity("npz/route")
    space_tag = pipeline._s00_route_identity("npz route")

    assert production == {
        "fits_tag": "",
        "route_id": "production",
        "namespace": "production",
    }
    assert slash_tag["fits_tag"] == "npz/route"
    assert slash_tag["route_id"] == "npz/route"
    assert slash_tag["namespace"] != space_tag["namespace"]


def test_begin_s00_attempt_removes_only_same_route_and_blocks_glob_fallback(tmp_path):
    route_tag = "v31alt"
    same_route = [
        _write(tmp_path / "eovsa.synoptic.v31alt.20260722T180000Z.s00-01.tb.fits"),
        _write(tmp_path / "eovsa.synoptic.v31alt.20260723T000000Z.s00-01.tb.disk.fits"),
        _write(tmp_path / "eovsa.synoptic_daily.v31alt.20260722T200000Z.s00-01.tb.fits"),
        _write(tmp_path / "eovsa.synoptic_daily.v31alt.20260722T200000Z.s00-01.tb.disk.fits"),
    ]
    other_route = _write(
        tmp_path / "eovsa.synoptic.sql.20260722T180000Z.s00-01.tb.fits"
    )
    production = _write(
        tmp_path / "eovsa.synoptic.20260722T180000Z.s00-01.tb.fits"
    )
    other_band = _write(
        tmp_path / "eovsa.synoptic.v31alt.20260722T180000Z.s02-04.tb.fits"
    )

    policies = {4: {"state": "unrelated"}}
    manifests = {0: ["stale.fits"], 4: ["other.fits"]}
    removed = pipeline._begin_s00_attempt(
        policies,
        manifests,
        0,
        str(tmp_path),
        "20260722",
        "00-01",
        fits_tag=route_tag,
    )

    assert policies[0] is None
    assert manifests[0] == []
    assert policies[4] == {"state": "unrelated"}
    assert manifests[4] == ["other.fits"]
    assert set(map(Path, removed)) == set(same_route)
    assert all(not path.exists() for path in same_route)
    assert other_route.exists()
    assert production.exists()
    assert other_band.exists()


def test_begin_s00_attempt_fails_closed_when_stale_product_cannot_be_removed(
    monkeypatch, tmp_path
):
    stale = _write(
        tmp_path / "eovsa.synoptic.v31alt.20260722T180000Z.s00-01.tb.fits"
    )
    policies = {}
    manifests = {}
    real_remove = pipeline.os.remove

    def deny_stale(path):
        if Path(path) == stale:
            raise PermissionError("read only")
        return real_remove(path)

    monkeypatch.setattr(pipeline.os, "remove", deny_stale)

    with pytest.raises(RuntimeError, match="Unable to withhold stale s00"):
        pipeline._begin_s00_attempt(
            policies,
            manifests,
            0,
            str(tmp_path),
            "20260722",
            "00-01",
            fits_tag="v31alt",
        )

    assert policies[0] is None
    assert manifests[0] == []
    assert stale.exists()


def test_prepare_s00_uses_route_specific_qa_dir_and_seed_prefix(monkeypatch, tmp_path):
    captured_prefixes = []

    monkeypatch.setattr(
        pipeline,
        "_central_wsclean_interval",
        lambda *_args, **_kwargs: {"interval": [0, 2]},
    )

    def failed_seed(_msfile, prefix, *_args, **_kwargs):
        captured_prefixes.append(prefix)
        return {"status": 1, "model": None, "image": None}

    monkeypatch.setattr(pipeline, "_run_s00_wsclean", failed_seed)

    common = dict(
        msfile="day.ms",
        msname="day",
        sp_index="0,1",
        workdir=str(tmp_path / "work"),
        imgoutdir=str(tmp_path / "images"),
        wsclean_intervals=object(),
        reftime_daily=object(),
        model_fits=None,
        disk_mask_fits=None,
        briggs_val=0.0,
        bmsize=60.0,
        pols="XX",
        data_column="CORRECTED_DATA",
    )
    assert pipeline._prepare_s00_final_policy(
        **common, fits_tag="npz/route"
    ) is None
    assert pipeline._prepare_s00_final_policy(
        **common, fits_tag="sql route"
    ) is None

    npz_route = pipeline._s00_route_identity("npz/route")
    sql_route = pipeline._s00_route_identity("sql route")
    assert (tmp_path / "images" / "qa" / "s00" / npz_route["namespace"]).is_dir()
    assert (tmp_path / "images" / "qa" / "s00" / sql_route["namespace"]).is_dir()
    assert npz_route["namespace"] in captured_prefixes[0]
    assert sql_route["namespace"] in captured_prefixes[1]
    assert captured_prefixes[0] != captured_prefixes[1]


def test_prepare_s00_never_reuses_another_routes_archived_model(monkeypatch, tmp_path):
    npz_route = pipeline._s00_route_identity("npz")
    npz_qa = tmp_path / "images" / "qa" / "s00" / npz_route["namespace"]
    _write(npz_qa / "eovsa-day-init-t0000-model.fits")
    seed_calls = []

    monkeypatch.setattr(
        pipeline,
        "_central_wsclean_interval",
        lambda *_args, **_kwargs: {"interval": [0, 2]},
    )

    def failed_seed(_msfile, prefix, *_args, **_kwargs):
        seed_calls.append(prefix)
        return {"status": 1, "model": None, "image": None}

    monkeypatch.setattr(pipeline, "_run_s00_wsclean", failed_seed)

    result = pipeline._prepare_s00_final_policy(
        "day.ms",
        "day",
        "0,1",
        str(tmp_path / "work"),
        str(tmp_path / "images"),
        object(),
        object(),
        None,
        None,
        0.0,
        60.0,
        "XX",
        "CORRECTED_DATA",
        fits_tag="sql",
    )

    assert result is None
    assert len(seed_calls) == 1
    assert pipeline._s00_route_identity("sql")["namespace"] in seed_calls[0]


def test_s00_adaptive_route_is_explicit_and_isolated_to_unfiltered_s00():
    assert not pipeline._adaptive_s00_enabled(False, "0~1", None)
    assert pipeline._adaptive_s00_enabled(True, "0~1", None)
    assert not pipeline._adaptive_s00_enabled(
        True, "0~1", ["2026-07-22T19:00~2026-07-22T21:00"]
    )
    assert not pipeline._adaptive_s00_enabled(True, "2~4", None)
    assert inspect.signature(pipeline.pipeline_run).parameters["adaptive_s00"].default is False


def test_default_s00_route_keeps_segmented_full_track_intervals():
    source = (_REPO_ROOT / "suncasa/eovsa/eovsa_synoptic_imaging_pipeline_wsclean.py").read_text()

    assert "s00_policy = None" in source
    assert "ri_final_coarse = ri_final" in source
    assert "intervals_out=(1 if s00_policy else ri_final['N1'])" in source
    assert "full-track default" in source
    assert "adaptive opt-in" in source


def _function_node(source, name):
    tree = ast.parse(source)
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _argument_default(function_node, name):
    positional = list(function_node.args.posonlyargs) + list(function_node.args.args)
    defaults = [None] * (len(positional) - len(function_node.args.defaults))
    defaults.extend(function_node.args.defaults)
    for argument, default in zip(positional, defaults):
        if argument.arg == name:
            return default
    raise AssertionError(f"argument {name!r} not found")


def test_outer_api_and_both_clis_propagate_adaptive_s00():
    outer_path = _REPO_ROOT / "suncasa/eovsa/eovsa_pipeline.py"
    inner_path = _REPO_ROOT / "suncasa/eovsa/eovsa_synoptic_imaging_pipeline_wsclean.py"
    outer_source = outer_path.read_text()
    inner_source = inner_path.read_text()

    for name in ("calib_pipeline", "pipeline"):
        default = _argument_default(_function_node(outer_source, name), "adaptive_s00")
        assert isinstance(default, ast.Constant) and default.value is False
    inner_default = _argument_default(_function_node(inner_source, "pipeline_run"), "adaptive_s00")
    assert isinstance(inner_default, ast.Constant) and inner_default.value is False

    assert outer_source.count("'adaptive_s00': adaptive_s00") >= 3
    assert outer_source.count("adaptive_s00=adaptive_s00") >= 3
    assert "--adaptive-s00" in outer_source
    assert "--adaptive-s00" in inner_source
    assert "adaptive_s00=args.adaptive_s00" in outer_source
    assert "adaptive_s00=args.adaptive_s00" in inner_source
