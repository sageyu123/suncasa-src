from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PARENT_SPWS = ["0~1", "2~4", "5~10", "11~20", "21~30", "31~43", "44~49"]
FINE_SPWS = [
    "0~1", "2", "3", "4", "5~6", "7~8", "9~10",
    "11~12", "13~14", "15~16", "17~18", "19~20",
    "21~22", "23~24", "25~26", "27~28", "29~30",
    "31~33", "34~35", "36~37", "38~39", "40~41", "42~43",
    "44~49",
]
EXPECTED_FINE_CHILDREN = {
    "0~1": [],
    "2~4": ["2~2", "3~3", "4~4"],
    "5~10": ["5~6", "7~8", "9~10"],
    "11~20": ["11~12", "13~14", "15~16", "17~18", "19~20"],
    "21~30": ["21~22", "23~24", "25~26", "27~28", "29~30"],
    "31~43": ["31~33", "34~35", "36~37", "38~39", "40~41", "42~43"],
    "44~49": [],
}


@pytest.fixture(scope="module")
def pipeline():
    return pytest.importorskip("suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean")


def test_default_coarse_spws_remain_unchanged(pipeline):
    from eovsapy.spw_config import SPWS_52BAND

    assert list(SPWS_52BAND) == PARENT_SPWS
    assert pipeline.PIPELINE_CONFIG["spwidx2proc"] == list(range(7))


def test_exact_fine_plan_routes_to_scratch_without_changing_default_parent_plan(pipeline):
    assert pipeline.FINE_SPECTRAL_SPWS_52BAND == FINE_SPWS
    assert len(FINE_SPWS) == 24

    normalized_fine_spws = [
        f"{start}~{end}"
        for start, end in map(pipeline._spw_range_bounds, FINE_SPWS)
    ]
    assert pipeline._resolve_fine_spectral_spw_mode(
        FINE_SPWS, True, fine_spectral_bootstrap=False) == (
        normalized_fine_spws,
        None,
    )
    assert pipeline._resolve_fine_spectral_spw_mode(
        FINE_SPWS, True, fine_spectral_bootstrap=True) == (
        None,
        normalized_fine_spws,
    )
    assert pipeline._resolve_fine_spectral_spw_mode(
        FINE_SPWS, False, fine_spectral_bootstrap=True) == (
        normalized_fine_spws,
        None,
    )
    assert pipeline._resolve_fine_spectral_spw_mode(
        None, False, fine_spectral_bootstrap=False) == (None, None)
    assert pipeline._resolve_fine_spectral_spw_mode(["2", "5~6"], True) == (
        ["2~2", "5~6"],
        None,
    )


def test_fine_children_are_exact_and_strictly_contained(pipeline):
    assert list(EXPECTED_FINE_CHILDREN) == PARENT_SPWS
    expected_children = [
        child
        for children in EXPECTED_FINE_CHILDREN.values()
        for child in children
    ]
    assert len(expected_children) == 22
    assert len(set(expected_children)) == 22

    for parent, expected in EXPECTED_FINE_CHILDREN.items():
        children = pipeline._fine_spectral_children_for_parent(parent, FINE_SPWS)
        assert children == expected
        parent_start, parent_end = pipeline._spw_range_bounds(parent)
        for child in children:
            child_start, child_end = pipeline._spw_range_bounds(child)
            assert parent_start <= child_start <= child_end <= parent_end
    assert [pipeline.format_spw(spw) for spw in EXPECTED_FINE_CHILDREN["2~4"]] == [
        "02-02", "03-03", "04-04",
    ]


def test_generic_custom_and_34_band_chunking_remain_unchanged(pipeline):
    assert pipeline._fine_spectral_spws_for_range("2~4") == ["2~4"]
    assert pipeline._fine_spectral_children_for_parent("2~4") == []
    assert pipeline._fine_spectral_spws_for_range("1~3") == ["1~3"]


def test_stale_fine_products_are_removed_without_touching_other_products(pipeline, tmp_path):
    stale_names = [
        "eovsa.synoptic.test.20260710T120000Z.s05-06.tb.fits",
        "eovsa.synoptic.test.20260711T010000Z.s05-06.tb.disk.fits",
        "eovsa.synoptic_daily.test.20260710T200000Z.s05-06.tb.fits",
        "eovsa.synoptic_daily.test.20260710T200000Z.s05-06.tb.disk.fits",
    ]
    keep_names = [
        "eovsa.synoptic.20260710T120000Z.s05-06.tb.fits",
        "eovsa.synoptic.test.20260710T120000Z.s07-08.tb.fits",
        "eovsa.synoptic.test.20260712T010000Z.s05-06.tb.fits",
    ]
    for name in stale_names + keep_names:
        (tmp_path / name).touch()

    pipeline._remove_fine_spectral_products(
        str(tmp_path), "20260710", "05-06", fits_tag="test")

    assert not any((tmp_path / name).exists() for name in stale_names)
    assert all((tmp_path / name).exists() for name in keep_names)


def test_parent_checkpoint_marker_distinguishes_new_hierarchical_archives(pipeline, tmp_path):
    archive_ms = tmp_path / "archive.ms"
    archive_ms.mkdir()

    assert pipeline._is_default_52_parent_plan(PARENT_SPWS)
    assert not pipeline._is_default_52_parent_plan(["1~3", "4~9", "10~16"])
    assert not pipeline._has_fine_bootstrap_parent_marker(str(archive_ms))

    marker = pipeline._write_fine_bootstrap_parent_marker(str(archive_ms))

    assert Path(marker).name == pipeline.FINE_BOOTSTRAP_PARENT_MARKER
    assert pipeline._has_fine_bootstrap_parent_marker(str(archive_ms))
    assert "parents=0~1,2~4,5~10,11~20,21~30,31~43,44~49" in Path(marker).read_text()


def test_parentless_fine_only_mode_is_rejected_before_touching_an_ms(pipeline):
    with pytest.raises(RuntimeError, match="no trusted coarse-final MODEL_DATA"):
        pipeline.pipeline_run("unused.ms", fine_spectral_only=True)


def _bootstrap_helper(pipeline):
    helper = getattr(pipeline, "_run_bootstrapped_fine_product", None)
    assert helper is not None, "fine products need a parent MODEL_DATA bootstrap helper"
    return helper


@pytest.mark.parametrize(
    ("parent_data_column", "expected_split_column"),
    [("DATA", "data"), ("CORRECTED_DATA", "corrected")],
)
def test_bootstrap_copies_parent_model_then_runs_one_phase_solve(
        pipeline, tmp_path, parent_data_column, expected_split_column):
    helper = _bootstrap_helper(pipeline)
    parent_ms = str(tmp_path / "parent.ms")
    fine_ms = str(tmp_path / "fine-05-06.ms")
    Path(parent_ms).mkdir()
    events = []
    calls = {}

    def fake_splitx(vis, **kwargs):
        events.append("splitX")
        calls["splitX"] = (vis, kwargs.copy())
        Path(kwargs["outputvis"]).mkdir()
        Path(f'{kwargs["outputvis"]}.flagversions').mkdir()
        Path(f"{vis}.tmpms").mkdir()
        return kwargs["outputvis"]

    def fake_gaincal(**kwargs):
        events.append("gaincal")
        calls["gaincal"] = kwargs.copy()
        Path(kwargs["caltable"]).mkdir()

    def fake_applycal(**kwargs):
        events.append("applycal")
        calls["applycal"] = kwargs.copy()

    final_result = (["fine.fits"], ["fine-imaging-object"])

    def fake_final_imaging(**kwargs):
        events.append("final_imaging")
        calls["final_imaging"] = kwargs.copy()
        return final_result

    with (
        patch.object(pipeline.mstl, "splitX", side_effect=fake_splitx),
        patch.object(pipeline, "gaincal", side_effect=fake_gaincal),
        patch.object(pipeline, "applycal", side_effect=fake_applycal),
        patch.object(pipeline, "_run_final_imaging", side_effect=fake_final_imaging),
        patch.object(pipeline, "_caltable_has_unflagged_solutions", return_value=True),
        patch.object(pipeline, "_run_slfcal_round") as scratch_selfcal,
    ):
        result = helper(
            parent_msfile=parent_ms,
            parent_data_column=parent_data_column,
            source_sp_index="5,6",
            fine_spw="5~6",
            fine_spwstr="05-06",
            fine_msfile=fine_ms,
            workdir=str(tmp_path),
            antenna="0~12",
            final_imaging_kwargs={"sidx": "fine:05-06"},
        )

    assert result == final_result
    assert events == ["splitX", "gaincal", "applycal", "final_imaging"]
    scratch_selfcal.assert_not_called()

    split_vis, split_kwargs = calls["splitX"]
    assert split_vis == parent_ms
    assert split_kwargs["outputvis"] == fine_ms
    assert split_kwargs["spw"] == "5,6"
    assert split_kwargs["datacolumn"] == expected_split_column
    assert split_kwargs["datacolumn2"] == "MODEL_DATA"

    gain_kwargs = calls["gaincal"]
    assert gain_kwargs["vis"] == fine_ms
    assert gain_kwargs["spw"] == "0~1"
    assert gain_kwargs["calmode"] == "p"
    assert gain_kwargs["solint"] == "inf"

    apply_kwargs = calls["applycal"]
    assert apply_kwargs["vis"] == fine_ms
    assert apply_kwargs["spw"] == "0~1"
    assert apply_kwargs["gaintable"] == [gain_kwargs["caltable"]]

    final_kwargs = calls["final_imaging"]
    assert final_kwargs["msfile"] == fine_ms
    assert final_kwargs["spw"] == "5~6"
    assert final_kwargs["spwstr"] == "05-06"
    assert final_kwargs["sp_index"] == "0,1"
    assert final_kwargs["data_column"] == "CORRECTED_DATA"
    assert not any(
        kwargs.get("vis") == parent_ms
        for kwargs in (gain_kwargs, apply_kwargs)
    )
    assert not Path(fine_ms).exists()
    assert not Path(f"{fine_ms}.flagversions").exists()
    assert not Path(gain_kwargs["caltable"]).exists()
    assert not Path(f"{parent_ms}.tmpms").exists()


def test_unusable_phase_solutions_skip_apply_and_final_and_clean_scratch(pipeline, tmp_path):
    helper = _bootstrap_helper(pipeline)
    parent_ms = str(tmp_path / "parent.ms")
    fine_ms = str(tmp_path / "fine-09-10.ms")
    Path(parent_ms).mkdir()

    def fake_splitx(_vis, **kwargs):
        Path(kwargs["outputvis"]).mkdir()
        Path(f'{kwargs["outputvis"]}.flagversions').mkdir()
        Path(f"{_vis}.tmpms").mkdir()
        return kwargs["outputvis"]

    def fake_gaincal(**kwargs):
        Path(kwargs["caltable"]).mkdir()

    with (
        patch.object(pipeline.mstl, "splitX", side_effect=fake_splitx),
        patch.object(pipeline, "gaincal", side_effect=fake_gaincal),
        patch.object(pipeline, "_caltable_has_unflagged_solutions", return_value=False),
        patch.object(pipeline, "applycal") as applycal_mock,
        patch.object(pipeline, "_run_final_imaging") as final_imaging_mock,
        patch.object(pipeline, "_run_slfcal_round") as scratch_selfcal,
    ):
        result = helper(
            parent_msfile=parent_ms,
            parent_data_column="CORRECTED_DATA",
            source_sp_index="9,10",
            fine_spw="9~10",
            fine_spwstr="09-10",
            fine_msfile=fine_ms,
            workdir=str(tmp_path),
            antenna="0~12",
            final_imaging_kwargs={"sidx": "fine:09-10"},
        )

    assert result is None
    applycal_mock.assert_not_called()
    final_imaging_mock.assert_not_called()
    scratch_selfcal.assert_not_called()
    assert not Path(fine_ms).exists()
    assert not Path(f"{fine_ms}.flagversions").exists()
    assert not (tmp_path / "caltb_fine_bootstrap_sp09-10.pha").exists()
    assert not Path(f"{parent_ms}.tmpms").exists()


@pytest.mark.parametrize("create_output", [False, True])
def test_missing_parent_snapshot_skips_without_scratch_fallback(pipeline, tmp_path, create_output):
    helper = _bootstrap_helper(pipeline)
    parent_ms = str(tmp_path / "parent.ms")
    fine_ms = str(tmp_path / "fine-07-08.ms")
    Path(parent_ms).mkdir()

    def unavailable_split(_vis, **kwargs):
        Path(f"{_vis}.tmpms").mkdir()
        if create_output:
            Path(kwargs["outputvis"]).mkdir()
        raise RuntimeError("MODEL_DATA is unavailable")

    with (
        patch.object(pipeline.mstl, "splitX", side_effect=unavailable_split),
        patch.object(pipeline, "gaincal") as gaincal_mock,
        patch.object(pipeline, "applycal") as applycal_mock,
        patch.object(pipeline, "_run_final_imaging") as final_imaging_mock,
        patch.object(pipeline, "_run_slfcal_round") as scratch_selfcal,
    ):
        result = helper(
            parent_msfile=parent_ms,
            parent_data_column="CORRECTED_DATA",
            source_sp_index="7,8",
            fine_spw="7~8",
            fine_spwstr="07-08",
            fine_msfile=fine_ms,
            workdir=str(tmp_path),
            antenna="0~12",
            final_imaging_kwargs={"sidx": "fine:07-08"},
        )

    assert result is None
    gaincal_mock.assert_not_called()
    applycal_mock.assert_not_called()
    final_imaging_mock.assert_not_called()
    scratch_selfcal.assert_not_called()
    assert not Path(fine_ms).exists()
    assert not Path(f"{parent_ms}.tmpms").exists()


def test_ms_selfcal_accepts_local_spw_index_override(pipeline, tmp_path):
    imaging = pipeline.MSselfcal(
        "unused.ms",
        [],
        [],
        [],
        "5~6",
        0,
        0,
        str(tmp_path),
        sp_index="0,1",
    )

    assert imaging.spw == "5~6"
    assert imaging.sp_index == "0,1"


def test_failed_parent_final_clean_is_not_bootstrap_ready(pipeline, tmp_path):
    clean = MagicMock()
    clean.run.return_value = 9
    imaging_objs = {0: None}

    class FrequencySetupStub:
        @staticmethod
        def get_reffreq_and_cdelt(_spw, return_bmsize=False):
            assert return_bmsize is True
            return "5GHz", "1GHz", 20.0

    with (
        patch.object(pipeline.ww, "WSClean", return_value=clean),
        patch.object(pipeline, "count_unflagged_solar_antennas", return_value=13),
        patch.object(pipeline, "clean_junk"),
        patch.object(pipeline.hf, "imreg") as imreg_mock,
    ):
        result = pipeline._run_final_imaging(
            msfile="parent.ms",
            sidx=0,
            spw="5~10",
            spwstr="05-10",
            sp_index="5,6,7,8,9,10",
            workdir=str(tmp_path),
            imgoutdir=str(tmp_path),
            msname="parent",
            ri_final={"N1": 1},
            briggs_val=-1.0,
            bmsize=20.0,
            pols="XX",
            reftime_daily=None,
            viz_timerange="",
            date_str="20260710",
            is_segmented=True,
            imaging_objs=imaging_objs,
            freq_setup=FrequencySetupStub(),
            solar_antenna_total=13,
        )

    assert result == ([], imaging_objs)
    clean.run.assert_called_once_with(dryrun=False)
    imreg_mock.assert_not_called()
