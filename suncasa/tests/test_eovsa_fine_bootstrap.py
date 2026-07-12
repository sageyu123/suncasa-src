from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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
    assert pipeline._resolve_fine_spectral_spw_mode(
        None, True, fine_spectral_bootstrap=False) == (
        normalized_fine_spws,
        None,
    )
    assert pipeline._resolve_fine_spectral_spw_mode(
        None, True, fine_spectral_bootstrap=True) == (
        None,
        normalized_fine_spws,
    )
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
    with pytest.raises(RuntimeError, match="pre-disk"):
        pipeline.pipeline_run("unused.ms", fine_spectral_only=True)


def test_bootstrap_resume_mode_is_rejected_before_touching_an_ms(pipeline):
    with pytest.raises((RuntimeError, ValueError), match="pre-disk"):
        pipeline.pipeline_run(
            "unused.ms",
            fine_spectral_imaging=True,
            fine_spectral_bootstrap=True,
            imaging_only=True,
        )


def test_bootstrap_rejects_nonexact_custom_plan_before_touching_an_ms(pipeline):
    with pytest.raises(ValueError, match="exact.*52-band.*plan"):
        pipeline.pipeline_run(
            "unused.ms",
            fine_spectral_imaging=True,
            fine_spectral_bootstrap=True,
            custom_spws=["5~6", "7~8"],
        )


def _bootstrap_helper(pipeline):
    helper = getattr(pipeline, "_run_bootstrapped_fine_product", None)
    assert helper is not None, "fine products need a parent MODEL_DATA bootstrap helper"
    return helper


def test_parent_checkpoint_is_captured_before_disk_subtraction(pipeline, tmp_path):
    parent_ms = str(tmp_path / "parent.ms")
    checkpoint_ms = str(tmp_path / "parent.pre-disk.ms")
    Path(parent_ms).mkdir()
    events = []
    split_calls = []
    predict_count = 0

    def fake_run_gaincal_step(*args, **_kwargs):
        return list(args[5]) + [f"phase-{len(args[5])}"]

    def fake_gaincal(**kwargs):
        Path(kwargs["caltable"]).mkdir()

    def fake_applycal(**_kwargs):
        events.append("applycal")

    def fake_flagdata(**_kwargs):
        events.append("flagdata")

    def fake_split(**kwargs):
        events.append("checkpoint")
        split_calls.append(kwargs.copy())
        Path(kwargs["outputvis"]).mkdir()

    def fake_predict(*_args, **_kwargs):
        nonlocal predict_count
        predict_count += 1
        events.append(f"predict-{predict_count}")
        return MagicMock(returncode=0)

    def fake_uvsub(**_kwargs):
        events.append("uvsub")

    feature_model = MagicMock()
    feature_model.model_minor_name_str = str(tmp_path / "parent-feature")
    freq_setup = MagicMock()
    freq_setup.get_reffreq_and_cdelt.return_value = ("5GHz", "0.3GHz", 20.0)

    with (
        patch.object(pipeline, "delmod"),
        patch.object(pipeline, "glob", return_value=[str(tmp_path / "parent-feature-t0000-model.fits")]),
        patch.object(pipeline, "add_convolved_disk_to_fits"),
        patch.object(pipeline, "_run_gaincal_step", side_effect=fake_run_gaincal_step),
        patch.object(pipeline, "gaincal", side_effect=fake_gaincal),
        patch.object(pipeline, "applycal", side_effect=fake_applycal),
        patch.object(pipeline, "flagdata", side_effect=fake_flagdata),
        patch.object(pipeline, "split", side_effect=fake_split),
        patch.object(pipeline.mstl, "flagcaltboutliers"),
        patch.object(pipeline.subprocess, "run", side_effect=fake_predict),
        patch.object(pipeline, "uvsub", side_effect=fake_uvsub),
    ):
        pipeline._run_disk_selfcal(
            parent_ms,
            0,
            "5~6",
            "05-06",
            "5,6",
            workdir=str(tmp_path),
            antenna="0~12",
            caltbs=[],
            slfcal_init_obj=feature_model,
            imname_init_disk_strlist=[str(tmp_path / "disk-init")],
            freq_setup=freq_setup,
            dsize=["960arcsec"] * 50,
            fdens=[1.0] * 50,
            ri_init={"N1": 1, "N2": 1},
            ri_final={"N1": 1, "N2": 1},
            tdur=60,
            uvmin_l_str_sidx=">200lambda",
            overwrite_caltb=True,
            pols="XX",
            pre_disk_outputvis=checkpoint_ms,
        )

    assert split_calls == [{
        "vis": parent_ms,
        "outputvis": checkpoint_ms,
        "spw": "5~6",
        "datacolumn": "corrected",
    }]
    assert events.index("applycal") < events.index("flagdata")
    assert events.index("flagdata") < events.index("checkpoint")
    post_checkpoint_predicts = [
        idx for idx, event in enumerate(events)
        if event.startswith("predict-") and idx > events.index("checkpoint")
    ]
    assert post_checkpoint_predicts
    assert events.index("checkpoint") < post_checkpoint_predicts[0]
    assert post_checkpoint_predicts[-1] < events.index("uvsub")


def _fullsky_child_builder(pipeline):
    builder = getattr(pipeline, "_build_fullsky_fine_child", None)
    assert builder is not None, "fine bootstrap needs a pre-disk/full-sky child builder"
    return builder


def test_fullsky_child_combines_predisk_data_disk_model_and_final_residual_model(
        pipeline, tmp_path):
    builder = _fullsky_child_builder(pipeline)
    pre_disk_parent = str(tmp_path / "parent.pre-disk.ms")
    residual_parent = str(tmp_path / "parent.final-residual.ms")
    fine_ms = str(tmp_path / "fine-13-14.ms")
    residual_model_ms = str(tmp_path / "fine-13-14.residual-model.ms")
    Path(pre_disk_parent).mkdir()
    Path(residual_parent).mkdir()
    events = []
    split_calls = []

    def fake_split(**kwargs):
        event = {
            "data": "split_pre_disk_data",
            "model": "split_final_residual_model",
        }[kwargs["datacolumn"]]
        events.append(event)
        split_calls.append(kwargs.copy())
        Path(kwargs["outputvis"]).mkdir()

    def fake_clearcal(**_kwargs):
        events.append("clear_model")

    def fake_predict(*_args, **_kwargs):
        events.append("predict_disk")
        return MagicMock(returncode=0)

    def fake_add_model(source_ms, target_ms, **_kwargs):
        events.append("add_final_residual_model")
        assert source_ms == residual_model_ms
        assert target_ms == fine_ms

    disk_prefixes = [
        str(tmp_path / "disk-sp13"),
        str(tmp_path / "disk-sp14"),
    ]

    with (
        patch.object(pipeline, "split", side_effect=fake_split),
        patch.object(
            pipeline,
            "_ms_column_has_unflagged_signal_for_spws",
            return_value=True,
        ),
        patch.object(pipeline, "clearcal", side_effect=fake_clearcal),
        patch.object(pipeline, "glob", return_value=["disk-model.fits"]),
        patch.object(pipeline.subprocess, "run", side_effect=fake_predict),
        patch.object(pipeline, "_add_aligned_model_data", side_effect=fake_add_model),
    ):
        builder(
            pre_disk_parent_ms=pre_disk_parent,
            residual_parent_ms=residual_parent,
            pre_disk_sp_index="2,3",
            residual_sp_index="13,14",
            fine_msfile=fine_ms,
            residual_model_ms=residual_model_ms,
            disk_model_prefixes=disk_prefixes,
            pols="XX",
        )

    assert split_calls == [
        {
            "vis": pre_disk_parent,
            "outputvis": fine_ms,
            "spw": "2,3",
            "datacolumn": "data",
        },
        {
            "vis": residual_parent,
            "outputvis": residual_model_ms,
            "spw": "13,14",
            "datacolumn": "model",
        },
    ]
    assert events == [
        "split_pre_disk_data",
        "split_final_residual_model",
        "clear_model",
        "predict_disk",
        "predict_disk",
        "add_final_residual_model",
    ]


@pytest.mark.parametrize(
    ("bad_second_spw", "expected"),
    [
        (None, True),
        ("zero", False),
        ("flagged", False),
        ("nan", False),
        ("yy_only", False),
    ],
)
def test_ms_column_requires_finite_nonzero_unflagged_signal_in_every_spw(
        pipeline, tmp_path, bad_second_spw, expected):
    helper = getattr(pipeline, "_ms_column_has_unflagged_signal_for_spws", None)
    assert helper is not None, "fine bootstrap needs a per-SPW residual-signal guard"

    class SignalSubtable:
        def __init__(self, data, flags):
            self.data = data
            self.flags = flags

        def nrows(self):
            return self.data.shape[-1]

        def colnames(self):
            return ["DATA", "FLAG"]

        def getcol(self, column, startrow=0, nrow=-1):
            values = self.data if column == "DATA" else self.flags
            stop = None if nrow < 0 else startrow + nrow
            return values[..., startrow:stop].copy()

        def close(self):
            return None

    class SignalTable:
        def __init__(self, per_ddid):
            self.per_ddid = per_ddid

        def open(self, _path):
            return None

        def close(self):
            return None

        def colnames(self):
            return ["DATA", "FLAG", "DATA_DESC_ID"]

        def query(self, expression):
            ddid = int(expression.rsplit("==", 1)[1].strip())
            data, flags = self.per_ddid[ddid]
            return SignalSubtable(data, flags)

    valid_data = np.ones((2, 1, 1), dtype=np.complex128)
    valid_flags = np.zeros((2, 1, 1), dtype=bool)
    second_data = valid_data.copy()
    second_flags = valid_flags.copy()
    if bad_second_spw == "zero":
        second_data[:] = 0.0
    elif bad_second_spw == "flagged":
        second_flags[:] = True
    elif bad_second_spw == "nan":
        second_data[:] = np.nan + 0.0j
    elif bad_second_spw == "yy_only":
        second_data[:] = 0.0
        second_data[1] = 1.0

    signal_table = SignalTable({
        0: (valid_data, valid_flags),
        1: (second_data, second_flags),
    })
    msfile = str(tmp_path / "residual-model.ms")
    Path(msfile).mkdir()

    def fake_ddids(_msfile, spws):
        assert _msfile == msfile
        return [int(spw) for spw in spws]

    with (
        patch.object(pipeline, "tbtool", return_value=signal_table),
        patch.object(
            pipeline,
            "_data_description_ids_for_spws",
            side_effect=fake_ddids,
        ),
    ):
        assert helper(msfile, "DATA", [0, 1], pols="XX") is expected


def test_builder_rejects_incomplete_residual_model_before_model_addition(
        pipeline, tmp_path):
    builder = _fullsky_child_builder(pipeline)
    pre_disk_parent = str(tmp_path / "parent.pre-disk.ms")
    residual_parent = str(tmp_path / "parent.final-residual.ms")
    fine_ms = str(tmp_path / "fine-13-14.ms")
    residual_model_ms = str(tmp_path / "fine-13-14.residual-model.ms")
    Path(pre_disk_parent).mkdir()
    Path(residual_parent).mkdir()

    def fake_split(**kwargs):
        Path(kwargs["outputvis"]).mkdir()

    with (
        patch.object(pipeline, "split", side_effect=fake_split),
        patch.object(
            pipeline,
            "_ms_column_has_unflagged_signal_for_spws",
            return_value=False,
        ) as signal_guard,
        patch.object(pipeline, "clearcal") as clearcal_mock,
        patch.object(pipeline, "_predict_fine_disk_model") as predict_mock,
        patch.object(pipeline, "_add_aligned_model_data") as add_model_mock,
    ):
        with pytest.raises(RuntimeError, match="residual model.*every child SPW"):
            builder(
                pre_disk_parent_ms=pre_disk_parent,
                residual_parent_ms=residual_parent,
                pre_disk_sp_index="2,3",
                residual_sp_index="13,14",
                fine_msfile=fine_ms,
                residual_model_ms=residual_model_ms,
                disk_model_prefixes=["disk-sp13", "disk-sp14"],
                pols="XX",
            )

    signal_guard.assert_called_once_with(
        residual_model_ms, "DATA", [0, 1], pols="XX")
    clearcal_mock.assert_not_called()
    predict_mock.assert_not_called()
    add_model_mock.assert_not_called()


@pytest.mark.parametrize("mismatched_rows", [False, True])
def test_aligned_model_addition_rejects_row_mismatch_before_writing(
        pipeline, mismatched_rows):
    class FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.put_calls = []

        def open(self, _path, **_kwargs):
            return None

        def close(self):
            return None

        def flush(self):
            return None

        def colnames(self):
            return list(self.columns)

        def nrows(self):
            return len(self.columns["TIME"])

        def getcol(self, column, startrow=0, nrow=-1):
            values = self.columns[column]
            stop = None if nrow < 0 else startrow + nrow
            if column in {"DATA", "MODEL_DATA"}:
                return values[..., startrow:stop].copy()
            return values[startrow:stop].copy()

        def putcol(self, column, values, startrow=0, nrow=-1):
            self.put_calls.append((column, startrow, nrow))
            stop = None if nrow < 0 else startrow + nrow
            self.columns[column][..., startrow:stop] = values

    row_identity = {
        "TIME": np.array([1.0, 2.0]),
        "ANTENNA1": np.array([0, 1]),
        "ANTENNA2": np.array([2, 2]),
        "DATA_DESC_ID": np.array([0, 1]),
    }
    source_model = np.ones((2, 1, 2), dtype=np.complex128)
    disk_model = np.full((2, 1, 2), 2.0 + 0.0j, dtype=np.complex128)
    source_table = FakeTable({
        **{name: values.copy() for name, values in row_identity.items()},
        "DATA": source_model.copy(),
    })
    target_table = FakeTable({
        **{name: values.copy() for name, values in row_identity.items()},
        "MODEL_DATA": disk_model.copy(),
    })
    if mismatched_rows:
        target_table.columns["ANTENNA1"][1] = 9

    with (
        patch.object(pipeline, "tbtool", side_effect=[source_table, target_table]),
        patch.object(
            pipeline,
            "_ms_reference_frequencies",
            side_effect=[np.array([5.0e9, 5.3e9]), np.array([5.0e9, 5.3e9])],
        ),
    ):
        if mismatched_rows:
            with pytest.raises(RuntimeError, match="row alignment failed for ANTENNA1"):
                pipeline._add_aligned_model_data("residual.ms", "fine.ms")
        else:
            assert pipeline._add_aligned_model_data("residual.ms", "fine.ms") == 2

    if mismatched_rows:
        assert target_table.put_calls == []
        np.testing.assert_array_equal(target_table.columns["MODEL_DATA"], disk_model)
    else:
        assert target_table.put_calls == [("MODEL_DATA", 0, 2)]
        np.testing.assert_array_equal(
            target_table.columns["MODEL_DATA"], disk_model + source_model)


@pytest.mark.parametrize("all_spws_solved", [True, False])
def test_fine_selfcal_round_uses_local_spws_and_cumulative_calibration(
        pipeline, tmp_path, all_spws_solved):
    imaging = MagicMock()
    imaging.succeeded = True
    imaging.last_wsclean_status = 0
    seed_caltable = str(tmp_path / "fine-seed.pha")
    ri = {
        "N1": 12,
        "N2": 2,
        "time_intervals": ["minor-intervals"],
        "time_intervals_major_avg": ["major-averages"],
        "time_intervals_minor_avg": ["minor-averages"],
    }

    with (
        patch.object(pipeline, "MSselfcal", return_value=imaging) as selfcal_class,
        patch.object(pipeline, "gaincal") as gaincal_mock,
        patch.object(
            pipeline,
            "_caltable_has_unflagged_solutions_for_spws",
            return_value=all_spws_solved,
        ) as validate_mock,
        patch.object(pipeline, "applycal") as applycal_mock,
    ):
        if all_spws_solved:
            result = pipeline._run_fine_selfcal_round(
                msfile="fine.ms",
                fine_spw="13~14",
                fine_spwstr="13-14",
                local_casa_spw="0~1",
                local_sp_index="0,1",
                workdir=str(tmp_path),
                antenna="0~12",
                tdur=4,
                pols="XX",
                bmsize=20.0,
                ri=ri,
                seed_caltable=seed_caltable,
                fits_mask="fine-mask.fits",
                clearcache=False,
            )
        else:
            with pytest.raises(RuntimeError, match="every child SPW"):
                pipeline._run_fine_selfcal_round(
                    msfile="fine.ms",
                    fine_spw="13~14",
                    fine_spwstr="13-14",
                    local_casa_spw="0~1",
                    local_sp_index="0,1",
                    workdir=str(tmp_path),
                    antenna="0~12",
                    tdur=4,
                    pols="XX",
                    bmsize=20.0,
                    ri=ri,
                    seed_caltable=seed_caltable,
                    fits_mask="fine-mask.fits",
                    clearcache=False,
                )

    selfcal_args, selfcal_kwargs = selfcal_class.call_args
    assert selfcal_args == (
        "fine.ms",
        ri["time_intervals"],
        ri["time_intervals_major_avg"],
        ri["time_intervals_minor_avg"],
        "13~14",
        12,
        2,
        str(tmp_path),
    )
    assert selfcal_kwargs == {
        "image_marker": "fine_round1",
        "niter": 200,
        "briggs": 0.0,
        "auto_mask": 4,
        "auto_threshold": 2,
        "fits_mask": "fine-mask.fits",
        "pols": "XX",
        "beam_size": 20.0,
        "circular_beam": False,
        "data_column": "CORRECTED_DATA",
        "sp_index": "0,1",
    }
    imaging.run.assert_called_once_with(clearcache=False)
    gain_kwargs = gaincal_mock.call_args.kwargs
    assert gain_kwargs["vis"] == "fine.ms"
    assert gain_kwargs["spw"] == "0~1"
    assert gain_kwargs["gaintable"] == [seed_caltable]
    assert gain_kwargs["solint"] == "10s"
    assert gain_kwargs["calmode"] == "p"
    fine_caltable = str(tmp_path / "caltb_fine_round1_sp13-14.pha")
    validate_mock.assert_called_once_with(fine_caltable, [0, 1])

    if all_spws_solved:
        assert result == (imaging, fine_caltable)
        apply_kwargs = applycal_mock.call_args.kwargs
        assert apply_kwargs["vis"] == "fine.ms"
        assert apply_kwargs["spw"] == "0~1"
        assert apply_kwargs["gaintable"] == [seed_caltable, fine_caltable]
    else:
        applycal_mock.assert_not_called()


def test_fine_selfcal_round_rejects_nonzero_wsclean_status_before_gaincal(
        pipeline, tmp_path):
    imaging = MagicMock()
    imaging.succeeded = True
    imaging.last_wsclean_status = 9
    ri = {
        "N1": 12,
        "N2": 2,
        "time_intervals": [],
        "time_intervals_major_avg": [],
        "time_intervals_minor_avg": [],
    }

    with (
        patch.object(pipeline, "MSselfcal", return_value=imaging),
        patch.object(pipeline, "gaincal") as gaincal_mock,
        patch.object(pipeline, "applycal") as applycal_mock,
    ):
        with pytest.raises(RuntimeError, match="fine self-calibration imaging failed"):
            pipeline._run_fine_selfcal_round(
                msfile="fine.ms",
                fine_spw="13~14",
                fine_spwstr="13-14",
                local_casa_spw="0~1",
                local_sp_index="0,1",
                workdir=str(tmp_path),
                antenna="0~12",
                tdur=4,
                pols="XX",
                bmsize=20.0,
                ri=ri,
                seed_caltable=str(tmp_path / "fine-seed.pha"),
            )

    imaging.run.assert_called_once_with(clearcache=True)
    gaincal_mock.assert_not_called()
    applycal_mock.assert_not_called()


def test_corrected_bootstrap_lifecycle_runs_a_real_fine_round_before_disk_subtraction(
        pipeline, tmp_path):
    helper = _bootstrap_helper(pipeline)
    pre_disk_parent = str(tmp_path / "parent.pre-disk.ms")
    residual_parent = str(tmp_path / "parent.final-residual.ms")
    fine_ms = str(tmp_path / "fine-13-14.ms")
    residual_model_ms = str(tmp_path / "fine-13-14.residual-model.ms")
    Path(pre_disk_parent).mkdir()
    Path(residual_parent).mkdir()
    events = []
    calls = {}

    def fake_build(**kwargs):
        events.append("build_fullsky_child")
        calls["build"] = kwargs.copy()
        Path(kwargs["fine_msfile"]).mkdir()
        Path(kwargs["residual_model_ms"]).mkdir()
        return "0~1", "0,1"

    def fake_gaincal(**kwargs):
        events.append("seed_gaincal")
        calls["gaincal"] = kwargs.copy()
        Path(kwargs["caltable"]).mkdir()

    def fake_applycal(**kwargs):
        events.append("seed_applycal")
        calls["applycal"] = kwargs.copy()

    def fake_fine_round(**kwargs):
        events.append("fine_selfcal_round")
        calls["fine_round"] = kwargs.copy()
        return [kwargs["seed_caltable"], str(tmp_path / "fine-round.pha")]

    def fake_clear_model(**_kwargs):
        events.append("clear_fullsky_model")

    def fake_disk_predict(*_args, **_kwargs):
        events.append("disk_predict")

    def fake_uvsub(**kwargs):
        events.append("disk_uvsub")
        calls["uvsub"] = kwargs.copy()

    final_result = (["fine.fits"], {"fine:13-14": "image"})

    def fake_final_imaging(**kwargs):
        events.append("final_imaging")
        calls["final"] = kwargs.copy()
        return final_result

    with (
        patch.object(pipeline, "_build_fullsky_fine_child", side_effect=fake_build),
        patch.object(pipeline, "gaincal", side_effect=fake_gaincal),
        patch.object(pipeline, "_caltable_has_unflagged_solutions_for_spws", return_value=True),
        patch.object(pipeline, "applycal", side_effect=fake_applycal),
        patch.object(pipeline, "_run_fine_selfcal_round", side_effect=fake_fine_round),
        patch.object(pipeline, "delmod", side_effect=fake_clear_model),
        patch.object(pipeline, "_predict_fine_disk_model", side_effect=fake_disk_predict),
        patch.object(pipeline, "uvsub", side_effect=fake_uvsub),
        patch.object(pipeline, "_run_final_imaging", side_effect=fake_final_imaging),
        patch.object(pipeline, "_run_slfcal_round") as scratch_selfcal,
    ):
        result = helper(
            pre_disk_parent_ms=pre_disk_parent,
            residual_parent_ms=residual_parent,
            pre_disk_sp_index="2,3",
            residual_sp_index="13,14",
            fine_spw="13~14",
            fine_spwstr="13-14",
            fine_msfile=fine_ms,
            residual_model_ms=residual_model_ms,
            disk_model_prefixes=[
                str(tmp_path / "disk-sp13"),
                str(tmp_path / "disk-sp14"),
            ],
            workdir=str(tmp_path),
            antenna="0~12",
            pols="XX",
            bmsize=20.0,
            fine_round_ri={"N1": 12, "N2": 1},
            tdur=240,
            final_imaging_kwargs={"sidx": "fine:13-14"},
        )

    assert result == final_result
    assert events[0:4] == [
        "build_fullsky_child",
        "seed_gaincal",
        "seed_applycal",
        "fine_selfcal_round",
    ]
    assert events[4:7] == [
        "clear_fullsky_model",
        "disk_predict",
        "disk_uvsub",
    ]
    assert events[7:] == ["final_imaging"]
    scratch_selfcal.assert_not_called()
    assert calls["build"]["pre_disk_parent_ms"] == pre_disk_parent
    assert calls["build"]["residual_parent_ms"] == residual_parent
    assert calls["gaincal"]["gaintable"] == []
    assert calls["gaincal"]["spw"] == "0~1"
    assert calls["gaincal"]["calmode"] == "p"
    assert calls["applycal"]["gaintable"] == [calls["gaincal"]["caltable"]]
    assert calls["fine_round"]["seed_caltable"] == calls["gaincal"]["caltable"]
    assert calls["uvsub"]["vis"] == fine_ms
    assert calls["final"]["data_column"] == "CORRECTED_DATA"
    assert not Path(fine_ms).exists()
    assert not Path(residual_model_ms).exists()


def test_missing_predisk_parent_state_skips_without_scratch_fallback(pipeline, tmp_path):
    helper = _bootstrap_helper(pipeline)
    missing_pre_disk_parent = str(tmp_path / "missing.pre-disk.ms")
    residual_parent = str(tmp_path / "parent.final-residual.ms")
    Path(residual_parent).mkdir()

    with (
        patch.object(
            pipeline,
            "_build_fullsky_fine_child",
            side_effect=RuntimeError("pre-disk parent checkpoint is unavailable"),
        ) as build_mock,
        patch.object(pipeline, "gaincal") as gaincal_mock,
        patch.object(pipeline, "applycal") as applycal_mock,
        patch.object(pipeline, "_run_fine_selfcal_round") as fine_round_mock,
        patch.object(pipeline, "_run_final_imaging") as final_imaging_mock,
        patch.object(pipeline, "_run_slfcal_round") as scratch_selfcal,
    ):
        result = helper(
            pre_disk_parent_ms=missing_pre_disk_parent,
            residual_parent_ms=residual_parent,
            pre_disk_sp_index="2,3",
            residual_sp_index="13,14",
            fine_spw="13~14",
            fine_spwstr="13-14",
            fine_msfile=str(tmp_path / "fine-13-14.ms"),
            residual_model_ms=str(tmp_path / "fine-13-14.residual-model.ms"),
            disk_model_prefixes=["disk-sp13", "disk-sp14"],
            workdir=str(tmp_path),
            antenna="0~12",
            pols="XX",
            bmsize=20.0,
            fine_round_ri={"N1": 12, "N2": 1},
            tdur=240,
            final_imaging_kwargs={"sidx": "fine:13-14"},
    )

    assert result is None
    build_mock.assert_called_once()
    gaincal_mock.assert_not_called()
    applycal_mock.assert_not_called()
    fine_round_mock.assert_not_called()
    final_imaging_mock.assert_not_called()
    scratch_selfcal.assert_not_called()


@pytest.mark.parametrize(
    "builder_error",
    ["row alignment mismatch", "residual model missing child SPW signal"],
)
def test_child_build_failure_skips_before_seed_and_cleans_without_scratch_fallback(
        pipeline, tmp_path, builder_error):
    helper = _bootstrap_helper(pipeline)
    pre_disk_parent = str(tmp_path / "parent.pre-disk.ms")
    residual_parent = str(tmp_path / "parent.final-residual.ms")
    fine_ms = str(tmp_path / "fine-13-14.ms")
    residual_model_ms = str(tmp_path / "fine-13-14.residual-model.ms")
    Path(pre_disk_parent).mkdir()
    Path(residual_parent).mkdir()

    def fail_child_build(**_kwargs):
        Path(fine_ms).mkdir()
        Path(residual_model_ms).mkdir()
        raise RuntimeError(builder_error)

    with (
        patch.object(pipeline, "_build_fullsky_fine_child", side_effect=fail_child_build),
        patch.object(pipeline, "gaincal") as gaincal_mock,
        patch.object(pipeline, "applycal") as applycal_mock,
        patch.object(pipeline, "_run_fine_selfcal_round") as fine_round_mock,
        patch.object(pipeline, "_run_final_imaging") as final_imaging_mock,
        patch.object(pipeline, "_run_slfcal_round") as scratch_selfcal,
    ):
        result = helper(
            pre_disk_parent_ms=pre_disk_parent,
            residual_parent_ms=residual_parent,
            pre_disk_sp_index="2,3",
            residual_sp_index="13,14",
            fine_spw="13~14",
            fine_spwstr="13-14",
            fine_msfile=fine_ms,
            residual_model_ms=residual_model_ms,
            disk_model_prefixes=["disk-sp13", "disk-sp14"],
            workdir=str(tmp_path),
            antenna="0~12",
            pols="XX",
            bmsize=20.0,
            fine_round_ri={"N1": 12, "N2": 1},
            tdur=240,
            final_imaging_kwargs={"sidx": "fine:13-14"},
        )

    assert result is None
    gaincal_mock.assert_not_called()
    applycal_mock.assert_not_called()
    fine_round_mock.assert_not_called()
    final_imaging_mock.assert_not_called()
    scratch_selfcal.assert_not_called()
    assert not Path(fine_ms).exists()
    assert not Path(residual_model_ms).exists()


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
