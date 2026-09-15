import json

import numpy as np
import pytest

from suncasa.eovsa import antenna_flagging


class FakeTableTool:
    """Small CASA table double supporting fixed and variable FLAG cells."""

    def __init__(self, tables, absolute_var_keys=False):
        self.tables = tables
        self.current = None
        self.absolute_var_keys = absolute_var_keys
        self.putcell_calls = []

    def open(self, path, nomodify=True):
        self.current = self.tables[path]

    def close(self):
        self.current = None

    def nrows(self):
        for value in self.current.values():
            if isinstance(value, np.ndarray) and value.ndim:
                return value.shape[-1] if value.ndim > 1 else len(value)
            if isinstance(value, list):
                return len(value)
        return 0

    def getcol(self, name, startrow=None, nrow=None):
        value = self.current[name]
        if startrow is None:
            return value.copy() if hasattr(value, 'copy') else value
        if isinstance(value, list):
            return np.asarray(value[startrow:startrow + nrow]).copy()
        if value.ndim == 1:
            return value[startrow:startrow + nrow].copy()
        return value[..., startrow:startrow + nrow].copy()

    def getvarcol(self, name, startrow, nrow):
        value = self.current[name]

        def key(index):
            offset = startrow if self.absolute_var_keys else 0
            return 'r{}'.format(offset + index + 1)

        if isinstance(value, list):
            return {
                key(index): value[startrow + index][..., None].copy()
                for index in range(nrow)
            }
        return {
            key(index): value[..., startrow + index][..., None].copy()
            for index in range(nrow)
        }

    def putcell(self, name, row, value):
        target = self.current[name]
        self.putcell_calls.append((name, row, np.asarray(value).copy()))
        if isinstance(target, list):
            target[row] = value
        else:
            target[..., row] = value

    def putcol(self, name, value, startrow, nrow):
        target = self.current[name]
        if target.ndim == 1:
            target[startrow:startrow + nrow] = value
        else:
            target[..., startrow:startrow + nrow] = value


def utc_seconds(value):
    return antenna_flagging._utc_seconds(value)


def make_fake_ms(tmp_path, times=None, basename='UDB20260905.ms'):
    msfile = str(tmp_path / basename)
    main_path = msfile
    antenna_path = msfile + '/ANTENNA'
    if times is None:
        times = np.arange(5, dtype=float)
    times = np.asarray(times, dtype=float)
    nrow = len(times)
    tables = {
        antenna_path: {
            'NAME': np.array([
                b'EOVSA Ant 1', b'EOVSA Ant 2', b'EOVSA Ant 3', b'Ant 4']),
        },
        main_path: {
            'TIME': times,
            'INTERVAL': np.ones(nrow, dtype=float),
            'ANTENNA1': np.resize(np.array([0, 1, 2, 2, 0], dtype=int), nrow),
            'ANTENNA2': np.resize(np.array([1, 2, 3, 0, 2], dtype=int), nrow),
            'FLAG': np.zeros((2, 2, nrow), dtype=bool),
            'DATA': np.full((2, 2, nrow), 7.0, dtype=float),
            'CORRECTED_DATA': np.full((2, 2, nrow), 8.0, dtype=float),
            'WEIGHT': np.full((2, nrow), 9.0, dtype=float),
        },
    }
    return msfile, tables


def test_integration_overlap_respects_partial_rows_and_boundaries():
    assert antenna_flagging.integration_overlap_mask(
        [1.0, 2.0, 3.0], [(1.25, 2.25)], [1.0, 1.0, 1.0]
    ).tolist() == [True, True, False]
    assert antenna_flagging.integration_overlap_mask(
        [1.0], [(1.5, 2.0)], [1.0]
    ).tolist() == [False]


def test_flag_ant3_intervals_is_chunked_idempotent_and_data_safe(tmp_path):
    msfile, tables = make_fake_ms(tmp_path)
    original = {key: value.copy() for key, value in tables[msfile].items()}
    tool = FakeTableTool(tables)
    provenance = tmp_path / 'ant3_flags.json'

    first = antenna_flagging.flag_ant3_stow(
        msfile, intervals=[(1.25, 2.25)], table_tool=tool,
        row_chunk=2, provenance_path=str(provenance))
    assert first['resolved_antenna_id'] == 2
    assert first['rows_selected'] == 2
    assert first['rows_changed'] == 2
    assert first['cells_newly_flagged'] == 8
    assert tables[msfile]['FLAG'][..., 1].all()
    assert tables[msfile]['FLAG'][..., 2].all()
    assert not tables[msfile]['FLAG'][..., 0].any()
    assert not tables[msfile]['FLAG'][..., 3].any()
    for key in ('TIME', 'INTERVAL', 'ANTENNA1', 'ANTENNA2', 'DATA',
                'CORRECTED_DATA', 'WEIGHT'):
        np.testing.assert_array_equal(tables[msfile][key], original[key])

    second = antenna_flagging.flag_ant3_stow(
        msfile, intervals=[(1.25, 2.25)], table_tool=tool, row_chunk=2)
    assert second['cells_newly_flagged'] == 0
    assert second['rows_changed'] == 0
    assert json.loads(provenance.read_text())['operation'] == 'or_flag'


def test_flag_ant3_handles_variable_flag_shapes_per_row(tmp_path):
    msfile, tables = make_fake_ms(tmp_path)
    tables[msfile]['FLAG'] = [
        np.zeros((2, 2), dtype=bool),
        np.zeros((4, 1), dtype=bool),
        np.zeros((3, 2), dtype=bool),
        np.zeros((3, 2), dtype=bool),
        np.zeros((2, 2), dtype=bool),
    ]
    report = antenna_flagging.flag_ant3_stow(
        msfile, intervals=[(1.25, 2.25)],
        table_tool=FakeTableTool(tables), row_chunk=2)
    assert report['cells_newly_flagged'] == 10
    assert tables[msfile]['FLAG'][1].all()
    assert tables[msfile]['FLAG'][2].all()
    assert not tables[msfile]['FLAG'][0].any()


def test_flag_ant3_handles_absolute_variable_row_keys(tmp_path):
    msfile, tables = make_fake_ms(tmp_path)
    tables[msfile]['FLAG'] = [np.zeros((2, 2), dtype=bool) for _ in range(5)]
    tables[msfile]['FLAG'][1][0, 0] = True
    tool = FakeTableTool(tables, absolute_var_keys=True)
    report = antenna_flagging.flag_ant3_stow(
        msfile, intervals=[(2.25, 3.25)], table_tool=tool, row_chunk=2)
    assert report['rows_selected'] == 2
    assert tables[msfile]['FLAG'][2].all()
    assert tables[msfile]['FLAG'][3].all()
    assert tables[msfile]['FLAG'][1][0, 0]
    assert len(tool.putcell_calls) == 2
    assert [row for _, row, _ in tool.putcell_calls] == [2, 3]


def test_resolve_physical_antenna_requires_unique_name(tmp_path):
    msfile, tables = make_fake_ms(tmp_path)
    tables[msfile + '/ANTENNA']['NAME'] = np.array(
        [b'ANT1', b'ANT3', b'ANT3', b'ANT4'])
    with pytest.raises(ValueError, match='2 MS rows'):
        antenna_flagging.resolve_physical_antenna_id(
            msfile, table_tool=FakeTableTool(tables))


def test_resolve_physical_antenna_accepts_eovsa_eo_names(tmp_path):
    msfile, tables = make_fake_ms(tmp_path)
    tables[msfile + '/ANTENNA']['NAME'] = np.array(
        [b'eo01', b'eo02', b'eo03', b'eo04'])
    assert antenna_flagging.resolve_physical_antenna_id(
        msfile, table_tool=FakeTableTool(tables)) == 2


def test_empty_manifest_fails_with_incomplete_marker(tmp_path, monkeypatch):
    times = [utc_seconds('2026-09-05T14:23:00Z')]
    msfile, tables = make_fake_ms(tmp_path, times=times)
    original = tables[msfile]['FLAG'].copy()
    provenance = tmp_path / 'ant3_flags.json'
    monkeypatch.setattr(antenna_flagging, 'HISTORICAL_NO_ANT3_INTERVALS_UTC', ())

    with pytest.raises(RuntimeError, match=antenna_flagging.POLICY_ID):
        antenna_flagging.flag_ant3_stow(
            msfile, table_tool=FakeTableTool(tables),
            provenance_path=str(provenance))
    report = json.loads(provenance.read_text())
    assert report['operation'] == 'incomplete'
    assert report['skip_reason'] == 'historical_no_ant3_interval_manifest_empty'
    assert report['policy_id'] == antenna_flagging.POLICY_ID
    np.testing.assert_array_equal(tables[msfile]['FLAG'], original)


def test_automatic_selector_uses_time_for_idb_and_udb_names(tmp_path, monkeypatch):
    interval = ('2026-09-05T14:22:00Z', '2026-09-05T14:24:00Z')
    monkeypatch.setattr(antenna_flagging,
                        'HISTORICAL_NO_ANT3_INTERVALS_UTC', (interval,))
    times = [utc_seconds('2026-09-05T14:23:00Z')]
    reports = []
    for basename in ('UDB20260905.ms', 'IDB20260905.ms'):
        msfile, tables = make_fake_ms(tmp_path, times=times, basename=basename)
        report = antenna_flagging.flag_ant3_stow(
            msfile, table_tool=FakeTableTool(tables))
        reports.append(report)
        assert report['operation'] == 'or_flag'
        assert report['interval_source'] == 'verified_historical_no_ant3'
        assert report['policy_id'] == antenna_flagging.POLICY_ID
        assert report['matched_intervals_utc'] == [list(interval)]
        assert report['cells_newly_flagged'] == 0
    assert reports[0]['source_ms'].endswith('UDB20260905.ms')
    assert reports[1]['source_ms'].endswith('IDB20260905.ms')


def test_automatic_selector_flags_first_and_last_intervals(tmp_path, monkeypatch):
    intervals = (
        ('2026-09-05T14:22:00Z', '2026-09-05T14:24:00Z'),
        ('2026-09-06T01:20:00Z', '2026-09-06T01:24:00Z'),
    )
    monkeypatch.setattr(antenna_flagging,
                        'HISTORICAL_NO_ANT3_INTERVALS_UTC', intervals)
    times = [
        utc_seconds('2026-09-05T14:23:00Z'),
        utc_seconds('2026-09-05T15:00:00Z'),
        utc_seconds('2026-09-06T01:22:00Z'),
    ]
    msfile, tables = make_fake_ms(tmp_path, times=times)
    tables[msfile]['ANTENNA1'][:] = [2, 1, 2]
    tables[msfile]['ANTENNA2'][:] = [3, 2, 3]
    report = antenna_flagging.flag_ant3_stow(
        msfile, table_tool=FakeTableTool(tables), row_chunk=2)
    assert report['operation'] == 'or_flag'
    assert report['matched_intervals_utc'] == [list(item) for item in intervals]
    assert report['rows_selected'] == 2
    assert report['cells_newly_flagged'] == 8
    assert tables[msfile]['FLAG'][..., 0].all()
    assert tables[msfile]['FLAG'][..., 2].all()


def test_midday_and_before_after_policy_are_noop(tmp_path, monkeypatch):
    interval = ('2026-09-05T14:22:00Z', '2026-09-05T14:24:00Z')
    monkeypatch.setattr(antenna_flagging,
                        'HISTORICAL_NO_ANT3_INTERVALS_UTC', (interval,))
    cases = [
        [utc_seconds('2026-09-05T12:00:00Z')],
        [utc_seconds('2026-08-31T23:59:00Z')],
        [utc_seconds('2026-09-09T07:00:00Z')],
    ]
    for index, times in enumerate(cases):
        msfile, tables = make_fake_ms(tmp_path, times=times,
                                      basename='case{}.ms'.format(index))
        original = tables[msfile]['FLAG'].copy()
        report = antenna_flagging.flag_ant3_stow(
            msfile, table_tool=FakeTableTool(tables))
        assert report['operation'] == 'skip'
        assert report['skip_reason'] == 'outside_historical_no_ant3_intervals'
        assert report['cells_newly_flagged'] == 0
        np.testing.assert_array_equal(tables[msfile]['FLAG'], original)


def test_sep9_pre07_rollover_is_selected(tmp_path, monkeypatch):
    interval = ('2026-09-09T06:00:00Z', '2026-09-09T06:05:00Z')
    monkeypatch.setattr(antenna_flagging,
                        'HISTORICAL_NO_ANT3_INTERVALS_UTC', (interval,))
    msfile, tables = make_fake_ms(
        tmp_path, times=[utc_seconds('2026-09-09T06:02:00Z')])
    report = antenna_flagging.flag_ant3_stow(
        msfile, table_tool=FakeTableTool(tables))
    assert report['operation'] == 'or_flag'
    assert report['matched_intervals_utc'] == [list(interval)]
    assert report['cells_newly_flagged'] == 0


def test_sep9_morning_interval_flags_ant3_only(tmp_path):
    times = [
        utc_seconds('2026-09-09T14:30:00Z'),
        utc_seconds('2026-09-09T14:30:00Z'),
        utc_seconds('2026-09-09T14:55:22Z'),
        utc_seconds('2026-09-09T15:01:00Z'),
    ]
    msfile, tables = make_fake_ms(
        tmp_path, times=times, basename='UDB20260909.ms')
    tables[msfile]['ANTENNA1'][:] = [2, 0, 2, 2]
    tables[msfile]['ANTENNA2'][:] = [0, 1, 0, 0]

    report = antenna_flagging.flag_ant3_stow(
        msfile, table_tool=FakeTableTool(tables), row_chunk=2)

    assert report['policy_id'] == 'ant3-no-ant3-20260901-20260909-v3'
    assert report['matched_intervals_utc'] == [[
        '2026-09-09T14:25:03.5Z', '2026-09-09T14:55:21.5Z']]
    assert report['rows_selected'] == 1
    assert tables[msfile]['FLAG'][..., 0].all()
    assert not tables[msfile]['FLAG'][..., 1].any()
    assert not tables[msfile]['FLAG'][..., 2].any()
    assert not tables[msfile]['FLAG'][..., 3].any()


def test_boundary_overlap_uses_integration_coverage(tmp_path, monkeypatch):
    interval = ('2026-09-05T14:22:00Z', '2026-09-05T14:23:00Z')
    monkeypatch.setattr(antenna_flagging,
                        'HISTORICAL_NO_ANT3_INTERVALS_UTC', (interval,))
    msfile, tables = make_fake_ms(tmp_path, times=[
        utc_seconds('2026-09-05T14:21:30Z'),
        utc_seconds('2026-09-05T14:22:30Z'),
    ])
    tables[msfile]['INTERVAL'][:] = 60.0
    report = antenna_flagging.flag_ant3_stow(
        msfile, table_tool=FakeTableTool(tables))
    assert report['rows_selected'] == 1
    assert tables[msfile]['FLAG'][..., 0].sum() == 0
    assert tables[msfile]['FLAG'][..., 1].all()


def test_manifest_rejects_naive_or_out_of_policy_timestamps(monkeypatch):
    monkeypatch.setattr(antenna_flagging,
                        'HISTORICAL_NO_ANT3_INTERVALS_UTC',
                        (('2026-09-05T14:22:00', '2026-09-05T14:24:00Z'),))
    with pytest.raises(ValueError, match='timestamp'):
        antenna_flagging._historical_intervals()

    monkeypatch.setattr(antenna_flagging,
                        'HISTORICAL_NO_ANT3_INTERVALS_UTC',
                        (('2026-08-31T23:00:00Z', '2026-09-01T01:00:00Z'),))
    with pytest.raises(ValueError, match='outside'):
        antenna_flagging._historical_intervals()


def test_sep9_evening_only_does_not_extend_into_sep10_morning(tmp_path):
    times = [utc_seconds(value) for value in (
        '2026-09-10T00:50:00Z', '2026-09-10T00:50:00Z',
        '2026-09-10T14:30:00Z')]
    msfile, tables = make_fake_ms(tmp_path, times=times)
    tables[msfile]['ANTENNA1'][:] = [2, 0, 2]
    tables[msfile]['ANTENNA2'][:] = [0, 1, 0]
    report = antenna_flagging.flag_ant3_stow(msfile, table_tool=FakeTableTool(tables))
    assert report['rows_selected'] == 1
    assert report['matched_intervals_utc'] == [[
        '2026-09-10T00:47:00.5Z', '2026-09-10T07:00:00Z']]
    assert tables[msfile]['FLAG'][..., 0].all()
    assert not tables[msfile]['FLAG'][..., 1:].any()
