"""Time-dependent antenna flagging helpers for EOVSA measurement sets.

The functions in this module deliberately operate on the CASA ``FLAG``
column only.  They do not alter visibility, calibration, or weight columns.
"""

from __future__ import absolute_import

import json
from datetime import datetime, timezone
import os
import re

import numpy as np


_ANT3_POLICY_START_MJD = 61284.0  # 2026-09-01T00:00:00 UTC
_ANT3_POLICY_END_MJD = 61293.0 + 7.0 / 24.0  # End of Sep 9 observing day
POLICY_ID = 'ant3-no-ant3-20260901-20260909-v3'

# Historical SUN_NO_ANT3 / STOW_ANT3 episodes, verified against one-second
# primary SQL records. Boundary brackets are one second wide. Night intervals
# end at the 07:00 UTC observing-day cutoff, not a measured tracking transition.
# Evidence for the original 16 intervals: historical_noant3_intervals.json, SHA256
# 35d8929dbc14c55c26df41c68d50ce0cc6d25eaa8d6edcb40c62b6ebf6cb5b50
# Sep 9 addition: output/calibration-comparison-20260905/ant3-remedy/september9/
# september9_noant3_interval.json, SHA256
# 27d3866a6cb6046018ab89ab4c8d50f5c3a7ac496a3f2a704b570e342bb960e6
# Sep 9 evening: september9-evening/interval.json; SQL boundary 00:47:00--01.
HISTORICAL_NO_ANT3_INTERVALS_UTC = (
    ('2026-09-01T14:18:03.5Z', '2026-09-01T14:48:21.5Z'),
    ('2026-09-02T00:58:00.5Z', '2026-09-02T07:00:00Z'),
    ('2026-09-02T14:19:03.5Z', '2026-09-02T14:49:21.5Z'),
    ('2026-09-03T00:57:00.5Z', '2026-09-03T07:00:00Z'),
    ('2026-09-03T14:20:03.5Z', '2026-09-03T14:50:21.5Z'),
    ('2026-09-04T00:56:00.5Z', '2026-09-04T07:00:00Z'),
    ('2026-09-04T14:21:03.5Z', '2026-09-04T14:51:21.5Z'),
    ('2026-09-05T00:54:00.5Z', '2026-09-05T07:00:00Z'),
    ('2026-09-05T14:22:03.5Z', '2026-09-05T14:51:21.5Z'),
    ('2026-09-06T00:53:00.5Z', '2026-09-06T07:00:00Z'),
    ('2026-09-06T14:22:03.5Z', '2026-09-06T14:52:21.5Z'),
    ('2026-09-07T00:51:00.5Z', '2026-09-07T07:00:00Z'),
    ('2026-09-07T14:23:03.5Z', '2026-09-07T14:53:21.5Z'),
    ('2026-09-08T00:50:00.5Z', '2026-09-08T07:00:00Z'),
    ('2026-09-08T14:24:03.5Z', '2026-09-08T14:54:21.5Z'),
    ('2026-09-09T00:48:00.5Z', '2026-09-09T07:00:00Z'),
    ('2026-09-09T14:25:03.5Z', '2026-09-09T14:55:21.5Z'),
    ('2026-09-10T00:47:00.5Z', '2026-09-10T07:00:00Z'),
)


_ANTENNA_NAME_RE = re.compile(
    r"^(?:(?:EOVSA)?ANT(?:ENNA)?|EO)0*(\d+)$", re.IGNORECASE)


def _as_float_vector(values, name):
    """Return finite one-dimensional numeric values."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains non-finite values".format(name))
    return array


def normalize_intervals(intervals):
    """Validate, sort, and merge half-open exclusion intervals.

    :param intervals: Iterable of ``(start, end)`` pairs in the same numeric
        time unit as the MS ``TIME`` column.
    :type intervals: iterable
    :returns: Sorted, merged floating-point intervals.
    :rtype: list[tuple[float, float]]
    :raises ValueError: If an interval is malformed, non-finite, or reversed.
    """
    if intervals is None:
        return []
    values = list(intervals)
    if not values:
        return []
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("intervals must contain (start, end) pairs")
    if not np.all(np.isfinite(array)):
        raise ValueError("intervals contain non-finite values")
    if np.any(array[:, 1] < array[:, 0]):
        raise ValueError("interval end precedes interval start")

    array = array[np.argsort(array[:, 0], kind='mergesort')]
    merged = []
    for start, end in array:
        start = float(start)
        end = float(end)
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def integration_overlap_mask(times, intervals, integration_seconds):
    """Return rows whose integration intervals overlap an exclusion interval.

    ``times`` and ``integration_seconds`` are MS row-center times and row
    durations in the same unit as ``intervals``.  Positive-length intervals
    use strict positive overlap, so a row ending exactly at a transition is
    retained.  A zero-length exclusion interval matches a row whose center is
    exactly inside it.

    :param times: Row-center times.
    :type times: array-like
    :param intervals: Sorted or unsorted ``(start, end)`` exclusions.
    :type intervals: iterable
    :param integration_seconds: Row integration durations.
    :type integration_seconds: array-like
    :returns: Boolean row mask.
    :rtype: numpy.ndarray
    :raises ValueError: If inputs have incompatible lengths or invalid values.
    """
    times = _as_float_vector(times, 'times')
    durations = _as_float_vector(integration_seconds, 'integration_seconds')
    if len(times) != len(durations):
        raise ValueError("times and integration_seconds must have equal length")
    if np.any(durations < 0):
        raise ValueError("integration_seconds cannot be negative")

    result = np.zeros(len(times), dtype=bool)
    left = times - durations / 2.0
    right = times + durations / 2.0
    for start, end in normalize_intervals(intervals):
        if start == end:
            result |= (times >= start) & (times <= end)
        else:
            result |= (right > start) & (left < end)
    return result


def _antenna_number(name):
    """Extract a physical antenna number from a conventional MS name."""
    if isinstance(name, bytes):
        name = name.decode('ascii', 'replace')
    compact = re.sub(r'[^A-Za-z0-9]', '', str(name).strip())
    if compact.isdigit():
        return int(compact)
    match = _ANTENNA_NAME_RE.match(compact)
    return None if match is None else int(match.group(1))


def resolve_physical_antenna_id(msfile, physical_antenna=3, table_tool=None):
    """Resolve a physical antenna number through the MS ``ANTENNA`` table.

    :param msfile: Measurement Set directory.
    :type msfile: str
    :param physical_antenna: One-based physical antenna number.
    :type physical_antenna: int
    :param table_tool: Optional CASA table tool or test double.
    :type table_tool: object, optional
    :returns: Zero-based CASA antenna-table row ID.
    :rtype: int
    :raises ValueError: If the physical antenna is absent or ambiguous.
    """
    physical_antenna = int(physical_antenna)
    if physical_antenna < 1:
        raise ValueError("physical_antenna must be one-based and positive")
    tool = _table_tool(table_tool)
    path = os.path.join(os.fspath(msfile), 'ANTENNA')
    _open_table(tool, path, nomodify=True)
    try:
        names = np.asarray(tool.getcol('NAME')).reshape(-1)
    finally:
        tool.close()
    matches = [index for index, name in enumerate(names)
               if _antenna_number(name) == physical_antenna]
    if len(matches) != 1:
        raise ValueError(
            "physical antenna {} resolves to {} MS rows in {}".format(
                physical_antenna, len(matches), path))
    return int(matches[0])


def _table_tool(table_tool):
    if table_tool is not None:
        return table_tool
    try:
        from casatools import table
    except ImportError:
        try:
            from taskinit import tb as table
        except ImportError:
            raise ImportError("CASA table tool is required to edit a Measurement Set")
    return table()


def _open_table(tool, path, nomodify):
    try:
        tool.open(path, nomodify=nomodify)
    except TypeError:
        tool.open(path)


def _write_provenance(path, payload):
    if path is None:
        return
    path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = path + '.tmp-{}'.format(os.getpid())
    with open(temporary, 'w') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def _varcol_row_index(key, startrow, nread):
    """Return a chunk-local row index and CASA key style.

    CASA returns absolute keys (for example ``r4097``) when ``getvarcol``
    starts at row 4096.  Test doubles and older table wrappers can return
    chunk-relative keys (``r1``), so accept and preserve either convention.
    """
    match = re.match(r'^r(\d+)$', str(key))
    if match is None:
        raise ValueError("unexpected CASA variable-column key {}".format(key))
    value = int(match.group(1)) - 1
    if startrow <= value < startrow + nread:
        return value - startrow, 'absolute'
    if 0 <= value < nread:
        return value, 'relative'
    if value < 0 or value >= nread:
        raise ValueError("CASA variable-column row key is outside the chunk")
    raise ValueError("CASA variable-column row key is outside the chunk")


def _get_flag_rows(tool, startrow, nread):
    """Read FLAG rows without assuming a common DDID channel shape."""
    getvarcol = getattr(tool, 'getvarcol', None)
    if getvarcol is not None:
        values = getvarcol('FLAG', startrow, nread)
        rows = {}
        key_style = None
        for key, value in values.items():
            index, style = _varcol_row_index(key, startrow, nread)
            if key_style is None:
                key_style = style
            elif key_style != style:
                raise ValueError("CASA FLAG variable column mixed row-key styles")
            value = np.asarray(value, dtype=bool)
            if value.ndim == 3 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.ndim != 2:
                raise ValueError(
                    "CASA FLAG cell must be (polarization, channel)"
                )
            rows[index] = value.copy()
        if len(rows) != nread:
            raise ValueError("CASA FLAG variable column omitted rows")
        return rows, True, key_style
    flags = np.asarray(tool.getcol('FLAG', startrow, nread), dtype=bool)
    if flags.ndim < 1 or flags.shape[-1] != nread:
        raise ValueError("FLAG row axis is not the final axis")
    return {index: flags[..., index].copy() for index in range(nread)}, False, None


def _put_flag_rows(tool, startrow, nread, rows, variable, key_style=None):
    """Write FLAG rows using the matching CASA fixed/variable API.

    Variable-shape rows are written one cell at a time.  CASA's variable
    column dictionary keys are absolute for reads, while write behavior has
    varied between table-tool versions; ``putcell`` avoids key remapping and
    cannot reorder neighboring rows.
    """
    if variable:
        putcell = getattr(tool, 'putcell', None)
        if putcell is None:
            raise ValueError("CASA table lacks putcell for variable FLAG rows")
        for index, value in rows.items():
            putcell('FLAG', startrow + index, value)
        return
    first = next(iter(rows.values()))
    shape = first.shape + (nread,)
    flags = np.zeros(shape, dtype=bool)
    for index, value in rows.items():
        flags[..., index] = value
    tool.putcol('FLAG', flags, startrow, nread)


def flag_antenna_intervals(msfile, intervals, physical_antenna=3,
                           table_tool=None, row_chunk=4096,
                           provenance_path=None):
    """OR time-dependent physical-antenna flags into an MS copy.

    Rows are selected when either baseline endpoint is the resolved physical
    antenna and the row integration overlaps an exclusion interval.  Only the
    main-table ``FLAG`` column is written.  Repeated calls are idempotent and
    report zero newly flagged cells after the first call.

    :param msfile: Measurement Set directory to modify.  Callers should pass
        a pipeline-owned staging copy, never the durable imported source.
    :type msfile: str
    :param intervals: ``(start, end)`` intervals in the MS ``TIME`` unit.
    :type intervals: iterable
    :param physical_antenna: One-based physical antenna number.
    :type physical_antenna: int
    :param table_tool: Optional CASA table tool or test double.
    :type table_tool: object, optional
    :param row_chunk: Number of rows read per CASA table operation.
    :type row_chunk: int
    :param provenance_path: Optional JSON sidecar path.
    :type provenance_path: str, optional
    :returns: Counts and provenance for this operation.
    :rtype: dict
    :raises ValueError: If the MS columns, antenna mapping, or intervals are invalid.
    """
    if int(row_chunk) <= 0:
        raise ValueError("row_chunk must be positive")
    intervals = normalize_intervals(intervals)
    report = {
        'schema_version': 1,
        'operation': 'or_flag',
        'source_ms': os.path.abspath(os.fspath(msfile)),
        'physical_antenna': int(physical_antenna),
        'intervals': [[start, end] for start, end in intervals],
        'rows_scanned': 0,
        'rows_selected': 0,
        'rows_changed': 0,
        'cells_newly_flagged': 0,
    }
    if not intervals:
        _write_provenance(provenance_path, report)
        return report

    tool = _table_tool(table_tool)
    antenna_id = resolve_physical_antenna_id(
        msfile, physical_antenna=physical_antenna, table_tool=tool)
    antenna_path = os.path.join(os.fspath(msfile), 'ANTENNA')
    _open_table(tool, antenna_path, nomodify=True)
    try:
        antenna_count = int(tool.nrows())
    finally:
        tool.close()
    report['resolved_antenna_id'] = antenna_id
    _open_table(tool, os.fspath(msfile), nomodify=False)
    try:
        nrow = int(tool.nrows())
        for startrow in range(0, nrow, int(row_chunk)):
            nread = min(int(row_chunk), nrow - startrow)
            time = _as_float_vector(tool.getcol('TIME', startrow, nread), 'TIME')
            interval = _as_float_vector(
                tool.getcol('INTERVAL', startrow, nread), 'INTERVAL')
            ant1 = np.asarray(tool.getcol('ANTENNA1', startrow, nread), dtype=int).reshape(-1)
            ant2 = np.asarray(tool.getcol('ANTENNA2', startrow, nread), dtype=int).reshape(-1)
            if not (len(time) == len(interval) == len(ant1) == len(ant2) == nread):
                raise ValueError("MS row columns have inconsistent chunk lengths")
            if (np.any(ant1 < 0) or np.any(ant2 < 0) or
                    np.any(ant1 >= antenna_count) or np.any(ant2 >= antenna_count)):
                raise ValueError("MS contains an antenna ID outside the ANTENNA table")
            selected = integration_overlap_mask(time, intervals, interval)
            selected &= (ant1 == antenna_id) | (ant2 == antenna_id)
            flag_rows, variable_flags, variable_key_style = _get_flag_rows(
                tool, startrow, nread)
            report['rows_scanned'] += nread
            report['rows_selected'] += int(np.count_nonzero(selected))
            changed_rows = {}
            if np.any(selected):
                for index in np.flatnonzero(selected):
                    old = flag_rows[int(index)].copy()
                    flag_rows[int(index)][...] = True
                    report['cells_newly_flagged'] += int(np.count_nonzero(~old))
                    if np.any(~old):
                        changed_rows[int(index)] = flag_rows[int(index)]
                        report['rows_changed'] += 1
            if changed_rows:
                rows_to_write = changed_rows if variable_flags else flag_rows
                _put_flag_rows(tool, startrow, nread, rows_to_write, variable_flags,
                               variable_key_style)
    finally:
        tool.close()
    _write_provenance(provenance_path, report)
    return report


def _measurement_set_time_bounds(msfile, table_tool=None, row_chunk=4096):
    """Return integration-covered time bounds from an MS main table."""
    tool = _table_tool(table_tool)
    _open_table(tool, os.fspath(msfile), nomodify=True)
    try:
        nrow = int(tool.nrows())
        if nrow == 0:
            raise ValueError("MS main table is empty")
        earliest = None
        latest = None
        for startrow in range(0, nrow, int(row_chunk)):
            nread = min(int(row_chunk), nrow - startrow)
            times = _as_float_vector(tool.getcol('TIME', startrow, nread), 'TIME')
            durations = _as_float_vector(
                tool.getcol('INTERVAL', startrow, nread), 'INTERVAL')
            if len(times) != nread or len(durations) != nread:
                raise ValueError("MS TIME and INTERVAL have inconsistent lengths")
            if np.any(durations < 0):
                raise ValueError("MS INTERVAL contains a negative value")
            left = float(np.min(times - durations / 2.0))
            right = float(np.max(times + durations / 2.0))
            earliest = left if earliest is None else min(earliest, left)
            latest = right if latest is None else max(latest, right)
    finally:
        tool.close()
    return earliest, latest
_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)
_POLICY_START_SECONDS = _ANT3_POLICY_START_MJD * 86400.0
_POLICY_END_SECONDS = _ANT3_POLICY_END_MJD * 86400.0


def _utc_seconds(value):
    """Convert one timezone-aware UTC ISO-8601 string to MJD seconds."""
    if not isinstance(value, str):
        raise ValueError('historical interval timestamps must be UTC strings')
    text = value.strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        # Python 3.8 fromisoformat rejects single-digit fractional seconds.
        fmt = '%Y-%m-%dT%H:%M:%S' + ('.%f' if '.' in text else '') + '%z'
        timestamp = datetime.strptime(text, fmt)
    except ValueError as exc:
        raise ValueError('invalid historical UTC timestamp: {}'.format(value)) from exc
    seconds = (timestamp.astimezone(timezone.utc) - _MJD_EPOCH).total_seconds()
    if not np.isfinite(seconds):
        raise ValueError('historical interval timestamp is not finite: {}'.format(value))
    return float(seconds)


def _historical_intervals():
    """Convert and validate the static reviewed UTC interval table."""
    converted = []
    for index, pair in enumerate(HISTORICAL_NO_ANT3_INTERVALS_UTC):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('historical interval {} must be a UTC pair'.format(index))
        start_utc, end_utc = pair
        start = _utc_seconds(start_utc)
        end = _utc_seconds(end_utc)
        if end <= start:
            raise ValueError('historical interval {} is empty or reversed'.format(index))
        if start < _POLICY_START_SECONDS or end > _POLICY_END_SECONDS:
            raise ValueError('historical interval {} is outside policy window'.format(index))
        converted.append((start, end, start_utc, end_utc))
    converted.sort(key=lambda item: item[0])
    for previous, current in zip(converted, converted[1:]):
        if current[0] < previous[1]:
            raise ValueError('historical Ant3 intervals overlap')
    return converted


def flag_ant3_stow(msfile, intervals=None, table_tool=None, row_chunk=4096,
                   provenance_path=None):
    """Flag reviewed historical Ant3 intervals on a pipeline-owned MS copy.

    Automatic selection uses only MS time coverage and the static UTC table;
    it never inspects filenames or queries SQL.  Explicit numeric intervals
    remain available for controlled tests and reruns.

    :param msfile: Pipeline-owned Measurement Set copy.
    :type msfile: str
    :param intervals: Explicit ``(start, end)`` intervals in MS TIME units.
    :type intervals: iterable, optional
    :param table_tool: Optional CASA table tool or test double.
    :type table_tool: object, optional
    :param row_chunk: Number of rows per CASA table operation.
    :type row_chunk: int
    :param provenance_path: Optional JSON sidecar path.
    :type provenance_path: str, optional
    :returns: Flagging, skip, or incomplete report.
    :rtype: dict
    :raises RuntimeError: If an affected automatic run has no static table.
    """
    if intervals is not None:
        report = flag_antenna_intervals(
            msfile, intervals, physical_antenna=3, table_tool=table_tool,
            row_chunk=row_chunk, provenance_path=None)
        report['interval_source'] = 'explicit'
        report['policy_id'] = None
        _write_provenance(provenance_path, report)
        return report

    start, end = _measurement_set_time_bounds(
        msfile, table_tool=table_tool, row_chunk=row_chunk)
    report = {
        'schema_version': 1,
        'operation': 'skip',
        'source_ms': os.path.abspath(os.fspath(msfile)),
        'physical_antenna': 3,
        'interval_source': 'verified_historical_no_ant3',
        'policy_id': POLICY_ID,
        'intervals': [],
        'matched_intervals': [],
        'matched_intervals_utc': [],
        'rows_scanned': 0,
        'rows_selected': 0,
        'rows_changed': 0,
        'cells_newly_flagged': 0,
        'ms_start': float(start),
        'ms_end': float(end),
    }
    if end <= _POLICY_START_SECONDS or start >= _POLICY_END_SECONDS:
        report['skip_reason'] = 'outside_historical_no_ant3_intervals'
        _write_provenance(provenance_path, report)
        return report

    converted = _historical_intervals()
    if not converted:
        report['operation'] = 'incomplete'
        report['skip_reason'] = 'historical_no_ant3_interval_manifest_empty'
        report['manifest_complete'] = False
        _write_provenance(provenance_path, report)
        raise RuntimeError('{} has no populated historical Ant3 interval manifest'.format(POLICY_ID))

    matched = [item for item in converted if item[1] > start and item[0] < end]
    if not matched:
        report['skip_reason'] = 'outside_historical_no_ant3_intervals'
        report['manifest_complete'] = True
        _write_provenance(provenance_path, report)
        return report

    numeric = [(item[0], item[1]) for item in matched]
    report = flag_antenna_intervals(
        msfile, numeric, physical_antenna=3, table_tool=table_tool,
        row_chunk=row_chunk, provenance_path=None)
    report.update({
        'interval_source': 'verified_historical_no_ant3',
        'policy_id': POLICY_ID,
        'manifest_complete': True,
        'ms_start': float(start),
        'ms_end': float(end),
        'matched_intervals': [list(item) for item in numeric],
        'matched_intervals_utc': [[item[2], item[3]] for item in matched],
    })
    _write_provenance(provenance_path, report)
    return report
