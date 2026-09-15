"""Helpers for working with sparse CASA calibration-table row layouts."""

import numpy as np


_SCALAR_TEMPLATE_COLUMNS = (
    "TIME",
    "FIELD_ID",
    "ANTENNA2",
    "INTERVAL",
    "SCAN_NUMBER",
    "OBSERVATION_ID",
)


def append_missing_caltable_spw_rows(table_tool, expected_spw_ids, nant):
    """Append complete antenna row blocks for spectral windows CASA omitted.

    ``bandpass`` may omit an otherwise valid spectral window when its template
    solve does not converge.  Residual BPS values come from Caleovsa/SQL, so
    callers need a structurally complete table in which to store those values.
    New rows inherit only time/selection metadata from an existing row; the
    caller remains responsible for writing CPARAM, FLAG, SNR, and PARAMERR.

    :param table_tool: Open, writable CASA table tool.
    :param expected_spw_ids: Spectral-window identifiers required by the MS.
    :type expected_spw_ids: iterable[int]
    :param nant: Number of antenna rows required for each spectral window.
    :type nant: int
    :returns: Spectral-window identifiers that were appended.
    :rtype: list[int]
    :raises ValueError: If the solver produced no template rows at all.
    """

    expected = [int(spw_id) for spw_id in expected_spw_ids]
    nant = int(nant)
    if nant <= 0:
        raise ValueError("nant must be positive")

    existing_spw_ids = np.asarray(
        table_tool.getcol("SPECTRAL_WINDOW_ID")
    ).reshape(-1)
    if existing_spw_ids.size == 0:
        raise ValueError(
            "cannot repair an empty calibration table without a template row"
        )

    missing = [spw_id for spw_id in expected if spw_id not in existing_spw_ids]
    if not missing:
        return []

    template_values = {
        column: table_tool.getcell(column, 0)
        for column in _SCALAR_TEMPLATE_COLUMNS
    }
    for spw_id in missing:
        startrow = int(table_tool.nrows())
        table_tool.addrows(nant)
        for column, value in template_values.items():
            table_tool.putcol(
                column,
                np.full(nant, value),
                startrow,
                nant,
            )
        table_tool.putcol(
            "SPECTRAL_WINDOW_ID",
            np.full(nant, spw_id, dtype=np.int64),
            startrow,
            nant,
        )
        table_tool.putcol(
            "ANTENNA1",
            np.arange(nant, dtype=np.int64),
            startrow,
            nant,
        )
    return missing


def resolve_caltable_spw_rows(
        spectral_window_ids, antenna_ids, spectral_window_id, nant):
    """Resolve one spectral window's contiguous calibration-table rows.

    CASA calibration solvers can omit an entire spectral window when no valid
    solution is available.  Callers must therefore use the table's
    ``SPECTRAL_WINDOW_ID`` and ``ANTENNA1`` columns instead of calculating row
    offsets as ``spectral_window_id * nant``.

    :param spectral_window_ids: Spectral-window identifier for every table row.
    :type spectral_window_ids: array-like
    :param antenna_ids: Antenna identifier for every table row.
    :type antenna_ids: array-like
    :param spectral_window_id: Spectral window to resolve.
    :type spectral_window_id: int
    :param nant: Number of antennas in the target measurement set.
    :type nant: int
    :returns: ``(startrow, antenna_indices)`` or ``None`` when CASA omitted the
        requested spectral window.
    :rtype: tuple[int, numpy.ndarray] or None
    :raises ValueError: If the table columns are inconsistent or the selected
        rows cannot be updated safely as one contiguous block.
    """

    spw_ids = np.asarray(spectral_window_ids).reshape(-1)
    ant_ids = np.asarray(antenna_ids).reshape(-1)
    if spw_ids.size != ant_ids.size:
        raise ValueError(
            "SPECTRAL_WINDOW_ID and ANTENNA1 must have the same length"
        )

    rows = np.flatnonzero(spw_ids == int(spectral_window_id))
    if rows.size == 0:
        return None
    if rows.size > 1 and not np.all(np.diff(rows) == 1):
        raise ValueError(
            "calibration-table rows for SPW {0} are not contiguous".format(
                int(spectral_window_id),
            )
        )

    row_antennas = np.asarray(ant_ids[rows], dtype=np.int64)
    if np.any(row_antennas < 0) or np.any(row_antennas >= int(nant)):
        raise ValueError(
            "calibration-table antenna index is outside target range"
        )
    if np.unique(row_antennas).size != row_antennas.size:
        raise ValueError(
            "calibration-table SPW contains duplicate antenna rows"
        )
    return int(rows[0]), row_antennas
