import numpy as np
import pytest

from suncasa.utils.caltable_layout import (
    append_missing_caltable_spw_rows,
    resolve_caltable_spw_rows,
)


class FakeTable:
    def __init__(self, columns):
        self.columns = {
            name: list(values)
            for name, values in columns.items()
        }

    def getcol(self, name):
        return np.asarray(self.columns[name])

    def getcell(self, name, row):
        return self.columns[name][row]

    def nrows(self):
        return len(self.columns["SPECTRAL_WINDOW_ID"])

    def addrows(self, count):
        for values in self.columns.values():
            values.extend([None] * count)

    def putcol(self, name, values, startrow, nrow):
        self.columns[name][startrow:startrow + nrow] = list(values)


def _sparse_fake_table():
    scalar_defaults = {
        "TIME": 123.0,
        "FIELD_ID": 0,
        "ANTENNA2": 0,
        "INTERVAL": 0.0,
        "SCAN_NUMBER": 0,
        "OBSERVATION_ID": 0,
    }
    columns = {
        name: [value] * 4
        for name, value in scalar_defaults.items()
    }
    columns.update({
        "SPECTRAL_WINDOW_ID": [0, 0, 2, 2],
        "ANTENNA1": [0, 1, 0, 1],
    })
    return FakeTable(columns)


def test_missing_spw_row_block_is_appended_for_authoritative_values():
    table = _sparse_fake_table()

    appended = append_missing_caltable_spw_rows(table, range(3), nant=2)

    assert appended == [1]
    np.testing.assert_array_equal(
        table.getcol("SPECTRAL_WINDOW_ID"),
        np.array([0, 0, 2, 2, 1, 1]),
    )
    np.testing.assert_array_equal(
        table.getcol("ANTENNA1"),
        np.array([0, 1, 0, 1, 0, 1]),
    )
    startrow, antennas = resolve_caltable_spw_rows(
        table.getcol("SPECTRAL_WINDOW_ID"),
        table.getcol("ANTENNA1"),
        1,
        2,
    )
    assert startrow == 4
    np.testing.assert_array_equal(antennas, np.array([0, 1]))


def test_complete_caltable_is_not_modified():
    table = _sparse_fake_table()

    assert append_missing_caltable_spw_rows(table, (0, 2), nant=2) == []
    assert table.nrows() == 4


def test_empty_caltable_cannot_be_repaired_without_template():
    table = _sparse_fake_table()
    for values in table.columns.values():
        values.clear()

    with pytest.raises(ValueError, match="empty calibration table"):
        append_missing_caltable_spw_rows(table, (0,), nant=2)


def test_missing_spw_does_not_shift_later_table_rows():
    spectral_windows = np.repeat(
        np.concatenate((np.arange(30), np.arange(31, 50))),
        16,
    )
    antennas = np.tile(np.arange(16), 49)

    assert resolve_caltable_spw_rows(
        spectral_windows,
        antennas,
        30,
        16,
    ) is None

    startrow, row_antennas = resolve_caltable_spw_rows(
        spectral_windows,
        antennas,
        31,
        16,
    )

    assert startrow == 30 * 16
    np.testing.assert_array_equal(row_antennas, np.arange(16))


def test_partial_spw_returns_only_antennas_present_in_table():
    spectral_windows = np.array([4, 4, 4, 5, 5])
    antennas = np.array([0, 3, 9, 0, 1])

    startrow, row_antennas = resolve_caltable_spw_rows(
        spectral_windows,
        antennas,
        4,
        16,
    )

    assert startrow == 0
    np.testing.assert_array_equal(row_antennas, np.array([0, 3, 9]))


def test_noncontiguous_spw_rows_are_rejected():
    with pytest.raises(ValueError, match="not contiguous"):
        resolve_caltable_spw_rows(
            np.array([4, 5, 4]),
            np.array([0, 0, 1]),
            4,
            16,
        )


@pytest.mark.parametrize(
    "spectral_windows,antennas,error",
    (
        (np.array([4, 4]), np.array([0]), "same length"),
        (np.array([4]), np.array([16]), "outside"),
        (np.array([4, 4]), np.array([1, 1]), "duplicate"),
    ),
)
def test_invalid_caltable_layout_is_rejected(
        spectral_windows, antennas, error):
    with pytest.raises(ValueError, match=error):
        resolve_caltable_spw_rows(
            spectral_windows,
            antennas,
            4,
            16,
        )
