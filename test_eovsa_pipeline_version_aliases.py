import importlib
import os
import sys
import tempfile
import types
from datetime import datetime

from astropy.time import Time
import pytest


_TEST_ROOT = os.path.join(
    tempfile.gettempdir(),
    'suncasa-eovsa-pipeline-version-tests',
)
for _name in (
    'EOVSAUDBMS',
    'EOVSAUDBMSSCL',
    'EOVSAUDBMSSLFCALED',
    'EOVSAUDB',
    'EOVSACAL',
    'EOVSASLFCAL',
    'EOVSAQLOOKFITS',
    'EOVSAQLOOKFIG',
    'EOVSASYNOPTICFIG',
    'EOVSA_WORKDIR',
):
    os.environ.setdefault(_name, os.path.join(_TEST_ROOT, _name.lower()))


sys.modules.setdefault(
    'suncasa.eovsa.eovsa_diskmodel',
    types.ModuleType('suncasa.eovsa.eovsa_diskmodel'),
)


eovsa_pipeline = importlib.import_module('suncasa.eovsa.eovsa_pipeline')


def test_public_version_names_and_storage_namespaces():
    assert eovsa_pipeline.PUBLIC_PIPELINE_VERSIONS == ('v1.0', 'v2.0')
    assert eovsa_pipeline.normalize_pipeline_version('v2.0') == 'v2.0'
    assert eovsa_pipeline.normalize_pipeline_version('v2.1') == 'v2.1'
    assert eovsa_pipeline.normalize_pipeline_version('v2.1_alt') == 'v2.1_alt'
    assert eovsa_pipeline.get_pipeline_storage_version('v2.0') == 'v2.0'
    assert eovsa_pipeline.get_pipeline_storage_version('v2.0_alt') == 'v2.0_alt'
    assert eovsa_pipeline.get_pipeline_storage_version('v2.1') == 'v2.1'
    assert eovsa_pipeline.get_pipeline_storage_version('legacy_v2.0') == 'legacy_v2.0'
    for removed in ('v3.0', 'v3.0_alt', 'v3.1', 'v3.1_alt'):
        with pytest.raises(ValueError):
            eovsa_pipeline.normalize_pipeline_version(removed)


def test_direct_output_helper_writes_canonical_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(eovsa_pipeline, 'qlookfitsdir', str(tmp_path))

    output_dir = eovsa_pipeline.get_synoptic_product_output_dir(
        Time('2026-09-09T20:00:00'), 'v2.0'
    )

    assert output_dir == str(tmp_path / '2026' / '09' / '09' / 'v2.0')


def test_v2_dispatches_to_wsclean_and_keeps_current_ms_namespace(
        monkeypatch, tmp_path):
    workdir = tmp_path / 'work'
    workdir.mkdir()
    input_ms = workdir / 'UDB20260909.ms'
    input_ms.mkdir()
    monkeypatch.setattr(eovsa_pipeline.os, 'chdir', lambda _path: None)
    monkeypatch.setattr(eovsa_pipeline, 'udbmsslfcaleddir', str(tmp_path / 'selfcal'))
    monkeypatch.setattr(eovsa_pipeline, 'slfcaltbdir', str(tmp_path / 'selfcal_tables'))
    monkeypatch.setattr(eovsa_pipeline, 'synopticfigdir', str(tmp_path / 'figures'))
    monkeypatch.setattr(
        eovsa_pipeline,
        'get_synoptic_product_output_dir',
        lambda *_args, **_kwargs: str(tmp_path / 'images'),
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        'trange2ms',
        lambda *_args, **_kwargs: pytest.fail('explicit input must bypass discovery/import'),
    )
    monkeypatch.setattr(
        eovsa_pipeline,
        'calibeovsa',
        lambda *_args, **_kwargs: pytest.fail('explicit input must bypass calibeovsa'),
    )
    esip = importlib.import_module(
        'suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean'
    )
    captured = {}

    def fake_pipeline_run(vis, **kwargs):
        captured['vis'] = vis
        captured.update(kwargs)
        return {'imaged': True}

    monkeypatch.setattr(esip, 'pipeline_run', fake_pipeline_run)

    result = eovsa_pipeline.calib_pipeline(
        Time('2026-09-09 20:00:00'),
        workdir=str(workdir),
        version='v2.0',
        doimport=False,
        input_ms=str(input_ms),
    )

    assert result == {'imaged': True}
    assert captured['vis'] == str(input_ms)
    assert captured['outputvis'].endswith('UDB20260909.v2.0.ms')


def test_v2_daily_filename_uses_current_wsclean_product_contract():
    qlook = importlib.import_module('suncasa.eovsa.eovsa_pltQlookImage')
    assert qlook.synoptic_daily_product_filename(
        datetime(2026, 9, 9), '00-01', version='v2.0'
    ) == 'eovsa.synoptic_daily.20260909T200000Z.s00-01.tb.disk.fits'
