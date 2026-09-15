import argparse
import json
from datetime import datetime, timedelta
# from astropy.time import Time
import shutil
import traceback
import numpy as np
from astropy.io import fits
from suncasa.suncasatasks import ptclean6 as ptclean
from suncasa.suncasatasks import importeovsa
# Import the private task directly (not the auto-generated CASA wrapper) so the
# cal_npz / refcal_npz_mode / secondary_npz arguments are exposed.
from suncasa.suncasatasks.private.task_calibeovsa import (
    _select_sql_refcal_family,
    calibeovsa,
)

import re
import sys
from eovsapy.dump_tsys import findfiles
from eovsapy.sqlutil import sql2phacalX, sql2refcalX, sql2refcal_bphsbdX
try:
    from eovsapy.sqlutil import sql2refcal_bpsX
except ImportError:
    sql2refcal_bpsX = None
from eovsapy.util import Time
from eovsapy.spw_config import SPWS_52BAND_SELFCAL
import os
from suncasa.eovsa import eovsa_diskmodel as ed
from suncasa.utils import mstools as mstl

from suncasa.casa_compat import import_casatasks, import_casatools
from suncasa.eovsa.update_log import  EOVSA15_UPGRADE_DATE,DCM_IF_FILTER_UPGRADE_DATE

tasks = import_casatasks('split', 'tclean', 'gencal', 'clearcal', 'applycal', 'gaincal',
                         'delmod', 'flagdata')
split = tasks.get('split')
tclean = tasks.get('tclean')
gencal = tasks.get('gencal')
clearcal = tasks.get('clearcal')
applycal = tasks.get('applycal')
gaincal = tasks.get('gaincal')
delmod = tasks.get('delmod')
flagdata = tasks.get('flagdata')

tools = import_casatools(['qatool', 'iatool', 'mstool', 'tbtool'])
qatool = tools['qatool']
iatool = tools['iatool']
mstool = tools['mstool']
tbtool = tools['tbtool']
ms = mstool()
tb = tbtool()


def get_tdate_from_basename(vis):
    # Define the regular expression pattern
    pattern = r'UDB(\d{8})(?:\d+)?(?:\..*)?\.ms(\.tar\.gz)?'

    # Extract the basename from the vis path
    basename = os.path.basename(vis)

    # Search for the pattern in the basename
    match = re.search(pattern, basename)

    if match:
        # Extract the date string
        date_str = match.group(1)

        # Convert the date string to a datetime object
        tdate = datetime.strptime(date_str, '%Y%m%d').replace(hour=20, minute=0, second=0)

        return tdate
    else:
        raise ValueError("The basename does not match the expected format.")


def normalize_reused_ms_input(input_ms, expected_date=None):
    """Validate and normalize an explicitly reused MS or MS archive."""
    if input_ms is None:
        return None

    input_ms = os.path.abspath(os.path.expanduser(os.path.normpath(input_ms)))
    lower_path = input_ms.lower()
    if lower_path.endswith('.ms.tar.gz'):
        valid_type = os.path.isfile(input_ms)
    elif lower_path.endswith('.ms'):
        valid_type = os.path.isdir(input_ms)
    else:
        raise ValueError('input_ms must be an existing .ms directory or .ms.tar.gz archive.')
    if not valid_type:
        raise ValueError(f'input_ms does not exist or has the wrong type: {input_ms}')

    input_date = get_tdate_from_basename(input_ms)
    if expected_date is not None:
        expected_date = Time(expected_date).datetime
        if input_date.date() != expected_date.date():
            raise ValueError(
                'input_ms observing date {0} does not match requested date {1}: {2}'.format(
                    input_date.strftime('%Y-%m-%d'),
                    expected_date.strftime('%Y-%m-%d'),
                    input_ms,
                )
            )
    return input_ms


def validate_reused_ms_options(input_ms, doimport=False, smart_cal_check=False,
                               cal_npz=None, secondary_npz=None, sql_cal_time=None,
                               end_time=None, force_lo_hi_smooth_extrap=False,
                               ndays=1, version='v2.0', imaging_only=False):
    """Reject options that would recalibrate or ambiguously validate a reused MS."""
    if input_ms is None:
        return
    version = normalize_pipeline_version(version)
    if ndays != 1:
        raise ValueError('input_ms supports exactly one requested date; set ndays=1.')
    if version not in WSCLEAN_PIPELINE_VERSIONS:
        raise ValueError(
            'input_ms reuse is supported only by WSClean pipeline versions: {0}'.format(
                ', '.join(WSCLEAN_PIPELINE_VERSIONS)
            )
        )
    if doimport:
        raise ValueError('input_ms and doimport are mutually exclusive; use skip-import.')
    if smart_cal_check:
        raise ValueError(
            'smart_cal_check validates calibration applied in the current run and cannot '
            'verify a reused input_ms; disable smart_cal_check for explicit MS reuse.'
        )

    calibration_options = {
        'cal_npz': cal_npz,
        'secondary_npz': secondary_npz,
        'sql_cal_time': sql_cal_time,
        'end_time': end_time,
        'force_lo_hi_smooth_extrap': force_lo_hi_smooth_extrap,
    }
    conflicts = [name for name, value in calibration_options.items() if value]
    if conflicts:
        raise ValueError(
            'input_ms bypasses import and calibeovsa; remove calibration-only option(s): {0}'.format(
                ', '.join(conflicts)
            )
        )

    basename = os.path.basename(input_ms)
    if basename.lower().endswith('.ms.tar.gz'):
        stem = basename[:-len('.ms.tar.gz')]
    else:
        stem = basename[:-len('.ms')]
    date_prefix = get_tdate_from_basename(input_ms).strftime('UDB%Y%m%d')
    suffix = stem[len(date_prefix):]
    expected_selfcal_prefix = f'.{get_pipeline_storage_version(version)}'
    if imaging_only:
        if not suffix.startswith(expected_selfcal_prefix):
            raise ValueError(
                'imaging_only input_ms must be the versioned selfcal product for {0}; '
                'a calibrated daily MS must run without imaging_only so selfcal is not skipped: {1}'.format(
                    version, input_ms
                )
            )
    elif suffix:
        raise ValueError(
            'non-imaging-only input_ms must be the unversioned calibrated daily MS; '
            'use imaging_only for a versioned selfcal product: {0}'.format(input_ms)
        )


def get_default_cal_tag(version, cal_tag=None):
    if cal_tag:
        return cal_tag
    raise ValueError('cal_tag is required when cal_npz is supplied; use a unique NPZ experiment label.')


def remove_path(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def path_is_within(path, directory):
    """Return whether ``path`` is inside ``directory`` after normalization."""
    if path is None:
        return False
    path = os.path.abspath(path)
    directory = os.path.abspath(directory)
    return os.path.commonpath((path, directory)) == directory


def clear_directory_preserving_path(directory, preserve_path):
    """Clear directory contents except the child containing ``preserve_path``."""
    directory = os.path.abspath(directory)
    preserve_path = os.path.abspath(preserve_path)
    if not path_is_within(preserve_path, directory):
        raise ValueError(f'preserve_path is outside directory: {preserve_path}')
    for name in os.listdir(directory):
        child = os.path.join(directory, name)
        if path_is_within(preserve_path, child):
            continue
        remove_path(child)


def stage_calibration_inputs(invis, workdir, stage_tag=None, preserve_imports=False):
    """Place mutable calibration inputs on the processing filesystem.

    :param preserve_imports: Copy even inputs already on processing storage,
        keeping reusable imports unchanged when applying observing-state flags.
    :type preserve_imports: bool
    """
    stage_name = 'calibration_imported_ms'
    if stage_tag:
        stage_name += '_' + str(stage_tag)
    stage_dir = os.path.join(workdir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)
    staged = []
    for src in invis:
        src = os.path.normpath(src)
        archived = src.endswith('.ms.tar.gz')
        if path_is_within(src, workdir) and not preserve_imports and not archived:
            print(f'Using UDB MS already staged on processing storage: {src}')
            staged.append(src)
            continue
        basename = os.path.basename(src.rstrip('/'))
        dst = os.path.join(stage_dir, basename[:-7] if archived else basename)
        if os.path.realpath(src) == os.path.realpath(dst):
            raise ValueError('Calibration staging destination is the preserved input: ' + src)
        if os.path.exists(dst):
            remove_path(dst)
        print(f'Staging imported UDB MS for calibration: {src} -> {dst}')
        if archived:
            shutil.unpack_archive(src, stage_dir, 'gztar')
            if not os.path.isdir(dst):
                raise ValueError('Scan archive does not contain expected MS: ' + dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def stage_cal_npz_inputs(invis, workdir, cal_tag):
    """Backward-compatible NPZ wrapper for calibration input staging."""
    return stage_calibration_inputs(invis, workdir, 'npz_' + (cal_tag or 'untagged'))


def stage_time_cutoff_inputs(invis, workdir, end_time):
    """Stage raw scan MS inputs and flag samples at or after ``end_time``."""
    cutoff = Time(end_time)
    cutoff_start = cutoff.datetime.strftime('%Y/%m/%d/%H:%M:%S')
    cutoff_end = (cutoff.datetime + timedelta(days=1)).strftime('%Y/%m/%d/%H:%M:%S')
    timerange = f'{cutoff_start}~{cutoff_end}'
    stage_dir = os.path.join(workdir, 'time_cutoff_imported_ms_' + cutoff.datetime.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(stage_dir, exist_ok=True)
    staged = []
    for src in invis:
        src = os.path.normpath(src)
        dst = os.path.join(stage_dir, os.path.basename(src.rstrip('/')))
        if os.path.exists(dst):
            remove_path(dst)
        print(f'Staging imported UDB MS for end-time cutoff: {src} -> {dst}')
        if os.path.isdir(src):
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
        clearcal(vis=dst, addmodel=False)
        print(f'Flagging staged MS data at or after {cutoff.iso}: {dst}')
        try:
            flagdata(vis=dst, mode='manual', timerange=timerange, flagbackup=False)
        except RuntimeError as exc:
            if 'MSSelectionNullSelection' not in str(exc):
                raise
            print(f'No staged MS samples at or after {cutoff.iso}: {dst}')
        staged.append(dst)
    return staged


import socket
import os

hostname = socket.gethostname()
is_on_server = hostname in ['pipeline', 'inti.hpcnet.campus.njit.edu']
is_on_inti = hostname == 'inti.hpcnet.campus.njit.edu'
if is_on_server:
    base_dir = "/inti/data/pipeline_mirror" if is_on_inti else ""
else:
    base_dir = './'


class Path_config:
    """Resolve EOVSA pipeline paths without modifying the filesystem.

    Directories are created by the processing operation that writes to them,
    not while this module is imported.

    :param base_dir: Prefix used for paths without an environment override.
    :type base_dir: str
    """

    def __init__(self, base_dir=base_dir):
        self.paths = {}

        self.base_dir = base_dir
        # Setting paths
        self.udbmsdir = self._get_env_var('EOVSAUDBMS', f'{base_dir}/data1/eovsa/fits/UDBms/')
        self.udbmsscldir = self._get_env_var('EOVSAUDBMSSCL', f'{base_dir}/data1/eovsa/fits/UDBms_scl/')
        self.udbmsslfcaleddir = self._get_env_var('EOVSAUDBMSSLFCALED', f'{base_dir}/data1/eovsa/fits/UDBms_slfcaled/')
        self.udbdir = self._get_env_var('EOVSAUDB', f'{base_dir}/data1/eovsa/fits/UDB/')
        self.caltbdir = self._get_env_var('EOVSACAL', f'{base_dir}/data1/eovsa/caltable/')
        self.slfcaltbdir = self._get_env_var('EOVSASLFCAL', f'{base_dir}/data1/eovsa/slfcaltable/')
        self.qlookfitsdir = self._get_env_var('EOVSAQLOOKFITS', f'{base_dir}/data1/eovsa/fits/synoptic/')
        self.qlookfigdir = self._get_env_var('EOVSAQLOOKFIG', f'{base_dir}/common/webplots/qlookimg_10m/')
        self.synopticfigdir = self._get_env_var('EOVSASYNOPTICFIG', f'{base_dir}/common/webplots/SynopticImg/')
        self.workdir_default = self._get_env_var('EOVSA_WORKDIR', f'{base_dir}/data1/workdir/')

        # Print a summary of paths
        self._print_summary()

    def _get_env_var(self, env_var, default_path):
        """Return an environment path override or its configured default.

        :param env_var: Environment variable containing an optional path.
        :type env_var: str
        :param default_path: Path returned when the variable is unset.
        :type default_path: str
        :returns: Resolved configured path without creating it.
        :rtype: str
        """
        path = os.getenv(env_var) or default_path
        self.paths[env_var] = path
        return path

    def _print_summary(self):
        print("Paths Configuration Summary:")
        for env_var, path in self.paths.items():
            print(f"  {env_var}: {path}")


# Usage
pathconfig = Path_config()
print(pathconfig.udbmsdir)  # Accessing the directory path

udbmsdir = pathconfig.udbmsdir
udbmsscldir = pathconfig.udbmsscldir
udbmsslfcaleddir = pathconfig.udbmsslfcaleddir
udbdir = pathconfig.udbdir
caltbdir = pathconfig.caltbdir
slfcaltbdir = pathconfig.slfcaltbdir
qlookfitsdir = pathconfig.qlookfitsdir
qlookfigdir = pathconfig.qlookfigdir
synopticfigdir = pathconfig.synopticfigdir
workdir_default = pathconfig.workdir_default

LEGACY_PIPELINE_VERSION = 'legacy_v2.0'
SUPPORTED_PIPELINE_VERSIONS = (
    'v1.0',
    'v2.0',
    LEGACY_PIPELINE_VERSION,
    'v2.0_alt',
    # Internal WSClean subversions. They retain the v2 naming family.
    'v2.1',
    'v2.1_alt',
)
PUBLIC_PIPELINE_VERSIONS = ('v1.0', 'v2.0')
WSCLEAN_PIPELINE_VERSIONS = (
    'v2.0',
    'v2.0_alt',
    'v2.1',
    'v2.1_alt',
)
PROVISIONAL_SUCCESS_STATE = 'provisional_success'
FALLBACK_RUNNING_STATE = 'running_with_fallback_calibration'
FALLBACK_PARTIAL_STATE = 'partial_with_fallback_calibration'
IMAGING_REVIEW_REQUIRED_STATE = 'imaging_review_required'
S00_PASS_QA_STATES = frozenset(('PASS_BASE_MASK', 'PASS_EXPANDED_MASK'))
PROVISIONAL_CALIBRATION_MODE = 'FALLBACK'
SAME_DAY_CALIBRATION_MODE = 'SAME_DAY'


def normalize_pipeline_version(version):
    """Validate and return the canonical selector used for new outputs."""
    if version not in SUPPORTED_PIPELINE_VERSIONS:
        raise ValueError(
            'Version {0} is not supported. Valid versions are {1}.'.format(
                version, ', '.join(SUPPORTED_PIPELINE_VERSIONS)
            )
        )
    return version


def get_pipeline_storage_version(version):
    """Return the internal self-cal MS suffix for a canonical selector.

    Canonical WSClean products use their v2-family selector. The former
    algorithm is kept separate under ``legacy_v2.0``.
    """
    return normalize_pipeline_version(version)


def get_synoptic_day_output_dir(tim):
    tim = Time(tim)
    return os.path.join(qlookfitsdir, tim.datetime.strftime("%Y/%m/%d"))


def get_synoptic_product_output_dir(tim, version):
    version = normalize_pipeline_version(version)
    return os.path.join(get_synoptic_day_output_dir(tim), version)


def get_local_day_bounds(tim):
    """Return the local-day bounds used by the daily EOVSA pipeline."""
    tim = Time(tim)
    if tim.mjd == np.fix(tim.mjd):
        tim = Time(tim.mjd + 0.5, format='mjd')
    dhr = tim.LocalTime.utcoffset().total_seconds() / 60 / 60 / 24
    btime = Time(np.fix(tim.mjd + dhr) - dhr, format='mjd')
    etime = Time(btime.mjd + 1, format='mjd')
    return btime, etime


def get_synoptic_output_info(tim, version='v2.0', fits_tag=''):
    """Return the expected synoptic daily FITS products and status file for one day.

    When ``fits_tag`` is a non-empty string (e.g. ``'test'`` for calwidget_v2
    NPZ benchmark runs), it is spliced into the FITS and status filenames as
    an infix so the alternate products sit alongside the production ones
    without collision.
    """
    from suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean import (
        FrequencySetup,
        format_spw,
    )

    version = normalize_pipeline_version(version)
    tim = Time(tim)
    date_str = tim.datetime.strftime('%Y%m%d')
    day_outdir = get_synoptic_day_output_dir(tim)
    imgoutdir = get_synoptic_product_output_dir(tim, version)
    tag = f'.{fits_tag}' if fits_tag else ''
    freq_setup = FrequencySetup(tim)
    fitsfiles = []
    for spw in freq_setup.spws:
        spwstr = format_spw(spw)
        fitsfiles.append(os.path.join(
            imgoutdir,
            f'eovsa.synoptic_daily{tag}.{date_str}T200000Z.s{spwstr}.tb.disk.fits'))
    statusfile = os.path.join(
        day_outdir,
        f'eovsa.synoptic_pipeline_status.{date_str}.{version}{tag}.json')
    return {
        'date_str': date_str,
        'version': version,
        'day_outdir': day_outdir,
        'imgoutdir': imgoutdir,
        'fitsfiles': fitsfiles,
        'statusfile': statusfile,
    }


def summarize_synoptic_outputs(tim, version='v2.0', fits_tag=''):
    """Summarize synoptic daily FITS availability for one pipeline day."""
    info = get_synoptic_output_info(tim, version=version, fits_tag=fits_tag)
    existing = [f for f in info['fitsfiles'] if os.path.exists(f)]
    return {
        **info,
        'existing_fitsfiles': existing,
        'fits_complete': len(existing) == len(info['fitsfiles']),
        'fits_count': len(existing),
        'fits_expected_count': len(info['fitsfiles']),
    }


def summarize_s00_imaging_qa(fitsfiles, require_qa=False):
    """Read the deterministic imaging QA state from the daily s00 FITS.

    Historical products without ``QASTATE`` remain eligible for the legacy
    status path unless ``require_qa`` marks the product as freshly generated.
    ``REVIEW_*`` states are advisory and remain visible in the status JSON
    without blocking an otherwise complete run. ``FAIL_*``, an unrecognized
    state, or an unreadable s00 FITS records a manual-review requirement while
    leaving product publication enabled.

    :param fitsfiles: Expected or existing daily synoptic FITS paths.
    :type fitsfiles: list[str]
    :param require_qa: Require a freshly generated s00 product to carry
        ``QASTATE``.
    :type require_qa: bool
    :returns: Compact s00 QA fields for the per-day status JSON.
    :rtype: dict
    """
    summary = {
        's00_qa_state': None,
        's00_qa_warning': False,
        's00_qa_requires_review': False,
        's00_qa_fitsfile': None,
        's00_qa_reason': 's00_product_not_present',
    }
    s00_fitsfiles = [
        path for path in fitsfiles
        if re.search(r'\.s00-01\.', os.path.basename(path))
    ]
    if not s00_fitsfiles:
        return summary

    s00_fitsfile = s00_fitsfiles[0]
    summary['s00_qa_fitsfile'] = s00_fitsfile
    try:
        with fits.open(s00_fitsfile, memmap=False) as hdul:
            image_header = next(
                (hdu.header for hdu in hdul if 'CDELT1' in hdu.header),
                None,
            )
            if image_header is None:
                raise ValueError('missing image HDU')
            qa_state = str(image_header.get('QASTATE', '')).strip().upper()
    except Exception as exc:
        summary.update({
            's00_qa_requires_review': True,
            's00_qa_reason': 's00_fits_read_failed: {0}'.format(exc),
        })
        return summary

    if not qa_state:
        summary['s00_qa_requires_review'] = bool(require_qa)
        summary['s00_qa_reason'] = 'qastate_not_present'
        return summary

    summary['s00_qa_state'] = qa_state
    if qa_state in S00_PASS_QA_STATES:
        summary['s00_qa_reason'] = 'pass'
    elif qa_state.startswith('REVIEW_'):
        summary['s00_qa_warning'] = True
        summary['s00_qa_reason'] = qa_state.lower()
    elif qa_state.startswith('FAIL_'):
        summary['s00_qa_requires_review'] = True
        summary['s00_qa_reason'] = qa_state.lower()
    else:
        summary['s00_qa_requires_review'] = True
        summary['s00_qa_reason'] = 'unknown_qastate'
    return summary


def apply_s00_qa_to_pipeline_state(base_state, fitsfiles, require_qa=False):
    """Attach s00 QA metadata without changing the outer pipeline state.

    Imaging QA is advisory for product publication.  ``REVIEW_*``, ``FAIL_*``,
    missing, unknown, and unreadable QA states remain visible in the status
    JSON, but they must not suppress compression or browser-preview creation.

    :param base_state: Pipeline state established by product completeness and
        calibration provenance.
    :type base_state: str
    :param fitsfiles: Existing daily synoptic FITS paths.
    :type fitsfiles: list[str]
    :param require_qa: Require the s00 FITS to carry ``QASTATE``.
    :type require_qa: bool
    :returns: Unchanged outer state and compact s00 QA status fields.
    :rtype: tuple[str, dict]
    """
    imaging_qa = summarize_s00_imaging_qa(fitsfiles, require_qa=require_qa)
    if imaging_qa['s00_qa_requires_review']:
        imaging_qa['imaging_base_state'] = base_state
    return base_state, imaging_qa


def set_synoptic_calibration_warning(fitsfiles, calibration_date=None):
    """Stamp or clear provisional calibration warning metadata in FITS files.

    :param fitsfiles: FITS files whose image extension headers should be updated.
    :type fitsfiles: list[str]
    :param calibration_date: Calibration date used for provisional products. If
        ``None``, any previous provisional warning state is cleared.
    :type calibration_date: str or None
    :returns: Number of files whose image extension header was updated.
    :rtype: int
    """
    updated = 0
    for fitsfile in fitsfiles:
        if not os.path.exists(fitsfile):
            continue
        try:
            with fits.open(fitsfile, mode='update') as hdul:
                for hdu in hdul:
                    if 'CDELT1' not in hdu.header:
                        continue
                    if calibration_date:
                        hdu.header.set('CALMODE', PROVISIONAL_CALIBRATION_MODE,
                                       'calibration provenance for preview warning')
                        hdu.header.set('CALDATE', str(calibration_date),
                                       'date of SQL calibration used for imaging')
                        hdu.header.set('CALWARN', True,
                                       'preview should warn about provisional calibration')
                    else:
                        hdu.header.set('CALMODE', SAME_DAY_CALIBRATION_MODE,
                                       'calibration provenance for preview warning')
                        hdu.header.set('CALWARN', False,
                                       'preview should warn about provisional calibration')
                        if 'CALDATE' in hdu.header:
                            del hdu.header['CALDATE']
                    updated += 1
                    break
        except Exception as exc:
            print(f'WARNING: Failed to update calibration warning metadata for {fitsfile}: {exc}')
    return updated


def read_pipeline_status(statusfile):
    """Read the per-day pipeline status JSON if it exists."""
    if not os.path.exists(statusfile):
        return {}
    try:
        with open(statusfile, 'r') as infile:
            return json.load(infile)
    except Exception as exc:
        print(f'WARNING: Failed to read status file {statusfile}: {exc}')
        return {}


def write_pipeline_status(statusfile, state, **extra):
    """Persist a compact per-day pipeline status JSON."""
    os.makedirs(os.path.dirname(statusfile), exist_ok=True)
    payload = {
        'state': state,
        'updated_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    payload.update(extra)
    with open(statusfile, 'w') as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)
        outfile.write('\n')
    return payload


def get_calibration_deadline_utc(tim, hour=4, day_offset=2):
    """Return the UTC hard deadline for one observing day label."""
    tim = Time(tim)
    return datetime(
        tim.datetime.year,
        tim.datetime.month,
        tim.datetime.day,
        hour,
        0,
        0,
    ) + timedelta(days=day_offset)


def get_calibration_readiness(tim, refcal_sql_mode='auto'):
    """Check whether the selected daily SQL calibration route is ready.

    ``bph_sbd`` requires a complete digest-matched type-14+16 family.
    ``auto`` uses that same complete family when type 16 exists, permits a
    historical type-14-only day only when no type-16 records exist, and falls
    back to type 8 only when neither companion type is present. Explicit
    ``legacy`` preserves the type-8 readiness behavior.

    :param tim: Observing-day time.
    :type tim: astropy.time.Time
    :param refcal_sql_mode: SQL reference-calibration route.
    :type refcal_sql_mode: str
    :returns: Readiness and provenance fields.
    :rtype: dict
    :raises ValueError: If ``refcal_sql_mode`` is unsupported.
    """

    tim = Time(tim)
    refcal_sql_mode = (refcal_sql_mode or 'auto').strip().lower()
    if refcal_sql_mode not in ('auto', 'bph_sbd', 'smb', 'legacy'):
        raise ValueError(
            'unsupported refcal_sql_mode {0!r}'.format(refcal_sql_mode)
        )
    btime, etime = get_local_day_bounds(tim)
    deadline_utc = get_calibration_deadline_utc(tim)
    now_utc = datetime.utcnow()
    readiness = {
        'ready': False,
        'reason': '',
        'day_start_utc': btime.iso,
        'day_end_utc': etime.iso,
        'hard_deadline_utc': deadline_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'deadline_expired': now_utc >= deadline_utc,
        'refcal_timestamp_utc': None,
        'phacal_count': 0,
        'latest_phacal_timestamp_utc': None,
        'phacal_warning': None,
        'refcal_family_status': None,
        'refcal_family_reason': None,
    }

    try:
        # Resolve the reference calibration the same way calibeovsa does at imaging
        # time. SQL refcal records are locatored at a ~07 UT lookup timestamp that
        # PRECEDES the day's observations, so querying at the day start (btime)
        # resolves the PREVIOUS day's record. Query at the day end so the same-day
        # refcal is selected.
        refcal = sql2refcalX(etime)
    except Exception as exc:
        readiness['reason'] = f'refcal_query_failed: {exc}'
        return readiness

    if not refcal:
        readiness['reason'] = 'missing_refcal'
        return readiness

    ref_ts = refcal['timestamp']

    # phacal['t_ref'] is the real refcal OBSERVATION time the phacal was solved
    # against, not the ~07 UT SQL record locator in refcal['timestamp'].
    ref_obs = refcal.get('t_bg') or ref_ts
    if refcal_sql_mode in ('auto', 'bph_sbd'):
        try:
            bphsbd_records = sql2refcal_bphsbdX(
                [btime, etime],
                nrecords=0,
                neat=False,
                verbose=False,
            )
        except Exception as exc:
            readiness['reason'] = 'refcal_family_query_failed'
            readiness['refcal_family_reason'] = (
                'type-14 query failed: {0}'.format(exc)
            )
            return readiness
        if sql2refcal_bpsX is None:
            readiness['reason'] = 'refcal_family_query_failed'
            readiness['refcal_family_reason'] = (
                'caltype-16 SQL reader is unavailable'
            )
            return readiness
        try:
            bps_records = sql2refcal_bpsX(
                [btime, etime],
                nrecords=0,
                neat=False,
                verbose=False,
            )
        except Exception as exc:
            readiness['reason'] = 'refcal_family_query_failed'
            readiness['refcal_family_reason'] = (
                'type-16 query failed: {0}'.format(exc)
            )
            return readiness
        family_lookup_time = Time(
            0.5 * (btime.mjd + etime.mjd),
            format='mjd',
        )
        family = _select_sql_refcal_family(
            bphsbd_records,
            bps_records,
            family_lookup_time,
        )
        readiness['refcal_family_status'] = family['status']
        readiness['refcal_family_reason'] = family.get('reason')
        if family['status'] == 'complete':
            ref_obs = family['bphsbd_record'].get('t_refcal') or ref_obs
        elif (
            refcal_sql_mode == 'auto'
            and family['status'] == 'historical_type14'
        ):
            ref_obs = family['bphsbd_record'].get('t_refcal') or ref_obs
        elif (
            refcal_sql_mode == 'auto'
            and family['status'] == 'missing'
        ):
            # No companion record exists, so auto retains the legacy type-8
            # route. A partial family is never treated this way.
            pass
        else:
            readiness['reason'] = (
                'incomplete_refcal_family'
                if family['status'] == 'partial'
                else 'missing_complete_refcal_family'
            )
            return readiness
    else:
        readiness['refcal_family_status'] = 'legacy_type8'
    readiness['refcal_timestamp_utc'] = ref_obs.iso
    if not (btime.mjd <= ref_obs.mjd < etime.mjd):
        readiness['reason'] = 'stale_refcal'
        return readiness

    try:
        phacals = sql2phacalX([btime, etime], nrecords=0, neat=True, verbose=False) or []
    except Exception as exc:
        readiness['ready'] = True
        readiness['reason'] = 'ready_phacal_query_failed'
        readiness['phacal_warning'] = f'phacal_query_failed: {exc}'
        return readiness

    valid_phacals = []
    for phacal in phacals:
        if abs(phacal['t_ref'].jd - ref_obs.jd) <= 30. / 1440.:
            valid_phacals.append(phacal)

    readiness['phacal_count'] = len(valid_phacals)
    if valid_phacals:
        readiness['latest_phacal_timestamp_utc'] = max(ph['t_pha'] for ph in valid_phacals).iso

    if not valid_phacals:
        readiness['ready'] = True
        readiness['reason'] = 'ready_without_phacal'
        readiness['phacal_warning'] = 'missing_phacal'
        return readiness

    readiness['ready'] = True
    readiness['reason'] = 'ready'
    return readiness


def _calibration_readiness_for_mode(tim, refcal_sql_mode):
    """Call readiness while retaining compatibility with auto-mode hooks."""

    mode = (refcal_sql_mode or 'auto').strip().lower()
    if mode == 'auto':
        return get_calibration_readiness(tim)
    return get_calibration_readiness(tim, refcal_sql_mode=mode)


def get_fallback_calibration_lookback_days():
    """Return how many previous observing days cron may use as provisional calibration."""
    raw_value = os.getenv('EOVSA_PIPELINE_FALLBACK_CAL_LOOKBACK_DAYS', '3')
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 3


def find_previous_ready_calibration(
        tim, max_lookback_days=None, refcal_sql_mode='auto'):
    """Find the newest previous day ready for the selected SQL route.

    :param tim: Target observing-day time.
    :type tim: astropy.time.Time
    :param max_lookback_days: Maximum number of earlier days to inspect.
    :type max_lookback_days: int or None
    :param refcal_sql_mode: SQL reference-calibration route.
    :type refcal_sql_mode: str
    :returns: Selected fallback description or ``None``.
    :rtype: dict or None
    """
    tim = Time(tim)
    if max_lookback_days is None:
        max_lookback_days = get_fallback_calibration_lookback_days()
    for day_offset in range(1, max_lookback_days + 1):
        cal_day = Time(tim.mjd - day_offset, format='mjd')
        readiness = _calibration_readiness_for_mode(
            cal_day,
            refcal_sql_mode,
        )
        if not readiness.get('ready'):
            continue
        _, etime = get_local_day_bounds(cal_day)
        lookup_time = Time(etime.mjd - 1.0 / 86400.0, format='mjd')
        return {
            'date': cal_day.iso[:10],
            'lookup_time_utc': lookup_time.iso,
            'lookback_days': day_offset,
            'readiness': readiness,
        }
    return None


def smart_cal_status_fields(readiness=None, fallback_calibration=None, sql_cal_time=None,
                            message=None, ran_after_deadline=False,
                            needs_same_day_calibration_rerun=False):
    """Build stable status fields for cron calibration decisions."""
    readiness = readiness or {}
    fields = dict(readiness)
    fields.update({
        'calibration_ready': readiness.get('ready'),
        'calibration_reason': readiness.get('reason'),
        'same_day_calibration_ready': readiness.get('ready'),
        'same_day_calibration_reason': readiness.get('reason'),
        'using_fallback_calibration': fallback_calibration is not None,
        'needs_same_day_calibration_rerun': bool(needs_same_day_calibration_rerun),
        'ran_after_deadline': bool(ran_after_deadline),
        'sql_cal_time_utc': Time(sql_cal_time).iso if sql_cal_time else None,
    })
    if fallback_calibration is not None:
        fallback_readiness = fallback_calibration.get('readiness') or {}
        fields.update({
            'fallback_calibration_date': fallback_calibration.get('date'),
            'fallback_calibration_lookup_time_utc': fallback_calibration.get('lookup_time_utc'),
            'fallback_calibration_lookback_days': fallback_calibration.get('lookback_days'),
            'fallback_refcal_timestamp_utc': fallback_readiness.get('refcal_timestamp_utc'),
            'fallback_phacal_count': fallback_readiness.get('phacal_count'),
            'fallback_latest_phacal_timestamp_utc': fallback_readiness.get('latest_phacal_timestamp_utc'),
        })
    if message:
        fields['message'] = message
    return fields


def summarize_applied_refcal_provenance(records, target_date):
    """Validate applied SQL refcal provenance and summarize the common record."""
    if not records:
        raise ValueError('missing applied refcal provenance')

    required_fields = (
        'vis',
        'lookup_time_utc',
        'source',
        'mode',
        'sql_record_time_utc',
        'refcal_time_utc',
        'refcal_date_utc',
    )
    signatures = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'refcal provenance record {index} is not a mapping')
        if record.get('applied') is not True:
            raise ValueError(f'refcal provenance record {index} was not applied')
        missing = [field for field in required_fields if not record.get(field)]
        if missing:
            raise ValueError(
                f'refcal provenance record {index} is missing {", ".join(missing)}'
            )
        signatures.add(tuple(record[field] for field in (
            'source',
            'mode',
            'sql_record_time_utc',
            'refcal_time_utc',
            'refcal_date_utc',
        )))

    if len(signatures) != 1:
        raise ValueError('inconsistent applied refcal provenance records')

    source, mode, sql_record_time, refcal_time, refcal_date = signatures.pop()
    btime, etime = get_local_day_bounds(Time(target_date))
    refcal_time_value = Time(refcal_time)
    return {
        'record_count': len(records),
        'source': source,
        'mode': mode,
        'sql_record_time_utc': sql_record_time,
        'refcal_time_utc': refcal_time,
        'refcal_date_utc': refcal_date,
        'same_day': bool(btime.mjd <= refcal_time_value.mjd < etime.mjd),
    }


def classify_synoptic_refcal_provenance(records, target_date, fits_complete):
    """Classify a smart SQL product from the refcal records actually applied."""
    summary = summarize_applied_refcal_provenance(records, target_date)
    using_fallback = not summary['same_day']
    if using_fallback:
        state = PROVISIONAL_SUCCESS_STATE if fits_complete else FALLBACK_PARTIAL_STATE
        message = (
            f'Completed with {summary["refcal_date_utc"]} SQL calibration; '
            f'will be replaced once {Time(target_date).iso[:10]} same-day calibration is ready.'
        )
    else:
        state = 'success' if fits_complete else 'partial'
        message = None
    calibration_reason = 'applied_fallback_refcal' if using_fallback else 'applied_same_day_refcal'

    return {
        'state': state,
        'message': message,
        'warning_calibration_date': summary['refcal_date_utc'] if using_fallback else None,
        'using_fallback_calibration': using_fallback,
        'needs_same_day_calibration_rerun': using_fallback,
        'ready': not using_fallback,
        'reason': calibration_reason,
        'calibration_ready': not using_fallback,
        'calibration_reason': calibration_reason,
        'same_day_calibration_ready': not using_fallback,
        'same_day_calibration_reason': calibration_reason,
        'refcal_timestamp_utc': summary['refcal_time_utc'],
        'applied_refcal_record_count': summary['record_count'],
        'applied_refcal_source': summary['source'],
        'applied_refcal_mode': summary['mode'],
        'applied_refcal_sql_record_time_utc': summary['sql_record_time_utc'],
        'applied_refcal_time_utc': summary['refcal_time_utc'],
        'applied_refcal_date_utc': summary['refcal_date_utc'],
    }


def should_enable_smart_cal_check(enable_flag=None):
    """Resolve the cron-oriented calibration gate behavior."""
    if enable_flag is not None:
        return enable_flag
    return os.getenv('EOVSA_PIPELINE_CRON') == '1'


def getspwfreq(vis):
    '''

    :param vis:
    :return: mid frequencies in GHz of each spw in the vis
    '''
    tb.open(vis + '/SPECTRAL_WINDOW')
    reffreqs = tb.getcol('REF_FREQUENCY')
    bdwds = tb.getcol('TOTAL_BANDWIDTH')
    cfreqs = reffreqs + bdwds / 2.
    tb.close()
    cfreqs = cfreqs / 1.0e9
    return cfreqs


def trange2ms(trange=None, doimport=False, verbose=False, doscaling=False,
              overwrite=True, prefer_scan_ms=False, import_outpath=None):
    '''This finds all solar UDBms files within a timerange; If the UDBms file does not exist 
       in EOVSAUDBMSSCL, create one by calling importeovsa
       Required inputs:
       trange - can be 1) a single string or Time() object in UTC: use the entire day, e.g., '2017-08-01' or Time('2017-08-01')
                          if just a date, find all scans withing the same date in local time. 
                          if a complete time stamp, find the local date first (which may be different from that provided, 
                            and return all scans within that day
                       2) a range of Time(), e.g., Time(['2017-08-01 00:00','2017-08-01 23:00'])
                       3) None -- use current date Time.now()
       doimport - Boolean. If true, call importeovsa to import UDB files that are missing from 
                  those found in the directory specified in EOVSAUDBMSSCL. Otherwise, return
                  a list of ms files it has found.
       doscaling - Boolean. If true, scale cross-correlation amplitudes by using auto-correlations
       verbose - Boolean. If true, return more information
       prefer_scan_ms - Boolean. If true, ignore the daily UDBYYYYMMDD.ms product
                        and return/import reusable raw scan MS files named
                        UDBYYYYMMDDhhmmss.ms.
       import_outpath - Optional processing-filesystem directory for newly
                        imported scan MS files. Existing durable scan MS files
                        are still reused from EOVSAUDBMS.
    '''
    import glob
    if trange is None:
        trange = Time.now()
    if type(trange) == list or type(trange) == str:
        try:
            trange = Time(trange)
        except:
            print('trange format not recognised. Abort....')
            return None

    # # in case of a single Time object was passed, adjusting the time range to start at the local beginning of the day.
    # # Initially, 'trange' is set to start at 20:00:00 UTC of the given day.
    # # After adjustment, 'trange' spans from 08:00:00 UTC of the same day (start of the local day)
    # # to 08:00:00 UTC of the following day, covering the entire local day.
    # # Example of change:
    # # Before: trange = "yyyy-mm-dd 20:00:00.000" (single starting point)
    # # After:  trange = ["yyyy-mm-dd 08:00:00.000", "yyyy-mm-dd+1 08:00:00.000"] (full day range)
    try:
        # if single Time object, the following line would report an error
        nt = len(trange)
        if len(trange) > 1:
            # more than one value
            trange = Time([trange[0], trange[-1]])
            tdatetime = trange[0].to_datetime()
        else:
            # single value in a list
            if trange[0].mjd == np.fix(trange[0].mjd):
                # if only date is given, move the time from 00 to 12 UT
                trange[0] = Time(trange[0].mjd + 0.5, format='mjd')

            tdatetime = trange[0].to_datetime()
            dhr = trange[0].LocalTime.utcoffset().total_seconds() / 60 / 60 / 24
            btime = Time(np.fix(trange[0].mjd + dhr) - dhr, format='mjd')
            etime = Time(btime.mjd + 1, format='mjd')
            trange = Time([btime, etime])
    except:
        # the case of a single Time object
        if trange.mjd == np.fix(trange.mjd):
            # if only date is given, move the time from 00 to 12 UT
            trange = Time(trange.mjd + 0.5, format='mjd')

        tdatetime = trange.to_datetime()
        dhr = trange.LocalTime.utcoffset().total_seconds() / 60 / 60 / 24
        btime = Time(np.fix(trange.mjd + dhr) - dhr, format='mjd')
        etime = Time(btime.mjd + 1, format='mjd')
        trange = Time([btime, etime])

    print('Selected idb files in the  time range (UTC): ', trange.iso)

    if doscaling:
        udbmspath = udbmsscldir
    else:
        udbmspath = udbmsdir
    inpath = '{}{}/'.format(udbdir, tdatetime.strftime("%Y"))
    outpath = '{}{}/'.format(udbmspath, tdatetime.strftime("%Y%m"))
    if not os.path.exists(outpath):
        if verbose:
            print(outpath + ' does not exist. Making a new directory.')
        os.makedirs(outpath)
        msfiles = []
    else:
        msfiles = [os.path.basename(ll).split('.')[0] for ll in glob.glob('{}UDB*.ms*'.format(outpath)) if
                   ll.endswith('.ms') or ll.endswith('.ms.tar.gz')]

    # Check the reusable processing-storage imports before deciding what to
    # import. Previously they were discovered only after importeovsa ran.
    if import_outpath:
        msfiles = set(msfiles).union(
            os.path.basename(candidate).split('.')[0]
            for candidate in glob.glob(os.path.join(import_outpath, 'UDB*.ms*'))
            if (candidate.endswith('.ms') and os.path.isdir(candidate))
            or (candidate.endswith('.ms.tar.gz') and os.path.isfile(candidate)))

    msfile_synoptic = os.path.join(outpath, 'UDB' + tdatetime.strftime("%Y%m%d") + '.ms')

    if os.path.exists(msfile_synoptic) and not prefer_scan_ms:
        if overwrite and doimport:
            os.system(f'rm -rf {msfile_synoptic}*')

    # sclist = ra.findfiles(trange, projid='NormalObserving', srcid='Sun')
    sclist = findfiles(trange, projid='NormalObserving', srcid='Sun')
    udbfilelist = sclist['scanlist']
    udbfilelist = [os.path.basename(ll) for ll in udbfilelist]

    if os.path.exists(msfile_synoptic) and not prefer_scan_ms:
        return {'mspath': outpath, 'udbpath': inpath, 'udbfile': sorted(udbfilelist), 'udb2ms': [],
                'ms': [msfile_synoptic],
                'tstlist': sclist['tstlist'], 'tedlist': sclist['tedlist']}
    else:
        udbfilelist_set = set(udbfilelist)
        msfiles = udbfilelist_set.intersection(msfiles)
        filelist = udbfilelist_set - msfiles
        filelist = sorted(list(filelist))
        if filelist and doimport:
            # import multiprocessing as mprocs
            # ncpu = mprocs.cpu_count()
            # if ncpu > 10:
            #    ncpu = 10
            # if ncpu > len(filelist):
            #    ncpu = len(filelist)
            ncpu = 1
            import_path = os.path.join(import_outpath, '') if import_outpath else outpath
            os.makedirs(import_path, exist_ok=True)
            importeovsa(idbfiles=[inpath + ll for ll in filelist], ncpu=ncpu, timebin="0s", width=1,
                        visprefix=import_path, nocreatms=False,
                        doconcat=False, modelms="", doscaling=doscaling, keep_nsclms=False, udb_corr=True)

        ms_locations = {}
        search_paths = [outpath]
        if import_outpath:
            search_paths.append(import_outpath)
        for search_path in search_paths:
            for candidate in glob.glob(os.path.join(search_path, 'UDB*.ms*')):
                if candidate.endswith('.ms') or candidate.endswith('.ms.tar.gz'):
                    scan_name = os.path.basename(candidate).split('.')[0]
                    unpacked = os.path.join(search_path, scan_name + '.ms')
                    ms_locations[scan_name] = unpacked if os.path.isdir(unpacked) else candidate
        msfiles = list(ms_locations)
        udbfilelist_set = set(udbfilelist)
        found_msfiles = udbfilelist_set.intersection(msfiles)
        filelist = udbfilelist_set - found_msfiles
        filelist = sorted(list(filelist))

        return {'mspath': outpath, 'udbpath': inpath, 'udbfile': sorted(udbfilelist), 'udb2ms': filelist,
                'ms': [ms_locations[ll] for ll in sorted(list(found_msfiles))], 'tstlist': sclist['tstlist'],
                'tedlist': sclist['tedlist']}


def calib_pipeline(trange, workdir=None, doimport=False, overwrite=False, clearcache=False, verbose=False, pols='XX',
                   version='v2.0', ncpu='auto', caltype=['refpha', 'phacal'], interp='nearest',
                   force_imaging_rerun=False, cal_npz=None, cal_tag=None, refcal_npz_mode='smooth_model',
                   secondary_npz=None, fine_spectral_imaging=False, fine_spectral_only=False,
                   custom_spws=None, force_lo_hi_smooth_extrap=False, refcal_sql_mode='bph_sbd',
                   sql_cal_time=None, force_feature_selfcal=False, imaging_only=False,
                   refcal_provenance=None, end_time=None, adaptive_s00=False,
                   input_ms=None):
    '''
       trange: can be 1) a single Time() object: use the entire day
                      2) a range of Time(), e.g., Time(['2017-08-01 00:00','2017-08-01 23:00'])
                      3) a single or a list of UDBms file(s)
                      4) None -- use current date Time.now()

       cal_npz: optional path to a calwidget_v2 calibeovsa NPZ (e.g.
                /common/webplots/phasecal/YYYYMMDD_calwidget_v2_calibeovsa.npz).
                When provided and version is v2.0 or v2.1, calibration is read from the
                NPZ via task_calibeovsa.calibeovsa instead of MySQL, and
                outputs are tagged so they do not collide with the production
                artefacts.
       cal_tag: required output filename tag for cal_npz runs.
       refcal_npz_mode: refcal apply mode for calwidget_v2 NPZ runs. ``triplet``
                preserves the legacy ph+sbd+mbd path; ``smooth_model`` applies
                sampled smooth refcal phase plus sbd and no refcal mbd table;
                ``bph_sbd`` applies saved band phase plus sbd and no refcal mbd
                table.
       secondary_npz: optional secondary calwidget_v2 calibeovsa NPZ. In
                ``bph_sbd`` runs, finite/unflagged secondary BPH fills missing
                primary BPH slots while primary SBD remains authoritative.
       force_lo_hi_smooth_extrap: in ``bph_sbd`` runs, force LO bands to use
                the HI smooth-model extrapolated phase base instead of LO BPH.
       refcal_sql_mode: no-NPZ SQL reference-calibration route. The default
                ``bph_sbd`` requires a complete digest-matched type-14+16
                BPH+SBD+BPS family and fails closed when it is unavailable.
       fine_spectral_imaging: bootstrap finer SPW chunks from the freshly
                imaged coarse parent, run one phase-only refinement, then run
                fine WSClean final imaging.
       fine_spectral_only: deprecated parentless resume mode; the request is
                rejected. Use imaging_only with fine_spectral_imaging so
                each coarse parent is regenerated first.
       imaging_only: rerun final (coarse + fine if fine_spectral_imaging) WSClean
                imaging from an existing selfcal'd MS product for this
                date/version/tag, skipping preprocessing and self-calibration.
                Fine resume requires an archive minted by the current
                seven-parent checkpoint path.
       input_ms: optional exact path to an existing ``.ms`` directory or
                ``.ms.tar.gz`` archive. This bypasses discovery, import, and
                calibeovsa. Without imaging_only it is treated as a calibrated
                daily MS and continues through selfcal plus imaging; with
                imaging_only it is treated as a selfcal product and only final
                imaging is run.
       custom_spws: optional WSClean FrequencySetup SPW grouping override. With
                fine_spectral_imaging, the exact 24-product 52-band plan is
                treated as fine outputs over the unchanged seven parents.
       sql_cal_time: optional SQL lookup time override for no-NPZ refcal/phacal
                selection. Used by cron provisional runs to image with a previous
                ready calibration day while keeping the target observing date.
       refcal_provenance: optional mutable list receiving the SQL refcal provenance
                actually applied to each successfully calibrated input MS.
       force_feature_selfcal: TEST ONLY: force feature self-calibration for all
                processed SPW groups, bypassing the brightness gate. Default off.
       end_time: optional UTC cutoff timestamp. Data at or after this time are
                flagged on staged scan MS inputs before calibration and selfcal.
       :param adaptive_s00: Explicitly opt in to the adaptive central-window s00
                route. The default preserves the historical segmented full-track
                route; the opt-in is limited to s00-01 without a time-range filter.
       :type adaptive_s00: bool
    '''

    version = normalize_pipeline_version(version)
    tdate = trange.datetime
    input_ms = normalize_reused_ms_input(input_ms, expected_date=tdate)
    validate_reused_ms_options(
        input_ms,
        doimport=doimport,
        cal_npz=cal_npz,
        secondary_npz=secondary_npz,
        sql_cal_time=sql_cal_time,
        end_time=end_time,
        force_lo_hi_smooth_extrap=force_lo_hi_smooth_extrap,
        version=version,
        imaging_only=imaging_only,
    )

    if cal_npz:
        cal_tag = get_default_cal_tag(version, cal_tag)
    elif not fine_spectral_only and not imaging_only:
        cal_tag = ''
    if fine_spectral_only:
        fine_spectral_imaging = True
        raise ValueError(
            'fine_spectral_only is no longer safe; use imaging_only with '
            'fine_spectral_imaging to regenerate the coarse parents first.')
    if end_time and imaging_only:
        raise ValueError('end_time requires calibration from staged scan MS inputs; imaging_only is unsafe.')
    # The Ant3 no-antenna schedule needs observing-state flags before any
    # averaging/selfcal. Reapply calibration on copies of reusable imports.
    preserve_scan_imports = (version in WSCLEAN_PIPELINE_VERSIONS
                             and tdate >= datetime(2026, 9, 1))
    apply_stow_flags = preserve_scan_imports and tdate < datetime(2026, 9, 10)
    use_imported_scan_ms = input_ms is None and bool(cal_npz or end_time or preserve_scan_imports)
    scan_import_dir = os.path.join(
        workdir_default, 'imported_ms', tdate.strftime('%Y%m%d'))

    if workdir is None:
        workdir = workdir_default
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    udbmspath = udbmsslfcaleddir
    outpath = os.path.join(udbmspath, tdate.strftime('%Y%m')) + '/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    imgoutdir = get_synoptic_product_output_dir(Time(tdate), version)
    if not os.path.exists(imgoutdir):
        os.makedirs(imgoutdir)
    figoutdir = os.path.join(synopticfigdir, tdate.strftime("%Y/"))
    if not os.path.exists(figoutdir):
        os.makedirs(figoutdir)

    ms_tag = f'.{cal_tag}' if cal_tag else ''
    storage_version = get_pipeline_storage_version(version)
    if version == 'v1.0':
        output_file_path = os.path.join(outpath, tdate.strftime('UDB%Y%m%d') + f'{ms_tag}.ms')
    else:
        output_file_path = os.path.join(outpath, tdate.strftime('UDB%Y%m%d') + f'.{storage_version}{ms_tag}.ms')
    slfcaltbdir_path = os.path.join(slfcaltbdir, tdate.strftime('%Y%m')) + '/'

    if fine_spectral_only:
        slfcaled_vis = None
        for candidate in (output_file_path, output_file_path + '.tar.gz'):
            if os.path.exists(candidate):
                slfcaled_vis = candidate
                break
        if slfcaled_vis is None:
            print('WARNING: fine_spectral_only requested, but no selfcal MS product was found.')
            print(f'Checked: {output_file_path} and {output_file_path}.tar.gz')
            return None
        if version not in WSCLEAN_PIPELINE_VERSIONS:
            print(f'fine_spectral_only is only supported for WSClean versions: {WSCLEAN_PIPELINE_VERSIONS}')
            return None
        from suncasa.eovsa import eovsa_synoptic_imaging_pipeline_wsclean as esip
        if verbose:
            print('input of fine-only pipeline_run:')
            print({'vis': slfcaled_vis,
                   'outputvis': '',
                   'workdir': workdir,
                   'slfcaltbdir': slfcaltbdir_path,
                   'imgoutdir': imgoutdir,
                   'figoutdir': figoutdir,
                   'overwrite': overwrite,
                   'clearcache': clearcache,
                   'pols': pols, 'ncpu': ncpu,
                   'fine_spectral_only': fine_spectral_only,
                   'fine_spectral_imaging': fine_spectral_imaging,
                   'custom_spws': custom_spws,
                   'force_feature_selfcal': force_feature_selfcal,
                   'adaptive_s00': adaptive_s00,
                   'preserve_input': input_ms is not None})
        return esip.pipeline_run(slfcaled_vis, outputvis='',
                                 workdir=workdir,
                                 slfcaltbdir=slfcaltbdir_path,
                                 imgoutdir=imgoutdir, pols=pols,
                                 overwrite=overwrite,
                                 fits_tag=cal_tag,
                                 fine_spectral_imaging=True,
                                 fine_spectral_only=True,
                                 custom_spws=custom_spws,
                                 force_feature_selfcal=force_feature_selfcal,
                                 adaptive_s00=adaptive_s00)

    if imaging_only:
        slfcaled_vis = input_ms
        if slfcaled_vis is None:
            for candidate in (output_file_path, output_file_path + '.tar.gz'):
                if os.path.exists(candidate):
                    slfcaled_vis = candidate
                    break
        if slfcaled_vis is None:
            print('WARNING: imaging_only requested, but no selfcal MS product was found.')
            print(f'Checked: {output_file_path} and {output_file_path}.tar.gz')
            return None
        if version not in WSCLEAN_PIPELINE_VERSIONS:
            print(f'imaging_only is only supported for WSClean versions: {WSCLEAN_PIPELINE_VERSIONS}')
            return None
        from suncasa.eovsa import eovsa_synoptic_imaging_pipeline_wsclean as esip
        if verbose:
            print('input of imaging-only pipeline_run:')
            print({'vis': slfcaled_vis,
                   'outputvis': '',
                   'workdir': workdir,
                   'slfcaltbdir': slfcaltbdir_path,
                   'imgoutdir': imgoutdir,
                   'figoutdir': figoutdir,
                   'overwrite': overwrite,
                   'clearcache': clearcache,
                   'pols': pols, 'ncpu': ncpu,
                   'fine_spectral_imaging': fine_spectral_imaging,
                   'fine_spectral_only': False,
                   'imaging_only': imaging_only,
                   'custom_spws': custom_spws,
                   'force_feature_selfcal': force_feature_selfcal,
                   'adaptive_s00': adaptive_s00,
                   'preserve_input': input_ms is not None})
        return esip.pipeline_run(slfcaled_vis, outputvis='',
                                 workdir=workdir,
                                 slfcaltbdir=slfcaltbdir_path,
                                 imgoutdir=imgoutdir, pols=pols,
                                 overwrite=overwrite,
                                 fits_tag=cal_tag,
                                 fine_spectral_imaging=fine_spectral_imaging,
                                 custom_spws=custom_spws,
                                 force_feature_selfcal=force_feature_selfcal,
                                 imaging_only=True,
                                 adaptive_s00=adaptive_s00,
                                 preserve_input=input_ms is not None)

    if input_ms is not None:
        invis = [input_ms]
    elif isinstance(trange, Time):
        mslist = trange2ms(trange=trange, doimport=False, prefer_scan_ms=use_imported_scan_ms,
                          import_outpath=scan_import_dir)
        invis = mslist['ms']
    elif isinstance(trange, str):
        try:
            mslist = trange2ms(trange=trange, doimport=False, prefer_scan_ms=use_imported_scan_ms,
                              import_outpath=scan_import_dir)
            invis = mslist['ms']
        except:
            invis = [trange]

    for idx, f in enumerate(invis):
        invis[idx] = os.path.normpath(f)

    fileexist = False

    vispath = os.path.join(udbmsdir, tdate.strftime('%Y%m'))
    vis = os.path.join(vispath, tdate.strftime('UDB%Y%m%d') + '.ms')
    if input_ms is not None:
        vis = input_ms
        fileexist = True
        print(f'Using explicit visibility input: {vis}')
        print('Skipping visibility discovery, import, and calibeovsa.')
    elif use_imported_scan_ms:
        fileexist = bool(invis)
        print('Trying to use reusable imported scan-level UDB MS inputs for calibration.')
        print(f'Imported scan-level UDB MS count: {len(invis)}')
    else:
        print(f'Trying to use visibility file: {vis}')
        if os.path.exists(vis):
            print(f'Visibility file {vis} exists.')
            fileexist = True
        else:
            if os.path.exists(f'{vis}.tar.gz'):
                print(f'Visibility file {vis}.tar.gz exists. Extracting...')
                fileexist = True
                vis= f'{vis}.tar.gz'
                # os.system(f'tar -xzf {vis}.tar.gz -C {vispath}')

    print(f'Visibility file exists: {fileexist}')
    if doimport:
        print(f'doimport: {doimport}')
        if overwrite and not use_imported_scan_ms:
            print('Overwriting existing visibility file...')
            fileexist = False
        elif overwrite and use_imported_scan_ms:
            print('Preserving imported scan-level UDB MS files; overwrite applies to downstream products.')
        elif fileexist:
            print('Visibility file already exists; reusing it instead of overwriting.')

    if (not fileexist) or use_imported_scan_ms:
        if not doimport:
            if use_imported_scan_ms and fileexist:
                print('Reusing imported scan-level UDB MS files; skipping importeovsa.')
            else:
                print('WARNING: No reusable visibility input exists and doimport=False. Aborting without import.')
                print(f'DEBUG calib_pipeline: trange={trange}, doimport={doimport}')
                return None

        if (not fileexist) or doimport:
            print('Visibility input does not exist or import was requested. Running import lookup...')
            if isinstance(trange, Time):
                mslist = trange2ms(
                    trange=trange,
                    doimport=doimport,
                    overwrite=overwrite,
                    prefer_scan_ms=use_imported_scan_ms,
                    import_outpath=scan_import_dir,
                )
                invis = mslist['ms']
            if isinstance(trange, str):
                try:
                    mslist = trange2ms(
                        trange=trange,
                        doimport=doimport,
                        overwrite=overwrite,
                        prefer_scan_ms=use_imported_scan_ms,
                        import_outpath=scan_import_dir,
                    )
                    invis = mslist['ms']
                except:
                    invis = [trange]

            for idx, f in enumerate(invis):
                invis[idx] = f.rstrip('/')

            print("DEBUG calib_pipeline: trange2ms result")
            print(f"  trange={trange}")
            print(f"  doimport={doimport}")
            print(f"  fileexist={fileexist}")
            print(f"  prefer_scan_ms={use_imported_scan_ms}")
            print(f"  mslist.keys()={list(mslist.keys())}")
            print(f"  n_invis={len(invis)}")
            print(f"  invis={invis}")
            print(f"  mspath={mslist.get('mspath')}")
            print(f"  udbpath={mslist.get('udbpath')}")
            print(f"  udbfile={mslist.get('udbfile')}")
            print(f"  udb2ms={mslist.get('udb2ms')}")

        if not invis:
            print('WARNING: No MS files were returned. Aborting.')
            print(f'DEBUG calib_pipeline: mslist={mslist}')
            return None

        cal_invis = stage_calibration_inputs(
            invis, workdir, cal_tag or None, preserve_imports=use_imported_scan_ms)
        if end_time:
            cal_invis = stage_time_cutoff_inputs(cal_invis, workdir, end_time)
        if apply_stow_flags:
            from suncasa.eovsa.antenna_flagging import flag_ant3_stow
            for staged_ms in cal_invis:
                flag_report = flag_ant3_stow(
                    staged_ms, provenance_path=staged_ms + '.ant3_flags.json')
                print('Ant3 observing-state flags before calibration: ' + json.dumps(flag_report))
        daily_ms_tag = f'.{cal_tag}' if cal_tag else ''
        outputvis = os.path.join(
            os.path.dirname(cal_invis[0]),
            os.path.basename(cal_invis[0])[:11] + f'{daily_ms_tag}.ms')
        if use_imported_scan_ms and os.path.exists(outputvis):
            remove_path(outputvis)
        tdate = get_tdate_from_basename(outputvis)
        flagant = '13~15' if Time(tdate).mjd >= EOVSA15_UPGRADE_DATE.mjd else '15'
        if cal_npz:
            # NPZ calibration path: same task as below, plus the NPZ-specific
            # arguments (read calibration from the NPZ instead of MySQL).
            vis = calibeovsa(cal_invis, caltype=caltype, caltbdir=caltbdir, interp=interp,
                             doflag=True,
                             flagant=flagant,
                             doimage=False, doconcat=True,
                             concatvis=outputvis, keep_orig_ms=False,
                             keep_corrected_column=True,
                             cal_npz=cal_npz, refcal_npz_mode=refcal_npz_mode,
                             secondary_npz=secondary_npz,
                             force_lo_hi_smooth_extrap=force_lo_hi_smooth_extrap,
                             refcal_sql_mode=refcal_sql_mode)
        else:
            vis = calibeovsa(cal_invis, caltype=caltype, caltbdir=caltbdir, interp=interp,
                             doflag=True,
                             flagant=flagant,
                             doimage=False, doconcat=True,
                             concatvis=outputvis, keep_orig_ms=False,
                             refcal_sql_mode=refcal_sql_mode,
                             sql_cal_time=sql_cal_time,
                             refcal_provenance=refcal_provenance)
    else:
        if verbose:
            print(f'Using existing visibility file: {vis}')


    # tdate = mstl.get_trange(vis)[0]
    tdate = get_tdate_from_basename(vis)

    if verbose:
        print('input of pipeline_run:')
        print({'vis': vis,
               'outputvis': output_file_path,
               'workdir': workdir,
               'slfcaltbdir': slfcaltbdir_path,
               'imgoutdir': imgoutdir,
               'figoutdir': figoutdir,
               'overwrite': overwrite,
               'clearcache': clearcache,
               'pols': pols, 'ncpu': ncpu,
               'fine_spectral_imaging': fine_spectral_imaging,
               'fine_spectral_only': fine_spectral_only,
               'custom_spws': custom_spws,
               'force_feature_selfcal': force_feature_selfcal,
               'adaptive_s00': adaptive_s00,
               'preserve_input': input_ms is not None})
    overwrite_pipeline = overwrite or force_imaging_rerun
    if force_imaging_rerun and version in WSCLEAN_PIPELINE_VERSIONS:
        print(f'Cron recovery mode enabled for {tdate.strftime("%Y-%m-%d")}: rerunning imaging despite existing outputvis.')

    if version == 'v1.0':
        vis = ed.pipeline_run(vis, outputvis=output_file_path,
                              workdir=workdir,
                              slfcaltbdir=slfcaltbdir_path,
                              imgoutdir=imgoutdir, figoutdir=figoutdir, clearcache=clearcache, pols=pols)
    elif version == LEGACY_PIPELINE_VERSION:
        from suncasa.eovsa import eovsa_synoptic_imaging_pipeline as esip
        vis = esip.pipeline_run(vis, outputvis=output_file_path,
                                workdir=workdir,
                                slfcaltbdir=slfcaltbdir_path,
                                imgoutdir=imgoutdir, figoutdir=figoutdir, clearcache=clearcache, pols=pols, ncpu=ncpu,
                                overwrite=overwrite)
    elif version in WSCLEAN_PIPELINE_VERSIONS:
        from suncasa.eovsa import eovsa_synoptic_imaging_pipeline_wsclean as esip
        vis = esip.pipeline_run(vis, outputvis=output_file_path,
                                workdir=workdir,
                                slfcaltbdir=slfcaltbdir_path,
                                imgoutdir=imgoutdir, pols=pols, overwrite=overwrite_pipeline,
                                fits_tag=cal_tag,
                                fine_spectral_imaging=fine_spectral_imaging,
                                fine_spectral_only=fine_spectral_only,
                                custom_spws=custom_spws,
                                force_feature_selfcal=force_feature_selfcal,
                                adaptive_s00=adaptive_s00,
                                preserve_input=input_ms is not None)
        if clearcache:
            if path_is_within(input_ms, workdir):
                clear_directory_preserving_path(workdir, input_ms)
            else:
                os.system(f'rm -rf {workdir}/*')
    else:
        print(f'Version {version} is not supported. Valid versions are {", ".join(SUPPORTED_PIPELINE_VERSIONS)}. Use the default version 2.0.')
        vis = ed.pipeline_run(vis, outputvis=output_file_path,
                              workdir=workdir,
                              slfcaltbdir=slfcaltbdir_path,
                              imgoutdir=imgoutdir, figoutdir=figoutdir, clearcache=clearcache, pols=pols)
    return vis


def mk_qlook_image(trange, doimport=False, docalib=False, ncpu=10, twidth=12, stokes=None, antenna='0~12',
                   lowcutoff_freq=3.7, imagedir=None, spws=['1~5', '6~10', '11~15', '16~25'], toTb=True, overwrite=True,
                   doslfcal=False, verbose=False):
    '''
       trange: can be 1) a single Time() object: use the entire day
                      2) a range of Time(), e.g., Time(['2017-08-01 00:00','2017-08-01 23:00'])
                      3) a single or a list of UDBms file(s)
                      4) None -- use current date Time.now()
    '''
    antenna0 = antenna
    if isinstance(trange, Time):
        mslist = trange2ms(trange=trange, doimport=doimport)
        vis = mslist['ms']
        tsts = [l.to_datetime() for l in mslist['tstlist']]
    if isinstance(trange, str):
        try:
            date = Time(trange)
            mslist = trange2ms(trange=trange, doimport=doimport)
            vis = mslist['ms']
            tsts = [l.to_datetime() for l in mslist['tstlist']]
        except:
            vis = [trange]
            tsts = []
            for v in vis:
                tb.open(v + '/OBSERVATION')
                tsts.append(Time(tb.getcell('TIME_RANGE')[0] / 24 / 3600, format='mjd').datetime)
                tb.close()
    subdir = [tst.strftime("%Y/%m/%d/") for tst in tsts]

    for idx, f in enumerate(vis):
        if f[-1] == '/':
            vis[idx] = f[:-1]
    if not stokes:
        stokes = 'XX'

    if not imagedir:
        imagedir = './'
    imres = {'Succeeded': [], 'BeginTime': [], 'EndTime': [], 'ImageName': [], 'Spw': [], 'Vis': [],
             'Synoptic': {'Succeeded': [], 'BeginTime': [], 'EndTime': [], 'ImageName': [], 'Spw': [], 'Vis': []}}
    for n, msfile in enumerate(vis):
        msfilebs = os.path.basename(msfile)
        imdir = imagedir + subdir[n]
        if not os.path.exists(imdir):
            os.makedirs(imdir)
        if doslfcal:
            slfcalms = './' + msfilebs + '.xx'
            split(msfile, outputvis=slfcalms, datacolumn='corrected', correlation='XX')
        cfreqs = getspwfreq(msfile)
        for spw in spws:
            antenna = antenna0
            if spw == '':
                continue
            spwran = [s.zfill(2) for s in spw.split('~')]
            freqran = [cfreqs[int(s)] for s in spw.split('~')]
            cfreq = np.mean(freqran)
            bmsz = max(150. / cfreq, 20.)
            uvrange = '<10klambda'
            if doslfcal:
                slfcal_img = './' + msfilebs + '.slf.spw' + spw.replace('~', '-') + '.slfimg'
                slfcal_tb = './' + msfilebs + '.slf.spw' + spw.replace('~', '-') + '.slftb'
                try:
                    tclean(vis=slfcalms, antenna=antenna, imagename=slfcal_img, spw=spw, mode='mfs', timerange='',
                           deconvolver='hogbom',
                           imsize=[512, 512], cell=['5arcsec'], niter=100, gain=0.05, stokes='I',
                           weighting='natural',
                           restoringbeam=[str(bmsz) + 'arcsec'], pbcor=False, interactive=False, usescratch=True)
                except:
                    print('error in cleaning spw: ' + spw)
                    break
                gaincal(vis=slfcalms, refant='0', antenna=antenna, caltable=slfcal_tb, spw=spw, uvrange='',
                        gaintable=[], selectdata=True,
                        timerange='', solint='600s', gaintype='G', calmode='p', combine='', minblperant=3, minsnr=2,
                        append=False)
                if not os.path.exists(slfcal_tb):
                    print('No solution found in spw: ' + spw)
                    break
                else:
                    clearcal(slfcalms)
                    delmod(slfcalms)
                    applycal(vis=slfcalms, gaintable=[slfcal_tb], spw=spw, selectdata=True, antenna=antenna,
                             interp='nearest', flagbackup=False,
                             applymode='calonly', calwt=False)
                    msfile = slfcalms

            imsize = 512
            cell = ['5arcsec']
            if len(spwran) == 2:
                spwstr = spwran[0] + '~' + spwran[1]
            else:
                spwstr = spwran[0]

            restoringbeam = ['{0:.1f}arcsec'.format(bmsz)]
            imagesuffix = '.spw' + spwstr.replace('~', '-')
            if cfreq > 10.:
                antenna = antenna + ';!0&1;!0&2'  # deselect the shortest baselines
            # else:
            #     antenna = antenna + ';!0&1'  # deselect the shortest baselines

            res = ptclean(vis=msfile, imageprefix=imdir, imagesuffix=imagesuffix, twidth=twidth, uvrange=uvrange,
                          spw=spw, ncpu=ncpu, niter=1000,
                          gain=0.05, antenna=antenna, imsize=imsize, cell=cell, stokes=stokes, doreg=True,
                          usephacenter=False, overwrite=overwrite,
                          toTb=toTb, restoringbeam=restoringbeam, specmode="mfs", deconvolver="hogbom",
                          datacolumn='data', pbcor=True)

            if res:
                imres['Succeeded'] += res['Succeeded']
                imres['BeginTime'] += res['BeginTime']
                imres['EndTime'] += res['EndTime']
                imres['ImageName'] += res['ImageName']
                imres['Spw'] += [spwstr] * len(res['ImageName'])
                imres['Vis'] += [msfile] * len(res['ImageName'])
            else:
                continue

    if len(vis) == 1:
        # produce the band-by-band whole-day images
        ms.open(msfile)
        ms.selectinit()
        timfreq = ms.getdata(['time', 'axis_info'], ifraxis=True)
        tim = timfreq['time']
        ms.done()

        cfreqs = getspwfreq(msfile)
        imdir = imagedir + subdir[0]
        if not os.path.exists(imdir):
            os.makedirs(imdir)
        for spw in spws:
            antenna = antenna0
            if spw == '':
                spw = '{:d}~{:d}'.format(next(x[0] for x in enumerate(cfreqs) if x[1] > lowcutoff_freq),
                                         len(cfreqs) - 1)
            spwran = [s.zfill(2) for s in spw.split('~')]
            freqran = [cfreqs[int(s)] for s in spw.split('~')]
            cfreq = np.mean(freqran)
            bmsz = max(150. / cfreq, 20.)
            uvrange = ''
            imsize = 512
            cell = ['5arcsec']
            if len(spwran) == 2:
                spwstr = spwran[0] + '~' + spwran[1]
            else:
                spwstr = spwran[0]

            restoringbeam = ['{0:.1f}arcsec'.format(bmsz)]
            imagesuffix = '.synoptic.spw' + spwstr.replace('~', '-')
            antenna = antenna + ';!0&1'  # deselect the shortest baselines

            res = ptclean(vis=msfile, imageprefix=imdir, imagesuffix=imagesuffix, twidth=len(tim), uvrange=uvrange,
                          spw=spw, ncpu=1, niter=0,
                          gain=0.05, antenna=antenna, imsize=imsize, cell=cell, stokes=stokes, doreg=True,
                          usephacenter=False, overwrite=overwrite,
                          toTb=toTb, restoringbeam=restoringbeam, specmode="mfs", deconvolver="hogbom",
                          datacolumn='data', pbcor=True)
            if res:
                imres['Synoptic']['Succeeded'] += res['Succeeded']
                imres['Synoptic']['BeginTime'] += res['BeginTime']
                imres['Synoptic']['EndTime'] += res['EndTime']
                imres['Synoptic']['ImageName'] += res['ImageName']
                imres['Synoptic']['Spw'] += [spwstr] * len(res['ImageName'])
                imres['Synoptic']['Vis'] += [msfile] * len(res['ImageName'])
            else:
                continue

    # save it for debugging purposes
    np.savez('imres.npz', imres=imres)

    return imres


def plt_qlook_image(imres, figdir=None, verbose=True, synoptic=False):
    from matplotlib import pyplot as plt
    from sunpy import map as smap
    from sunpy import sun
    from matplotlib import colors
    import astropy.units as u
    from suncasa.utils import plot_mapX as pmX
    # from matplotlib import gridspec as gridspec

    if not figdir:
        figdir = './'

    nspw = len(set(imres['Spw']))
    plttimes = list(set(imres['BeginTime']))
    ntime = len(plttimes)
    # sort the imres according to time
    images = np.array(imres['ImageName'])
    btimes = Time(imres['BeginTime'])
    etimes = Time(imres['EndTime'])
    spws = np.array(imres['Spw'])
    suc = np.array(imres['Succeeded'])
    inds = btimes.argsort()
    images_sort = images[inds].reshape(ntime, nspw)
    btimes_sort = btimes[inds].reshape(ntime, nspw)
    suc_sort = suc[inds].reshape(ntime, nspw)
    if verbose:
        print('{0:d} figures to plot'.format(ntime))
    plt.ioff()
    fig = plt.figure(figsize=(8, 8))

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axs = []
    ims = []
    pltst = 0
    for i in range(ntime):
        plt.ioff()
        plttime = btimes_sort[i, 0]
        tofd = plttime.mjd - np.fix(plttime.mjd)
        suci = suc_sort[i]
        if not synoptic:
            if tofd < 16. / 24. or sum(
                    suci) < nspw - 2:  # if time of the day is before 16 UT (and 24 UT), skip plotting (because the old antennas are not tracking)
                continue
            else:
                if pltst == 0:
                    i0 = i
                    pltst = 1
        else:
            if pltst == 0:
                i0 = i
                pltst = 1
        if i == i0:
            if synoptic:
                timetext = fig.text(0.01, 0.98, plttime.iso[:10], color='w', fontweight='bold', fontsize=12, ha='left')
            else:
                timetext = fig.text(0.01, 0.98, plttime.iso[:19], color='w', fontweight='bold', fontsize=12, ha='left')
        else:
            if synoptic:
                timetext.set_text(plttime.iso[:10])
            else:
                timetext.set_text(plttime.iso[:19])
        if verbose:
            print('Plotting image at: ', plttime.iso)
        for n in range(nspw):
            plt.ioff()
            if i == i0:
                if nspw == 1:
                    ax = fig.add_subplot(111)
                else:
                    ax = fig.add_subplot(nspw / 2, 2, n + 1)
                axs.append(ax)
            else:
                ax = axs[n]
            image = images_sort[i, n]
            if suci[n] or os.path.exists(image):
                try:
                    eomap = smap.Map(image)
                except:
                    continue
                data = eomap.data
                sz = data.shape
                if len(sz) == 4:
                    data = data.reshape((sz[2], sz[3]))
                data[np.isnan(data)] = 0.0
                # add a basin flux to the image to avoid negative values
                data = data + 0.8e5
                data[data < 0] = 0.0
                data = np.sqrt(data)
                eomap = smap.Map(data, eomap.meta)
                # resample the image for plotting
                dim = u.Quantity([256, 256], u.pixel)
                eomap = eomap.resample(dim)
            else:
                # make an empty map
                data = np.zeros((256, 256))
                header = {"DATE-OBS": plttime.isot, "EXPTIME": 0., "CDELT1": 10., "NAXIS1": 256, "CRVAL1": 0.,
                          "CRPIX1": 128.5, "CUNIT1": "arcsec",
                          "CTYPE1": "HPLN-TAN", "CDELT2": 10., "NAXIS2": 256, "CRVAL2": 0., "CRPIX2": 128.5,
                          "CUNIT2": "arcsec", "CTYPE2": "HPLT-TAN",
                          "HGLT_OBS": sun.heliographic_solar_center(plttime)[1].value, "HGLN_OBS": 0.,
                          "RSUN_OBS": sun.solar_semidiameter_angular_size(plttime).value,
                          "RSUN_REF": sun.constants.radius.value,
                          "DSUN_OBS": sun.sunearth_distance(plttime).to(u.meter).value, }
                eomap = smap.Map(data, header)
            if i == i0:
                eomap_ = pmX.Sunmap(eomap)
                # im = eomap_.imshow(axes=ax, cmap='jet', norm=colors.LogNorm(vmin=0.1, vmax=1e8))
                im = eomap_.imshow(axes=ax, cmap='jet', norm=colors.Normalize(vmin=150, vmax=700))
                ims.append(im)
                if not synoptic:
                    eomap_.draw_limb(axes=ax)
                eomap_.draw_grid(axes=ax)
                ax.set_xlim([-1080, 1080])
                ax.set_ylim([-1080, 1080])
                try:
                    cfreq = eomap.meta['crval3'] / 1.0e9
                    bdwid = eomap.meta['cdelt3'] / 1.0e9
                    ax.text(0.98, 0.01, '{0:.1f} - {1:.1f} GHz'.format(cfreq - bdwid / 2.0, cfreq + bdwid / 2.0),
                            color='w', transform=ax.transAxes, fontweight='bold', ha='right')
                except:
                    pass
                ax.set_title(' ')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticklabels([''])
                ax.set_yticklabels([''])
            else:
                ims[n].set_data(eomap.data)

        fig_tdt = plttime.to_datetime()
        if synoptic:
            fig_subdir = fig_tdt.strftime("%Y/")
            figname = 'eovsa_qlimg_' + plttime.iso[:10].replace('-', '') + '.png'
        else:
            fig_subdir = fig_tdt.strftime("%Y/%m/%d/")
            figname = 'eovsa_qlimg_' + plttime.isot.replace(':', '').replace('-', '')[:15] + '.png'
        figdir_ = figdir + fig_subdir
        if not os.path.exists(figdir_):
            os.makedirs(figdir_)
        if verbose:
            print('Saving plot to :' + figdir_ + figname)

        plt.savefig(figdir_ + figname)
    plt.close(fig)


def qlook_image_pipeline(date, twidth=10, ncpu=15, doimport=False, docalib=False, synoptic=False, overwrite=True):
    ''' date: date string or Time object. e.g., '2017-07-15' or Time('2017-07-15')
    '''
    import pytz
    from datetime import datetime
    if date is None:
        date = Time.now()
    try:
        date = Time(date)
    except:
        print('date format not recognised. Abort....')
        return None

    if date.mjd >= DCM_IF_FILTER_UPGRADE_DATE.mjd:
        ## the last '' window is for fullBD synthesis image. Now obsolete.
        # spws = ['6~10', '11~20', '21~30', '31~43', '']
        # spws = ['6~10', '11~20', '21~30', '31~43']
        spws = list(SPWS_52BAND_SELFCAL)
    else:
        ## the last '' window is for fullBD synthesis image. Now obsolete.
        # spws = ['1~5', '6~10', '11~15', '16~25', '']
        # NOTE: this legacy branch uses a one-off pre-2019 grouping (distinct
        # from SPWS_34BAND); left inline as it is not part of the shared config.
        spws = ['1~3', '4~6', '7~10', '10~14', '15~20', '21~30']

    if docalib:
        vis = calib_pipeline(date, doimport=doimport, synoptic=synoptic)

    imagedir = qlookfitsdir
    if synoptic:
        vis_synoptic = os.path.join(udbmsdir, date.datetime.strftime("%Y%m"),
                                    'UDB' + date.datetime.strftime("%Y%m%d") + '.ms')
        if os.path.exists(vis_synoptic):
            date = vis_synoptic
        else:
            print('Whole-day ms file {} not existed. About..... Use pipeline to make one.'.format(vis_synoptic))
            return None

    imres = mk_qlook_image(date, twidth=twidth, ncpu=ncpu, doimport=doimport, docalib=docalib, imagedir=imagedir,
                           spws=spws, verbose=True, overwrite=overwrite)

    figdir = qlookfigdir
    plt_qlook_image(imres, figdir=figdir, verbose=True)
    # if imres['Synoptic']['Succeeded']:
    #     figdir = synopticfigdir
    #     imres_bds = {}
    #     # imres_allbd = {}
    #     for k, v in imres['Synoptic'].items():
    #         imres_bds[k] = v  # [:4]
    #         # imres_allbd[k] = v[4:]
    #
    #     plt_qlook_image(imres_bds, figdir=figdir, verbose=True, synoptic=True)
    #     # plt_qlook_image(imres_allbd, figdir=figdir + 'FullBD/', verbose=True, synoptic=True)


def pipeline(year=None, month=None, day=None, ndays=1, clearcache=True, overwrite=False, doimport=True, pols='XX',
             version='v2.0', ncpu='auto', debugging=False, caltype=['refpha', 'phacal'], interp='nearest',
             smart_cal_check=None, cal_npz=None, cal_tag=None, refcal_npz_mode='smooth_model',
             secondary_npz=None, fine_spectral_imaging=False, fine_spectral_only=False,
             custom_spws=None, force_lo_hi_smooth_extrap=False, refcal_sql_mode='bph_sbd',
             sql_cal_time=None, force_feature_selfcal=False, imaging_only=False,
             end_time=None, adaptive_s00=False, input_ms=None):
    """
    Main pipeline for importing and calibrating EOVSA visibility data.

    Name:
        eovsa_pipeline --- main pipeline for importing and calibrating EOVSA visibility data.

    Synopsis:
        eovsa_pipeline.py [options]... [DATE_IN_YY_MM_DD]

    Description:
        Import and calibrate EOVSA visibility data of the date specified
        by DATE_IN_YY_MM_DD (or from ndays before the DATE_IN_YY_MM_DD if option --ndays/-n is provided).
        If DATE_IN_YY_MM_DD is omitted, it will be set to 2 days before now by default.
        There are no mandatory arguments in this command.

    :param year: The year for which data should be processed, defaults to None.
    :type year: int, optional
    :param month: The month for which data should be processed, defaults to None.
    :type month: int, optional
    :param day: The day for which data should be processed, defaults to None.
    :type day: int, optional
    :param ndays: Number of days before the specified date to include in the processing, defaults to 1.
    :type ndays: int, optional
    :param clearcache: Whether to clear cache after processing, defaults to True.
    :type clearcache: bool, optional
    :param overwrite: Whether to overwrite existing files, defaults to True.
    :type overwrite: bool, optional
    :param doimport: Whether to perform the import step, defaults to True.
    :type doimport: bool, optional
    :param pols: Polarizations to process, can be 'XX', 'YY', or 'XXYY', defaults to 'XX'.
    :type pols: str, optional
    :param version: Version of the pipeline to use. Public choices are 'v1.0' and
        'v2.0'; internal WSClean subversions use the v2.1 naming family.
        Defaults to 'v2.0'.
    :type version: str, optional
    :param ncpu: Number of CPUs to use for processing, defaults to 'auto'.
    :type ncpu: str, optional
    :param debugging: Whether to run the pipeline in debugging mode, defaults to False.
    :type debugging: bool, optional
    :param caltype: Calibration types to use, defaults to ['refpha','phacal'].
    :type caltype: list, optional
    :param interp: Interpolation method to use for calibration tables, defaults to 'nearest'. Options are 'nearest', 'linear'
    :type interp: str, optional
    :param smart_cal_check: When True, perform cron-oriented calibration readiness checks and
        track per-day status before processing, defaults to ``None``. If ``None``, the behavior
        is enabled automatically when ``EOVSA_PIPELINE_CRON=1``. The gate prefers same
        observing-day calibration records; if they are not ready but an older observing day
        is ready, it runs provisionally with that older SQL calibration and marks the products
        for replacement when same-day calibration becomes ready.
    :type smart_cal_check: bool, optional
    :param cal_npz: optional path to a calwidget_v2 calibeovsa NPZ.
    :type cal_npz: str, optional
    :param cal_tag: required output filename tag for cal_npz runs.
    :type cal_tag: str, optional
    :param refcal_npz_mode: refcal apply mode for calwidget_v2 NPZ runs.
    :type refcal_npz_mode: str, optional
    :param secondary_npz: optional secondary calwidget_v2 calibeovsa NPZ for
        BPH fill in ``bph_sbd`` runs.
    :type secondary_npz: str, optional
    :param force_lo_hi_smooth_extrap: force LO bands to use the HI smooth-model
        extrapolated phase base in ``bph_sbd`` runs.
    :type force_lo_hi_smooth_extrap: bool, optional
    :param refcal_sql_mode: no-NPZ SQL reference-calibration route. The default
        ``bph_sbd`` requires a complete digest-matched type-14+16 BPH+SBD+BPS
        family and fails closed when it is unavailable.
    :type refcal_sql_mode: str, optional
    :param fine_spectral_imaging: bootstrap finer SPW chunks from freshly
        imaged coarse parents, run one phase-only refinement, and then run fine
        WSClean final imaging.
    :type fine_spectral_imaging: bool, optional
    :param fine_spectral_only: deprecated parentless resume mode; the request
        is rejected. Use ``imaging_only`` with ``fine_spectral_imaging`` to
        regenerate each parent first.
    :type fine_spectral_only: bool, optional
    :param end_time: Optional UTC cutoff timestamp. Samples at or after this
        time are flagged before calibration and selfcal on staged scan MS inputs.
    :type end_time: str, optional
    :param imaging_only: rerun final (coarse + fine if fine_spectral_imaging)
        WSClean imaging from an existing selfcal'd MS product for this
        date/version/tag, skipping preprocessing and self-calibration. Fine
        resume requires an archive minted by the current seven-parent path.
    :type imaging_only: bool, optional
    :param input_ms: exact existing ``.ms`` directory or ``.ms.tar.gz`` archive
        to reuse. This independently bypasses discovery, import, and calibeovsa.
        Without ``imaging_only`` the calibrated input continues through selfcal
        and imaging; with ``imaging_only`` a selfcal input is imaged directly.
    :type input_ms: str, optional
    :param custom_spws: optional WSClean FrequencySetup SPW grouping override.
        With fine imaging, the exact 24-product 52-band plan is treated as fine
        outputs over the unchanged seven parents.
    :type custom_spws: list or str, optional
    :param adaptive_s00: explicitly opt in to adaptive central-window imaging
        for s00-01. The default preserves historical segmented full-track
        imaging, and the opt-in is suppressed when a time-range filter is used.
    :type adaptive_s00: bool, optional
    :param sql_cal_time: optional SQL calibration lookup timestamp override.
        This is mainly for cron fallback runs that image the target date using
        an older ready calibration day.
    :type sql_cal_time: str, optional
    :param force_feature_selfcal: TEST ONLY: force feature self-calibration for
        all processed SPW groups, bypassing the brightness gate. Default off.
    :type force_feature_selfcal: bool, optional

    :raises ValueError: Raises an exception if the date parameters are out of the valid Gregorian calendar range.

    Example:
    --------
    To process data for November 24th, 2021 using version 2.0 of the pipeline, with all options enabled:

    >>> python eovsa_pipeline.py --date 2021-11-24T20:00 --clearcache --overwrite --doimport --pols XX --version v2.0 --ndays 2

    If you want to see the help message, you can run:

    >>> python eovsa_pipeline.py -h
    """
    version = normalize_pipeline_version(version)
    if fine_spectral_only:
        raise ValueError(
            'fine_spectral_only is no longer safe; use imaging_only with '
            'fine_spectral_imaging to regenerate the coarse parents first.')
    input_ms = normalize_reused_ms_input(input_ms)
    smart_cal_check = should_enable_smart_cal_check(smart_cal_check)
    validate_reused_ms_options(
        input_ms,
        doimport=doimport,
        smart_cal_check=smart_cal_check,
        cal_npz=cal_npz,
        secondary_npz=secondary_npz,
        sql_cal_time=sql_cal_time,
        end_time=end_time,
        force_lo_hi_smooth_extrap=force_lo_hi_smooth_extrap,
        ndays=ndays,
        version=version,
        imaging_only=imaging_only,
    )
    if cal_npz:
        # Calwidget_v2 NPZ supplies refcal+phacal directly, so MySQL readiness
        # gating is not applicable. Disable smart_cal_check in test runs to
        # avoid querying SQL for records the run is intentionally bypassing.
        smart_cal_check = False
        cal_tag = get_default_cal_tag(version, cal_tag)
    else:
        cal_tag = ''
    fits_tag = cal_tag
    workdir = workdir_default
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    if year is None:
        # Default behavior: Process data from one day prior to the current date.
        # Calculate the Modified Julian Date (MJD) for yesterday.
        mjdnow = Time.now().mjd - 1
        # Convert MJD to a datetime object and format it to start processing at 20:00 UT.
        t = Time(Time(mjdnow, format='mjd').to_datetime().strftime('%Y-%m-%dT20:00'))
    else:
        t = Time('{}-{:02d}-{:02d} 20:00'.format(year, month, day))
    if input_ms is not None:
        input_ms = normalize_reused_ms_input(input_ms, expected_date=t.datetime)
    failed_dates = []
    for d in range(ndays):
        t1 = Time(t.mjd - d, format='mjd')
        datestr = t1.iso[:10]
        synoptic_info = summarize_synoptic_outputs(t1, version=version, fits_tag=fits_tag)
        statusfile = synoptic_info['statusfile']
        is_wsclean_version = version in WSCLEAN_PIPELINE_VERSIONS
        refcal_provenance_records = [] if smart_cal_check and is_wsclean_version else None
        readiness = {}
        fallback_calibration = None
        sql_cal_time_for_run = sql_cal_time
        run_state = 'running'
        run_message = ''
        rerun_provisional_with_same_day_calibration = False
        overwrite_for_run = overwrite or (sql_cal_time_for_run is not None)
        if smart_cal_check and is_wsclean_version:
            previous_status = read_pipeline_status(statusfile)
            previous_state = previous_status.get('state', '')
            previous_base_state = previous_status.get(
                'imaging_base_state', previous_state
            )
            outputvis_root = os.path.join(
                udbmsslfcaleddir,
                t1.datetime.strftime('%Y%m'),
                t1.datetime.strftime('UDB%Y%m%d') + f'.{get_pipeline_storage_version(version)}.ms'
            )
            outputvis_exists = os.path.exists(outputvis_root) or os.path.exists(f'{outputvis_root}.tar.gz')
            # Seven standard daily products do not complete an explicitly
            # requested hierarchical fine run.  Always enter the run path so
            # missing/rejected fine children can be (re)attempted.
            if synoptic_info['fits_complete'] and not fine_spectral_imaging and not overwrite:
                if previous_base_state == PROVISIONAL_SUCCESS_STATE:
                    readiness = _calibration_readiness_for_mode(
                        t1,
                        refcal_sql_mode,
                    )
                    if not readiness['ready']:
                        run_message = previous_status.get('message') or (
                            f'Completed with fallback calibration; waiting for {datestr} '
                            'same-day calibration before replacing products.'
                        )
                        status_extra = smart_cal_status_fields(
                            readiness,
                            sql_cal_time=previous_status.get('sql_cal_time_utc'),
                            message=run_message,
                            needs_same_day_calibration_rerun=True,
                        )
                        for key in (
                            'fallback_calibration_date',
                            'fallback_calibration_lookup_time_utc',
                            'fallback_calibration_lookback_days',
                            'fallback_refcal_timestamp_utc',
                            'fallback_phacal_count',
                            'fallback_latest_phacal_timestamp_utc',
                            'applied_refcal_record_count',
                            'applied_refcal_source',
                            'applied_refcal_mode',
                            'applied_refcal_sql_record_time_utc',
                            'applied_refcal_time_utc',
                            'applied_refcal_date_utc',
                        ):
                            if key in previous_status:
                                status_extra[key] = previous_status[key]
                        status_extra['using_fallback_calibration'] = bool(
                            previous_status.get('using_fallback_calibration', True)
                        )
                        completed_state, imaging_qa = apply_s00_qa_to_pipeline_state(
                            PROVISIONAL_SUCCESS_STATE,
                            synoptic_info['existing_fitsfiles']
                        )
                        status_extra.update(imaging_qa)
                        write_pipeline_status(
                            statusfile,
                            completed_state,
                            date=datestr,
                            fits_count=synoptic_info['fits_count'],
                            fits_expected_count=synoptic_info['fits_expected_count'],
                            fitsfiles=synoptic_info['existing_fitsfiles'],
                            outputvis_exists=outputvis_exists,
                            **status_extra,
                        )
                        print(
                            f'Synoptic FITS are provisionally complete for {datestr}; '
                            f'same-day calibration is not ready yet ({readiness["reason"]}).'
                        )
                        continue
                    rerun_provisional_with_same_day_calibration = True
                    run_message = (
                        f'Same-day calibration is now ready for {datestr}; '
                        'replacing provisional fallback-calibrated products.'
                    )
                    print(run_message)
                else:
                    completed_state, imaging_qa = apply_s00_qa_to_pipeline_state(
                        'success',
                        synoptic_info['existing_fitsfiles']
                    )
                    status_extra = dict(imaging_qa)
                    write_pipeline_status(
                        statusfile,
                        completed_state,
                        date=datestr,
                        fits_count=synoptic_info['fits_count'],
                        fits_expected_count=synoptic_info['fits_expected_count'],
                        fitsfiles=synoptic_info['existing_fitsfiles'],
                        outputvis_exists=outputvis_exists,
                        **status_extra,
                    )
                    print(f'Synoptic FITS already complete for {datestr}. Skipping cron run.')
                    continue

            if not readiness:
                readiness = _calibration_readiness_for_mode(
                    t1,
                    refcal_sql_mode,
                )

            if sql_cal_time_for_run is not None:
                run_state = 'running_with_sql_cal_time_override'
                run_message = (
                    f'Running {datestr} with explicit SQL calibration lookup time '
                    f'{Time(sql_cal_time_for_run).iso}.'
                )
            elif not readiness['ready']:
                fallback_calibration = find_previous_ready_calibration(
                    t1,
                    refcal_sql_mode=refcal_sql_mode,
                )
                if fallback_calibration is not None:
                    sql_cal_time_for_run = fallback_calibration['lookup_time_utc']
                    run_state = FALLBACK_RUNNING_STATE
                    run_message = (
                        f'Calibration is not ready for {datestr} ({readiness["reason"]}); '
                        f'running provisionally with {fallback_calibration["date"]} '
                        'SQL calibration. Products will be replaced once the same-day '
                        'calibration is ready.'
                    )
                elif readiness.get('deadline_expired'):
                    run_state = 'running_after_deadline'
                    run_message = (
                        f'Calibration still not ready for {datestr} after hard deadline '
                        f'{readiness["hard_deadline_utc"]}; running with the most recent '
                        f'calibration records currently available in MySQL.'
                    )
                else:
                    write_pipeline_status(
                        statusfile,
                        'waiting_for_calibration',
                        date=datestr,
                        fits_count=synoptic_info['fits_count'],
                        fits_expected_count=synoptic_info['fits_expected_count'],
                        fitsfiles=synoptic_info['existing_fitsfiles'],
                        outputvis_exists=outputvis_exists,
                        **smart_cal_status_fields(readiness),
                    )
                    print(f'Skipping {datestr}: calibration not ready ({readiness["reason"]}).')
                    continue

            if run_message:
                print(run_message)

            overwrite_for_run = (
                overwrite
                or bool(fallback_calibration)
                or rerun_provisional_with_same_day_calibration
                or (sql_cal_time_for_run is not None)
            )
            write_pipeline_status(
                statusfile,
                run_state,
                date=datestr,
                fits_count=synoptic_info['fits_count'],
                fits_expected_count=synoptic_info['fits_expected_count'],
                fitsfiles=synoptic_info['existing_fitsfiles'],
                outputvis_exists=outputvis_exists,
                **smart_cal_status_fields(
                    readiness,
                    fallback_calibration=fallback_calibration,
                    sql_cal_time=sql_cal_time_for_run,
                    message=run_message,
                    ran_after_deadline=run_state == 'running_after_deadline',
                    needs_same_day_calibration_rerun=fallback_calibration is not None,
                ),
            )
        subdir = os.path.join(workdir, t1.datetime.strftime('%Y%m%d/'))
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        else:
            if overwrite_for_run:
                input_in_subdir = path_is_within(input_ms, subdir)
                if input_in_subdir:
                    print(
                        f'Preserving explicit input_ms inside {subdir}; '
                        'clearing other work products for overwrite.'
                    )
                    clear_directory_preserving_path(subdir, input_ms)
                else:
                    os.system('rm -rf {}/*'.format(subdir))
        # ##debug
        # vis_corrected = calib_pipeline(datestr, overwrite=overwrite, doimport=doimport,
        #                                workdir=subdir, clearcache=False, pols=pols)

        if debugging:
            vis_corrected = calib_pipeline(t1, overwrite=overwrite_for_run, doimport=doimport,
                                           workdir=subdir, clearcache=False, pols=pols, version=version, ncpu=ncpu,
                                           caltype=caltype, interp=interp,
                                           force_imaging_rerun=smart_cal_check and is_wsclean_version,
                                           cal_npz=cal_npz, cal_tag=cal_tag, refcal_npz_mode=refcal_npz_mode,
                                           secondary_npz=secondary_npz,
                                           fine_spectral_imaging=fine_spectral_imaging,
                                           fine_spectral_only=fine_spectral_only,
                                           custom_spws=custom_spws,
                                           force_lo_hi_smooth_extrap=force_lo_hi_smooth_extrap,
                                           refcal_sql_mode=refcal_sql_mode,
                                           sql_cal_time=sql_cal_time_for_run,
                                           force_feature_selfcal=force_feature_selfcal,
                                           imaging_only=imaging_only,
                                           refcal_provenance=refcal_provenance_records,
                                           end_time=end_time,
                                           adaptive_s00=adaptive_s00,
                                           input_ms=input_ms)
        else:
            try:
                vis_corrected = calib_pipeline(t1, overwrite=overwrite_for_run, doimport=doimport,
                                               workdir=subdir, clearcache=False, pols=pols, version=version, ncpu=ncpu,
                                               caltype=caltype, interp=interp,
                                               force_imaging_rerun=smart_cal_check and is_wsclean_version,
                                               cal_npz=cal_npz, cal_tag=cal_tag, refcal_npz_mode=refcal_npz_mode,
                                               secondary_npz=secondary_npz,
                                               fine_spectral_imaging=fine_spectral_imaging,
                                               fine_spectral_only=fine_spectral_only,
                                               custom_spws=custom_spws,
                                               force_lo_hi_smooth_extrap=force_lo_hi_smooth_extrap,
                                               refcal_sql_mode=refcal_sql_mode,
                                               sql_cal_time=sql_cal_time_for_run,
                                               force_feature_selfcal=force_feature_selfcal,
                                               imaging_only=imaging_only,
                                               refcal_provenance=refcal_provenance_records,
                                               end_time=end_time,
                                               adaptive_s00=adaptive_s00,
                                               input_ms=input_ms)
            except Exception as e:
                print(f'error in processing {datestr}. Error message: {e}')
                print(traceback.format_exc())
                failed_dates.append(datestr)
                if smart_cal_check and is_wsclean_version:
                    synoptic_info = summarize_synoptic_outputs(t1, version=version, fits_tag=fits_tag)
                    write_pipeline_status(
                        statusfile,
                        'failed',
                        date=datestr,
                        fits_count=synoptic_info['fits_count'],
                        fits_expected_count=synoptic_info['fits_expected_count'],
                        fitsfiles=synoptic_info['existing_fitsfiles'],
                        error=str(e),
                        **smart_cal_status_fields(
                            readiness,
                            fallback_calibration=fallback_calibration,
                            sql_cal_time=sql_cal_time_for_run,
                            message=run_message,
                            ran_after_deadline=run_state == 'running_after_deadline',
                            needs_same_day_calibration_rerun=fallback_calibration is not None,
                        ),
                    )
                continue
        if smart_cal_check and is_wsclean_version:
            synoptic_info = summarize_synoptic_outputs(t1, version=version, fits_tag=fits_tag)
            outputvis_root = os.path.join(
                udbmsslfcaleddir,
                t1.datetime.strftime('%Y%m'),
                t1.datetime.strftime('UDB%Y%m%d') + f'.{get_pipeline_storage_version(version)}.ms'
            )
            outputvis_exists = os.path.exists(outputvis_root) or os.path.exists(f'{outputvis_root}.tar.gz')
            try:
                provenance_classification = classify_synoptic_refcal_provenance(
                    refcal_provenance_records,
                    datestr,
                    fits_complete=synoptic_info['fits_complete'],
                )
            except ValueError as exc:
                provenance_error = str(exc)
                print(f'Failed to validate applied refcal provenance for {datestr}: {provenance_error}')
                failed_dates.append(datestr)
                status_extra = smart_cal_status_fields(
                    readiness,
                    fallback_calibration=fallback_calibration,
                    sql_cal_time=sql_cal_time_for_run,
                    message=run_message,
                    ran_after_deadline=run_state == 'running_after_deadline',
                    needs_same_day_calibration_rerun=fallback_calibration is not None,
                )
                status_extra.update({
                    'refcal_provenance_error': provenance_error,
                    'refcal_provenance_records': refcal_provenance_records or [],
                })
                write_pipeline_status(
                    statusfile,
                    'failed',
                    date=datestr,
                    fits_count=synoptic_info['fits_count'],
                    fits_expected_count=synoptic_info['fits_expected_count'],
                    fitsfiles=synoptic_info['existing_fitsfiles'],
                    outputvis_exists=outputvis_exists,
                    pipeline_result_type=type(vis_corrected).__name__,
                    **status_extra,
                )
                continue

            calibration_warning_updates = set_synoptic_calibration_warning(
                synoptic_info['existing_fitsfiles'],
                calibration_date=provenance_classification['warning_calibration_date'],
            )
            state = provenance_classification['state']
            final_message = provenance_classification['message'] or run_message
            status_extra = smart_cal_status_fields(
                readiness,
                fallback_calibration=fallback_calibration,
                sql_cal_time=sql_cal_time_for_run,
                message=final_message,
                ran_after_deadline=run_state == 'running_after_deadline',
                needs_same_day_calibration_rerun=(
                    provenance_classification['needs_same_day_calibration_rerun']
                ),
            )
            status_extra.update({
                key: value
                for key, value in provenance_classification.items()
                if key not in ('state', 'message', 'warning_calibration_date')
            })
            if provenance_classification['using_fallback_calibration']:
                status_extra.update({
                    'fallback_calibration_date': provenance_classification['applied_refcal_date_utc'],
                    'fallback_refcal_timestamp_utc': provenance_classification['applied_refcal_time_utc'],
                })
            state, imaging_qa = apply_s00_qa_to_pipeline_state(
                state,
                synoptic_info['existing_fitsfiles'],
                require_qa=True,
            )
            status_extra.update(imaging_qa)
            calibration_warning_expected = len(synoptic_info['existing_fitsfiles'])
            if calibration_warning_updates != calibration_warning_expected:
                warning_error = (
                    f'updated calibration warning metadata for {calibration_warning_updates} '
                    f'of {calibration_warning_expected} FITS files'
                )
                print(f'Failed to finalize calibration warning metadata for {datestr}: {warning_error}')
                failed_dates.append(datestr)
                write_pipeline_status(
                    statusfile,
                    'failed',
                    date=datestr,
                    fits_count=synoptic_info['fits_count'],
                    fits_expected_count=synoptic_info['fits_expected_count'],
                    fitsfiles=synoptic_info['existing_fitsfiles'],
                    outputvis_exists=outputvis_exists,
                    calibration_warning_fits_count=calibration_warning_updates,
                    calibration_warning_fits_expected_count=calibration_warning_expected,
                    pipeline_result_type=type(vis_corrected).__name__,
                    error=warning_error,
                    **status_extra,
                )
                continue
            write_pipeline_status(
                statusfile,
                state,
                date=datestr,
                fits_count=synoptic_info['fits_count'],
                fits_expected_count=synoptic_info['fits_expected_count'],
                fitsfiles=synoptic_info['existing_fitsfiles'],
                outputvis_exists=outputvis_exists,
                calibration_warning_fits_count=calibration_warning_updates,
                pipeline_result_type=type(vis_corrected).__name__,
                **status_extra,
            )
        if clearcache:
            os.chdir(workdir)
            if path_is_within(input_ms, subdir):
                print(f'Preserving explicit input_ms while clearing cache in {subdir}.')
                clear_directory_preserving_path(subdir, input_ms)
            else:
                os.system('rm -rf {}'.format(subdir))

    if failed_dates:
        print('Pipeline finished with failures for {0} date(s): {1}'.format(
            len(failed_dates), ', '.join(failed_dates)))
    return {'failed_dates': failed_dates}


if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser(description='EOVSA Pipeline for importing and calibrating visibility data.')
    # Default date is set to one day before the current date, formatted as YYYY-MM-DDT20:00
    default_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT20:00')
    parser.add_argument('--date', type=str, default=default_date,
                        help='Date to process in YYYY-MM-DDT20:00 format, defaults to 20:00 UT of one day before the current date.')
    parser.add_argument('--clearcache', action='store_true', default=False,
                        help='Remove temporary files after processing')
    parser.add_argument('--ndays', type=int, default=1,
                        help='Process data from DATE_IN_YY_MM_DD-ndays to DATE_IN_YY_MM_DD, default is 1.')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing processed data')
    import_group = parser.add_mutually_exclusive_group()
    import_group.add_argument('--doimport', dest='doimport', action='store_true',
                              help='Discover/import raw scans before calibration')
    import_group.add_argument('--skip-import', '--no-import', dest='doimport', action='store_false',
                              help='Skip raw-data import (default; explicit for resume commands)')
    parser.set_defaults(doimport=False)
    parser.add_argument('--input-ms', type=str, default=None,
                        help='Exact existing .ms directory or .ms.tar.gz archive to reuse. This bypasses '
                             'discovery, import, and calibeovsa. Without --imaging-only it is treated as a '
                             'calibrated MS and proceeds through selfcal plus imaging; with --imaging-only '
                             'it must be a selfcal product and only final imaging is run.')
    parser.add_argument('--pols', type=str, default='XX', choices=['XX', 'YY', 'XXYY'],
                        help='Polarizations to process')
    parser.add_argument('--ncpu', type=str, default='auto', help='Number of CPUs to use for processing')
    parser.add_argument('--version', type=str, default='v2.0', choices=SUPPORTED_PIPELINE_VERSIONS,
                        help='Version of the EOVSA pipeline to use')
    parser.add_argument('--debugging', action='store_true', default=False, help='Run the pipeline in debugging mode')
    parser.add_argument('--caltype', type=str, nargs='+', default=['refpha', 'phacal'],
                        help='Calibration types to use, defaults to refpha and phacal')
    parser.add_argument('--interp', type=str, default='nearest', choices=['nearest', 'linear', 'auto'],
                        help='Interpolation method for calibration tables. Options: "nearest", "linear", "auto". '
                             'If "auto" is selected, calibeovsa automatically chooses the method: '
                             'if the time difference between the observation and the phacal calibrations is less than 1 hour, it uses "nearest"; '
                             'otherwise, it uses "linear".')
    parser.add_argument('--smart-cal-check', action='store_true', default=False,
                        help='For cron-style runs, use same-day observer-written refcal/phacal records when ready. '
                             'If same-day calibration is not ready but a previous observing day has ready SQL '
                             'calibration, run provisionally with that older calibration and mark the status for '
                             'same-day replacement later. Also track per-day status and allow imaging reruns when '
                             'outputvis exists but daily FITS are incomplete.')
    parser.add_argument('--cal-npz', type=str, default=None,
                        help='Path to a calwidget_v2 calibeovsa NPZ '
                             '(e.g. /common/webplots/phasecal/YYYYMMDD_calwidget_v2_calibeovsa.npz). '
                             'When provided, calibration is read from the NPZ instead of MySQL via '
                             'suncasa.suncasatasks.private.task_calibeovsa, outputs are tagged with '
                             '--cal-tag (required for cal-npz runs) '
                             'so they do not collide with production artefacts, '
                             'and --smart-cal-check is '
                             'force-disabled because MySQL-readiness gating does not apply.')
    parser.add_argument('--secondary-npz', type=str, default=None,
                        help='Optional secondary calwidget_v2 calibeovsa NPZ. In bph_sbd runs, '
                             'finite/unflagged secondary BPH fills missing primary BPH slots; '
                             'primary SBD remains authoritative.')
    parser.add_argument('--cal-tag', type=str, default=None,
                        help='Tag for cal-npz test outputs. Required for cal-npz runs. '
                             'Used for both FITS and MS products.')
    parser.add_argument('--refcal-npz-mode', type=str, default='smooth_model',
                        choices=['triplet', 'smooth_model', 'bph_sbd', 'smooth_bandpass'],
                        help='Refcal apply mode for calwidget_v2 NPZ runs. '
                             'triplet preserves ph+sbd+mbd; smooth_model uses sampled smooth phase plus sbd only; '
                             'bph_sbd uses saved band phase plus sbd only; '
                             'smooth_bandpass applies the per-channel smooth phase as a phase-only B table '
                             '(subsumes the refcal sbd, no separate ph/sbd tables).')
    parser.add_argument('--refcal-sql-mode', type=str, default='bph_sbd',
                        choices=['auto', 'bph_sbd', 'smb', 'legacy'],
                        help='Refcal apply mode for the no-NPZ SQL path. '
                             'bph_sbd (default) requires a complete digest-matched caltype-14+16 '
                             'BPH+SBD+BPS family; auto permits fallback to fresh caltype-15 SMB '
                             'and then legacy type-8 BPH + phacal MBD; '
                             'smb applies the caltype-15 smooth phase bandpass as a per-channel '
                             'phase-only B table; legacy uses type-8 BPH plus phacal MBD only.')
    parser.add_argument('--sql-cal-time', type=str, default=None,
                        help='Optional SQL calibration lookup timestamp override for no-NPZ SQL runs. '
                             'Cron fallback uses this internally to image a target date with an older '
                             'ready calibration day.')
    parser.add_argument('--force-lo-hi-smooth-extrap', action='store_true', default=False,
                        help='For bph_sbd calwidget_v2 NPZ runs, force LO bands to use the '
                             'HI smooth-model extrapolated phase base instead of LO BPH.')
    parser.add_argument('--fine-spectral-imaging', action='store_true', default=False,
                        help='For WSClean versions, bootstrap fine chunks from freshly imaged coarse '
                             'parents and run one phase-only refinement before fine imaging.')
    parser.add_argument('--fine-spectral-only', action='store_true', default=False,
                        help='Deprecated parentless resume mode; the request is rejected. Use '
                             '--imaging-only --fine-spectral-imaging instead.')
    parser.add_argument('--imaging-only', action='store_true', default=False,
                        help='Rerun final imaging from the existing selfcal MS archive, skipping '
                             'calibration and self-calibration; combine with --fine-spectral-imaging '
                             'for fine products. Fine resume requires a current seven-parent archive.')
    parser.add_argument('--end-time', '--time-cutoff', dest='end_time', type=str, default=None,
                        help='UTC cutoff in YYYY-MM-DDTHH:MM:SS form. Data at or after this time '
                             'are flagged on staged scan MS inputs before calibration and selfcal. '
                             'Cannot be used with --imaging-only.')
    parser.add_argument('--custom-spws', type=str, nargs='+', default=None,
                        help='For WSClean versions, override FrequencySetup SPW groupings. With '
                             '--fine-spectral-imaging, the exact 24-product plan preserves the '
                             'default seven parents and requests fine outputs.')
    parser.add_argument('--force-feature-selfcal', action='store_true', default=False,
                        help='TEST ONLY: force feature self-calibration for all processed SPW groups, '
                             'bypassing the brightness gate. Default off.')
    parser.add_argument('--adaptive-s00', action='store_true', default=False,
                        help='Opt in to adaptive central-window imaging for s00-01 only. '
                             'Default uses the historical segmented full-track route; '
                             'the opt-in is suppressed when an imaging time-range filter is supplied.')

    # Parse the arguments
    args = parser.parse_args()

    # Convert --date argument to an astropy Time object to ensure consistency in time handling
    t = Time(args.date)

    # Extract year, month, day from the --date argument
    year, month, day = t.datetime.year, t.datetime.month, t.datetime.day

    # Run the main pipeline function
    run_result = pipeline(
        year=year,
        month=month,
        day=day,
        ndays=args.ndays,
        clearcache=args.clearcache,
        overwrite=args.overwrite,
        doimport=args.doimport,
        pols=args.pols,
        version=args.version,
        ncpu=args.ncpu,
        debugging=args.debugging,
        caltype=args.caltype,
        interp=args.interp,
        smart_cal_check=args.smart_cal_check,
        cal_npz=args.cal_npz,
        cal_tag=args.cal_tag,
        refcal_npz_mode=args.refcal_npz_mode,
        secondary_npz=args.secondary_npz,
        fine_spectral_imaging=args.fine_spectral_imaging,
        fine_spectral_only=args.fine_spectral_only,
        custom_spws=args.custom_spws,
        force_lo_hi_smooth_extrap=args.force_lo_hi_smooth_extrap,
        refcal_sql_mode=args.refcal_sql_mode,
        sql_cal_time=args.sql_cal_time,
        force_feature_selfcal=args.force_feature_selfcal,
        imaging_only=args.imaging_only,
        end_time=args.end_time,
        adaptive_s00=args.adaptive_s00,
        input_ms=args.input_ms,
    )

    # Exit nonzero if any date failed so wrappers (set -e) do not treat a core
    # imaging/calibration failure as success and proceed to FITS/JP2/preview steps.
    failed_dates = run_result.get('failed_dates') if isinstance(run_result, dict) else None
    if failed_dates:
        print('ERROR: EOVSA pipeline failed for {0} date(s): {1}'.format(
            len(failed_dates), ', '.join(failed_dates)), file=sys.stderr)
        sys.exit(1)
