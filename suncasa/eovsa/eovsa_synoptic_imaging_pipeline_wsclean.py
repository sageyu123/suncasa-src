'''
Loop over SPWs once: run slfcal, disk insertion/subtraction, and imaging per SPW

This version combines all operations into a single loop over SPWs. For each SPW, it performs slfcal, inserts and subtracts the disk, and then proceeds directly to imaging before moving to the next SPW. Tasks are handled sequentially within each SPW iteration.

EOVSA/WSClean compatibility note:
Use the legacy WSClean wrapper instead of a bare system `wsclean` path.
The working stack on the pipeline server is the compatibility wrapper
`/usr/local/bin/wsclean-eovsa`, which calls the legacy
`/usr/local/bin/wsclean-eovsa-bin` binary with the older runtime libraries it
needs and depends on `casacore2`:
`libcfitsio.so.5`, `libmpi.so.20`, and `libhwloc.so.5`.

This matters because newer WSClean builds can fail on EOVSA MS files
with uneven SPWs, typically with:

    ArrayColumn::get: Table array conformance error

That error is the reason this pipeline should stay pinned to the known-
working wrapper unless the full runtime stack has been revalidated.
'''
import argparse
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tarfile
import traceback
from datetime import datetime, time, timedelta
from contextlib import contextmanager
from glob import glob
from functools import wraps
import matplotlib.pyplot as plt
import pandas as pd
from eovsapy.util import Time
from eovsapy.spw_config import SPWS_34BAND, SPWS_52BAND, SPW_EPOCH_SPLIT_MJD
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from sunpy import map as smap
from typing import NamedTuple, List, Union
from scipy.signal import fftconvolve
import numbers
from suncasa.casa_compat import import_casatasks
from suncasa.io import ndfits
from suncasa.utils import helioimage2fits as hf
from suncasa.utils import mstools as mstl
from suncasa.eovsa import wrap_wsclean as ww
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface, sun
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from scipy import constants
from suncasa.eovsa import s00_imaging_policy as s00qa
from suncasa.eovsa.update_log import EOVSA15_UPGRADE_DATE, DCM_IF_FILTER_UPGRADE_DATE

hostname = socket.gethostname()
is_on_server = hostname in ['pipeline', 'inti.hpcnet.campus.njit.edu']
is_on_inti = hostname == 'inti.hpcnet.campus.njit.edu'
script_mode = True

logging.basicConfig(level=logging.INFO, format='EOVSA pipeline: [%(levelname)s] - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

tasks = import_casatasks('gaincal', 'applycal', 'clearcal', 'delmod', 'ft', 'uvsub', 'split', 'concat', 'flagmanager',
                         'flagdata', 'tclean', 'hanningsmooth', 'imhead')
gaincal = tasks.get('gaincal')
applycal = tasks.get('applycal')
clearcal = tasks.get('clearcal')
delmod = tasks.get('delmod')
ft = tasks.get('ft')
uvsub = tasks.get('uvsub')
split = tasks.get('split')
concat = tasks.get('concat')
flagmanager = tasks.get('flagmanager')
flagdata = tasks.get('flagdata')
tclean = tasks.get('tclean')
hanningsmooth = tasks.get('hanningsmooth')
imhead = tasks.get('imhead')

from suncasa.casa_compat import import_casatools

tools = import_casatools(['qatool', 'iatool', 'cltool', 'mstool', 'tbtool'])
qatool = tools['qatool']
iatool = tools['iatool']
cltool = tools['cltool']
mstool = tools['mstool']
tbtool = tools['tbtool']
qa = qatool()
ia = iatool()
ms = mstool()
tb = tbtool()


def _resolve_wsclean_bin():
    """Return the preferred WSClean executable path for EOVSA.

    Prefer the EOVSA compatibility wrapper so the pipeline stays on the
    casacore2-backed binary that works with the legacy runtime stack.
    """
    preferred_wrapper = '/usr/local/bin/wsclean-eovsa'
    if os.path.isfile(preferred_wrapper) and os.access(preferred_wrapper, os.X_OK):
        return preferred_wrapper

    resolved = shutil.which('wsclean-eovsa')
    if resolved:
        return resolved

    resolved = shutil.which('wsclean')
    if resolved:
        return resolved
    return 'wsclean'


def _remove_existing_path_or_raise(path, description):
    """Remove a replacement destination and fail closed if cleanup fails.

    ``shutil.move`` treats an existing directory destination as a container
    and moves the source inside it.  That is unsafe for generated CASA
    sidecars such as ``.flagversions``: a failed cleanup would therefore
    produce a nested directory and a less useful permission error later.

    :param path: Destination path that must be absent before a replacement.
    :type path: str
    :param description: Human-readable description of the destination.
    :type description: str
    :raises PermissionError: If the existing destination cannot be removed.
    :raises RuntimeError: If the destination remains after cleanup.
    """
    if not os.path.lexists(path):
        return

    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except OSError as exc:
        raise PermissionError(
            f"Unable to replace existing {description} {path}; "
            "remove it or grant write permission to its parent directory."
        ) from exc

    if os.path.lexists(path):
        raise RuntimeError(
            f"Unable to replace existing {description} {path}: "
            "the destination still exists after cleanup."
        )


WSCLEAN_BIN = _resolve_wsclean_bin()
# ``WSClean`` owns its own command string, so keep the imported wrapper in
# lockstep with the path selected here.  Without this assignment the two
# modules can log different binaries and the wrapper can silently fall back
# to a missing ``wsclean-eovsa`` command.
ww.WSCLEAN_BIN = WSCLEAN_BIN
logging.info("Using wsclean executable: %s", WSCLEAN_BIN)

# ============================================================
# Pipeline Configuration Constants
# ============================================================

FINE_SPECTRAL_SPWS_52BAND = [
    '0~1', '2', '3', '4', '5~6', '7~8', '9~10',
    '11~12', '13~14', '15~16', '17~18', '19~20',
    '21~22', '23~24', '25~26', '27~28', '29~30',
    '31~33', '34~35', '36~37', '38~39', '40~41', '42~43',
    '44~49',
]
FINE_BOOTSTRAP_PARENT_MARKER = '.eovsa_52band_parent_checkpoints_v1'

PIPELINE_CONFIG = {
    # bright_thresh anchor for s31-43 (13.92 GHz) lowered 150->50 on 2026-07-03:
    # a forced-feature-selfcal A/B on 2026-04-03 showed feature selfcal doubles
    # the s31-43 coarse DR (195->446, reproduced twice at DR 462/464). Anchor set
    # from the per-date score distribution mined from 65 obs dates of pipeline
    # logs: s31-43 scores are bimodal (faint days <=25, source days >=75; 50
    # sits in the gap).
    # s44-49 (17.01 GHz) anchor was provisionally lowered 100->12 the same day
    # but REVERTED after the validation rerun: at s44-49's typical SNR (~10)
    # feature selfcal is unstable — the forced run converged (fringe removed,
    # clean field) but the identical no-force rerun diverged catastrophically
    # (700 MK blotches, 792 px > 100 MK). Do not re-lower without a post-selfcal
    # divergence guard (e.g. reject solutions if the post-round image peak or
    # off-source noise explodes vs the pre-selfcal image).
    'bright_thresh': [350, 900, 2000, 2800, 800, 50, 100],
    'bright_thresh_freq_ghz': [1.42, 2.87, 4.33, 6.93, 10.18, 13.92, 17.01],
    'tb_ratio_thresh': [1.5, 5.0, 5.0, 3.0, 1.5, 1.5, 1.5],
    'briggs': [-1.0, -1.0, -1.0, -1.0, -1.0, -0.5, -0.5],
    'spwidx2proc': [0, 1, 2, 3, 4, 5, 6],
    # --- Fine-spectral quality gate ---
    # At high frequency (faint Sun) the ~2-SPW fine-spectral sub-chunks can
    # have too little SNR to converge, producing off-source noise/peaks far
    # above the parent coarse-group image. When enabled, each fine-chunk
    # product is compared against its parent coarse-group product and
    # rejected (FITS deleted) if it looks diverged; the coarse product
    # already covers that frequency range, so rejection has no coverage gap.
    'fine_spectral_snr_gate': True,
    'fine_spectral_noise_factor': 3.5,
    'fine_spectral_dr_floor': 70.0,
    'fine_spectral_peak_tb_max': 2.0e6,   # fallback only (no coarse parent for the interval)
    # Primary peak criterion: relative to the coarse parent's peak in the same
    # interval. Real fine-chunk sources track the group MFS peak (spectral slope
    # ~x2 max); divergence blows up >~x10. Absolute ceilings misfire on active
    # days (2026-07-01: 2-10 MK sources with DR 80-200 were mass-rejected).
    'fine_spectral_peak_ratio_max': 3.0,
    # --- Joint fine-spectral imaging ---
    'fine_spectral_joint': True,        # one joint-deconvolution wsclean per group instead of per-chunk runs
    'fine_spectral_fit_spectral_pol': 2,
    'fine_spectral_exclude': ['44~49'], # no fine products for these coarse groups (too faint; owner decision)
    # The joint run's per-channel circular beams are fit per-channel (an accepted
    # simplification for deconvolution uv-coverage), but that leaves each channel
    # with an erratic, wsclean-chosen beam size (e.g. 19.0/14.1/11.4 arcsec across
    # intervals of the *same* chunk) instead of the fixed table beam the legacy
    # per-chunk convention used for Tb bookkeeping (Tb ~ 1/beam_area). Re-restore
    # each per-(interval, channel) image at its chunk's table fine_bmsize (arcsec,
    # forced circular) using `wsclean -beam-size <bmsize> -circular-beam -restore
    # <residual> <model> <output>` before registration, so Tb conversion matches
    # the legacy per-chunk convention exactly.
    'fine_spectral_force_table_beam': True,
}


def feature_brightness_threshold(freq_ghz):
    """Evaluate the smooth feature-routing brightness threshold.

    The seven production spectral-window thresholds are used as anchor points
    in frequency.  The interpolation is done in log-frequency/log-threshold
    space with a shape-preserving cubic Hermite interpolator.  Frequencies
    outside the anchored range are clamped to the nearest endpoint because the
    production thresholds were not validated as an extrapolation.

    :param freq_ghz: Frequency or frequencies at which to evaluate the threshold.
    :type freq_ghz: float or array-like
    :returns: Feature-routing threshold value(s) in the same metric units as
        ``M_bright``.
    :rtype: float or numpy.ndarray
    :raises ValueError: If any input frequency is non-finite or non-positive.
    """
    freq_arr = np.asarray(freq_ghz, dtype=float)
    scalar_input = freq_arr.ndim == 0
    if np.any(~np.isfinite(freq_arr)) or np.any(freq_arr <= 0):
        raise ValueError("freq_ghz must contain finite positive frequencies")

    anchor_freq = np.asarray(PIPELINE_CONFIG['bright_thresh_freq_ghz'], dtype=float)
    anchor_thresh = np.asarray(PIPELINE_CONFIG['bright_thresh'], dtype=float)
    log_freq = np.log(anchor_freq)
    log_thresh = np.log(anchor_thresh)
    interp = PchipInterpolator(log_freq, log_thresh, extrapolate=False)

    query = np.clip(np.log(freq_arr), log_freq[0], log_freq[-1])
    thresh = np.exp(interp(query))
    if scalar_input:
        return float(thresh)
    return thresh

# Self-calibration round definitions (drives the loop in pipeline_run)
SELFCAL_ROUNDS = [
    {
        'name': 'init',
        'interval_multipliers': {0: 20, 1: 20, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3},
        'niter': 100,
        'auto_mask': 6,
        'auto_threshold': 3,
        'briggs': 0.0,
        'data_column': 'DATA',
        'circular_beam': False,
        'gaincal_steps': [
            {'solint': 'inf', 'calmode': 'p', 'suffix': 'inf', 'refantmode': 'flex'},
            {'solint': 'auto_minor', 'calmode': 'p', 'suffix': 'int', 'refantmode': 'flex'},
        ],
        'interp': 'nearest',
    },
    {
        'name': 'round1',
        'interval_multipliers': {0: 6, 1: 3, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
        'niter': 200,
        'auto_mask': 4,
        'auto_threshold': 2,
        'briggs': 0.0,
        'data_column': None,  # determined at runtime (CORRECTED_DATA or DATA)
        'circular_beam': False,
        'gaincal_steps': [
            {'solint': 'auto_minor', 'calmode': 'p', 'suffix': 'int', 'refantmode': 'flex'},
        ],
        'interp': 'linear',
    },
    {
        'name': 'round2',
        'interval_multipliers': {0: 4, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
        'niter': 500,
        'auto_mask': 3,
        'auto_threshold': 1,
        'briggs': 0.0,
        'data_column': 'CORRECTED_DATA',
        'circular_beam': False,
        'gaincal_steps': [
            {'solint': 'auto_minor', 'calmode': 'p', 'suffix': 'int', 'refantmode': 'flex'},
            {'solint': 'inf', 'calmode': 'a', 'suffix': 'inf', 'refantmode': 'flex',
             'uvrange_key': 'uvmin'},
        ],
        'interp': 'linear',
    },
]

FINAL_IMAGING_CONFIG = {
    'interval_multipliers': {0: 2.4, 1: 1.2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
}

S00_FINAL_CONFIG = {
    # July 22 tests support a central 5 h product with 4 h as the safer
    # fallback.  Every date still has to pass its own coverage, PSF, and
    # differential-rotation gates before either duration is selected.
    'candidate_hours': (5.0, 4.0, 3.0),
    'psf_sidelobe_max': 0.15,
    'smear_beam_max': 1.0,
    # Coverage is a coarse guard, not a substitute for the PSF measured from
    # the actual sampled visibilities.  The July 22 benchmark contains one
    # 20.5-minute gap yet its 5 h PSF is 11.7%; admit that case while still
    # rejecting windows missing more than 15% of their nominal samples or a
    # contiguous half hour at the normal one-minute cadence.
    'min_coverage_fraction': 0.85,
    'max_gap_factor': 30.0,
    # The long-track init model is deliberately shallow.  Retain only
    # repeatable positive support, then dilate far enough to cover one beam
    # and the permitted source motion without opening the full solar disk.
    'support_sigma': 7.0,
    'support_peak_fraction': 0.02,
    'support_dilation_beams': 3.0,
    # Residual-mask retry gates.  A single full-image residual can only
    # trigger held-out tests; it can never authorize its own mask expansion.
    'residual_snr_trigger': 7.0,
    'residual_peak_ratio_trigger': 0.10,
    'half_snr_min': 5.0,
    'centroid_beam_fraction': 0.5,
    'max_mask_growth_fraction': 0.10,
    'max_added_beams_per_candidate': 3.0,
    'max_candidate_islands': 2,
    'max_mask_image_fraction': 0.25,
    'qa_clean_niter': 4000,
    'retry_residual_drop_min': 0.50,
    'retry_energy_drop_min': 0.20,
    'retry_rms_worsen_max': 0.05,
    'retry_source_flux_change_max': 0.10,
}

UV_CONFIG = {
    'uvmax_baseline_m': 15,
    'uvmin_baseline_m': 110,
    'uvmin_baseline_m_band0': 55,
    'uvmax_l_clamp': (100, 1200),
    'uvmin_l_clamp_max': 1800,
}


def log_print(level, message):
    # Get the caller's stack frame and extract information
    frame = inspect.stack()[1]
    module_name = inspect.getmodulename(frame[1])
    function_name = frame[3]

    # Format the custom context information
    context = f"{module_name}::{function_name}"

    # Prepare the full log message
    full_message = f"{context} - {message}"

    # Fetch the logger
    logger = logging.getLogger()

    # Log the message with the specified level
    if level.upper() == 'WARNING':
        logger.warning(full_message)
    elif level.upper() == 'INFO':
        logger.info(full_message)
    elif level.upper() == 'DEBUG':
        logger.debug(full_message)
    elif level.upper() == 'ERROR':
        logger.error(full_message)
    elif level.upper() == 'CRITICAL':
        logger.critical(full_message)
    else:
        logger.info(full_message)  # Default to INFO if an unsupported level is given


def _format_log_state(**state):
    """Format a stable key=value summary for diagnostic logging."""
    parts = []
    for key in sorted(state):
        value = state[key]
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "no-state"


def log_step(level, step, message, **state):
    """Emit a log line with an explicit pipeline step and state summary."""
    log_print(level, f"[{step}] {message} | {_format_log_state(**state)}")


def _write_compressed_synoptic_fits(src_fits, dst_fits, overwrite=True):
    """Write a tiled-compressed FITS copy for final synoptic products."""
    dst_tmp = dst_fits
    inplace = os.path.abspath(src_fits) == os.path.abspath(dst_fits)
    if inplace:
        dst_tmp = f"{dst_fits}.tmp_compressed"

    with fits.open(src_fits, memmap=False) as hdul:
        if any(isinstance(hdu, fits.hdu.CompImageHDU) for hdu in hdul):
            if not inplace or src_fits != dst_tmp:
                shutil.copy2(src_fits, dst_tmp)
        else:
            compressed_hdus = []
            for hdu_idx, hdu in enumerate(hdul):
                is_image_hdu = isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU))
                if hdu_idx == 0:
                    if hdu.data is None:
                        compressed_hdus.append(fits.PrimaryHDU(header=hdu.header.copy()))
                    else:
                        compressed_hdus.append(fits.PrimaryHDU())

                if is_image_hdu and hdu.data is not None:
                    data = np.array(hdu.data)
                    nonsingleton_ndim = data.ndim - np.count_nonzero(np.array(data.shape) == 1)
                    if nonsingleton_ndim > 3:
                        log_print('WARNING',
                                  f"Skipping FITS compression for {src_fits}: "
                                  f"image has {nonsingleton_ndim} non-singleton dimensions")
                        if not inplace:
                            shutil.copy2(src_fits, dst_fits)
                        return dst_fits
                    header, data = ndfits.headersqueeze(hdu.header.copy(), data)
                    compressed_hdus.append(fits.CompImageHDU(
                        data=data, header=header, compression_type='RICE_1', quantize_level=4.0))
                elif hdu_idx > 0:
                    compressed_hdus.append(hdu.copy())

            fits.HDUList(compressed_hdus).writeto(dst_tmp, overwrite=overwrite, output_verify='fix')

    if inplace:
        os.replace(dst_tmp, dst_fits)
    return dst_fits


def _compress_synoptic_fits_in_place(fitsfiles):
    """Compress final synoptic FITS files after all WSClean-facing steps are done."""
    files = [fitsfiles] if isinstance(fitsfiles, str) else list(fitsfiles)
    for fitsfile in files:
        if fitsfile and os.path.exists(fitsfile):
            _write_compressed_synoptic_fits(fitsfile, fitsfile)


def _set_daily_reference_time_keywords(fitsfile, reftime):
    """Stamp DATE-OBS/DATE-AVG with the daily reference epoch in place.

    The synoptic_daily image content is aligned to the daily reference time,
    so DATE-OBS must match it for time-based coalignment. The true acquisition
    window is preserved in STARTOBS/ENDOBS before DATE-OBS is overwritten.
    """
    reftime_str = Time(reftime).isot
    with fits.open(fitsfile, mode='update') as hdul:
        for hdu in hdul:
            if 'CDELT1' not in hdu.header:
                continue
            hd = hdu.header
            orig = hd.get('DATE-OBS')
            if 'STARTOBS' not in hd and orig:
                hd.set('STARTOBS', Time(orig).isot, 'start of true observing window')
                exptime = hd.get('EXPTIME')
                if 'ENDOBS' not in hd and exptime:
                    hd.set('ENDOBS', (Time(orig) + float(exptime) * u.s).isot,
                           'end of true observing window')
            hd.set('DATE-OBS', reftime_str, 'daily reference epoch; see STARTOBS/ENDOBS')
            hd.set('DATE-AVG', reftime_str, 'daily reference epoch (WCS)')
            hd.add_history('DATE-OBS set to daily reference epoch; acquisition window in STARTOBS/ENDOBS')
            break
    return fitsfile


def _set_imaging_antenna_keywords(fitsfile, n_ant_img, n_ant_total, antenna_ids=None):
    """Stamp the solar antenna availability used for imaging into a FITS file."""
    if n_ant_img is None or n_ant_total is None:
        return fitsfile
    with fits.open(fitsfile, mode='update') as hdul:
        for hdu in hdul:
            if 'CDELT1' not in hdu.header:
                continue
            hdu.header.set('NANTIMG', int(n_ant_img),
                           'unflagged solar antennas available for imaging')
            hdu.header.set('NANTTOT', int(n_ant_total),
                           'total solar antennas for this observing epoch')
            if antenna_ids is not None:
                hdu.header.set('ANTUSED', ','.join(str(int(ant)) for ant in antenna_ids),
                               'one-based unflagged solar antennas used')
            break
    return fitsfile


def _write_imaging_antenna_manifest(fitsfile, n_ant_total, antenna_ids):
    """Publish exact unflagged imaging antennas next to one FITS product."""
    if n_ant_total is None or antenna_ids is None:
        return None
    manifest_path = fitsfile + '.antennas.json'
    tmp_path = manifest_path + '.tmp.{}'.format(os.getpid())
    payload = {
        'schema_version': 1,
        'source': 'unflagged_imaging_input',
        'expected': int(n_ant_total),
        'used': [int(ant) for ant in antenna_ids],
    }
    with open(tmp_path, 'w') as outfile:
        json.dump(payload, outfile, sort_keys=True)
        outfile.write('\n')
    os.replace(tmp_path, manifest_path)
    return manifest_path


def _promote_imaging_antenna_manifest(source_fitsfiles, target_fitsfile):
    """Combine current-run antenna provenance for a merged daily product.

    :param source_fitsfiles: Interval FITS files supplied to the daily merge.
    :type source_fitsfiles: list(str)
    :param target_fitsfile: Merged daily FITS receiving the sidecar.
    :type target_fitsfile: str
    :returns: Written manifest path, or ``None`` when no valid source manifest
        is available.
    :rtype: str or None
    """
    expected = None
    used = set()
    for source_fitsfile in source_fitsfiles:
        manifest_path = source_fitsfile + '.antennas.json'
        try:
            with open(manifest_path) as infile:
                manifest = json.load(infile)
            if (manifest.get('schema_version') != 1
                    or manifest.get('source') != 'unflagged_imaging_input'
                    or not isinstance(manifest.get('expected'), int)
                    or not isinstance(manifest.get('used'), list)):
                continue
            manifest_expected = manifest['expected']
            manifest_used = manifest['used']
            if (manifest_expected < 1
                    or any(not isinstance(ant, int)
                           or isinstance(ant, bool)
                           or ant < 1
                           or ant > manifest_expected
                           for ant in manifest_used)):
                continue
            if expected is None:
                expected = manifest_expected
            elif manifest_expected != expected:
                continue
            used.update(manifest_used)
        except (OSError, TypeError, ValueError):
            continue
    if expected is None:
        return None
    return _write_imaging_antenna_manifest(
        target_fitsfile, expected, sorted(used))


def _synoptic_merge_inputs(out_key, synfits_manifest, imgoutdir, date_str,
                            spwstr, fits_tag='', merge_existing=False):
    """Resolve interval FITS for one daily merge without mixing pipeline runs.

    Normal imaging uses only paths recorded by the current run.  The broad
    filesystem lookup is reserved for explicit ``mergeFITSonly`` operation,
    where no current-run imaging manifest exists.

    :param out_key: Coarse or fine product key.
    :type out_key: int or str
    :param synfits_manifest: Interval FITS paths produced by the current run.
    :type synfits_manifest: dict
    :param imgoutdir: Synoptic FITS output directory.
    :type imgoutdir: str
    :param date_str: Observing date in ``YYYYMMDD`` form.
    :type date_str: str
    :param spwstr: Formatted spectral-window range.
    :type spwstr: str
    :param fits_tag: Optional output-route tag.
    :type fits_tag: str
    :param merge_existing: Whether to discover existing interval products.
    :type merge_existing: bool
    :returns: Ordered interval FITS paths for the daily merge.
    :rtype: list(str)
    """
    if not merge_existing:
        return list(synfits_manifest.get(out_key, []))
    post_tag = f'.{fits_tag}' if fits_tag else ''
    return sorted(glob(os.path.join(
        imgoutdir,
        f"eovsa.synoptic{post_tag}.{date_str[:-1]}?T??????Z.s{spwstr}.tb.fits")))


@contextmanager
def pipeline_stage(step, **state):
    """Log start/end/failure for a pipeline stage with traceback on error."""
    start = datetime.now()
    log_step('INFO', step, "start", **state)
    try:
        yield
    except Exception:
        elapsed = (datetime.now() - start).total_seconds()
        tb = traceback.format_exc()
        log_print('ERROR', f"[{step}] failed after {elapsed:.1f}s | {_format_log_state(**state)}\n{tb}")
        raise
    else:
        elapsed = (datetime.now() - start).total_seconds()
        log_step('INFO', step, f"done in {elapsed:.1f}s", **state)


def get_solar_radius_mask(rsun_pix, crpix1, crpix2, ny, nx, rsun_ratio=(1.2, 1.3)):
    y, x = np.ogrid[:ny, :nx]
    dist = np.sqrt((x - crpix1) ** 2 + (y - crpix2) ** 2)

    mask_ring = ((dist > rsun_pix * rsun_ratio[0]) &
                 (dist < rsun_pix * rsun_ratio[1]))
    mask_disk = dist <= rsun_pix * 1.01
    return mask_ring, mask_disk


def detect_noisy_images(data_stack,
                        rsun_pix,
                        crpix1,
                        crpix2,
                        rsun_ratio=(1.2, 1.3),
                        rms_threshold=None,
                        snr_threshold=None,
                        showplt=False,
                        stripe_qa=True,
                        stripe_threshold=50.0,
                        grad_aniso_threshold=12.0,
                        extreme_rms_factor=8.0):
    """
    Detect and reject noisy frames in a stack of solar images.

    This function computes for each image:
      1. RMS noise in an annular region outside the solar disk.
      2. SNR using the 99.999th percentile inside the disk.
      3. (when `rms_threshold` is None) an automatic “knee” in the sorted RMS
         curve that separates the flat cluster from the noisy outliers.

    If the knee detector fails (elbow at first/last point, or would reject zero
    frames), we fall back to `median(rms) + std(rms)`.

    :param data_stack: 3D array of images, shape (n_images, height, width).
    :type data_stack: np.ndarray
    :param rsun_pix: Radius of the solar disk in pixels.
    :type rsun_pix: float
    :param crpix1: X-coordinate of the disk center.
    :type crpix1: float
    :param crpix2: Y-coordinate of the disk center.
    :type crpix2: float
    :param rsun_ratio: Inner and outer scale factors for the annular mask.
                        Defaults to (1.2, 1.3).
    :type rsun_ratio: tuple(float, float)
    :param rms_threshold: User-supplied RMS cutoff. If None, an automatic
                          knee detection is used.
    :type rms_threshold: float or None
    :param snr_threshold: SNR cutoff. If None, set to median(SNR) – std(SNR).
    :type snr_threshold: float or None
    :param showplt: If True, display a diagnostic plot of RMS and SNR with their cutoffs.
    :type showplt: bool
    :param stripe_qa: If True, reject coherent directional stripe/artifact frames
                      using disk-free image structure.
    :type stripe_qa: bool
    :param stripe_threshold: Absolute Fourier angular-concentration cutoff for
                             stripe rejection. Frames are also rejected on a
                             within-stack relative-outlier rule (see
                             :func:`detect_stripe_artifacts`), so this only needs
                             to catch uniformly artifact-dominated bands.
    :type stripe_threshold: float
    :param grad_aniso_threshold: Directional gradient anisotropy threshold for
                                 stripe rejection (weak secondary check).
    :type grad_aniso_threshold: float
    :param extreme_rms_factor: Reject frames whose off-disk annulus RMS is this
                               many times above the stack median, even if their
                               formal SNR is high. This catches high-band
                               stripe/artifact frames whose bright structured
                               sidelobes otherwise bypass the ordinary RMS
                               gate.
    :type extreme_rms_factor: float

    :returns:
      - `select_indices` (bool array): True for images kept (not noisy).
      - `rms_values` (float array): RMS noise for each image.
      - `snr_values` (float array): SNR for each image.
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    # Prepare masks
    images = np.asarray(data_stack, dtype=float)
    images = np.squeeze(images)
    if images.ndim == 2:
        images = images[np.newaxis, :, :]
    ny, nx = images.shape[1], images.shape[2]
    mask_ring, mask_disk = get_solar_radius_mask(rsun_pix, crpix1, crpix2, ny, nx, rsun_ratio=rsun_ratio)

    # Compute RMS and SNR for all frames (vectorized)
    ring_pixels = images[:, mask_ring]   # (n_images, n_ring_pixels)
    disk_pixels = images[:, mask_disk]   # (n_images, n_disk_pixels)
    ring_rms_values = np.sqrt(np.nanmean(ring_pixels ** 2, axis=1))
    disk_rms_values = np.sqrt(np.nanmean(disk_pixels ** 2, axis=1))
    signals = np.nanpercentile(disk_pixels, 99.999, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr_values = signals / ring_rms_values

    # --- Automatic RMS threshold (knee detection) ---
    fallback = np.nanmedian(ring_rms_values) + np.nanstd(ring_rms_values)

    if rms_threshold is None:
        sorted_rms = np.sort(ring_rms_values)
        N = len(sorted_rms)

        pts = np.vstack((np.arange(N), sorted_rms)).T
        p0, pN = pts[0], pts[-1]
        vec = pN - p0
        length = np.linalg.norm(vec)

        if N < 2 or length <= 0:
            rms_threshold = fallback
        else:
            rel = pts - p0
            proj = np.dot(rel, vec / length)
            proj_vec = np.outer(proj, vec / length)
            dists = np.linalg.norm(rel - proj_vec, axis=1)

            knee = np.argmax(dists)

            if knee in (0, N - 1):
                rms_threshold = fallback
            else:
                thr = sorted_rms[knee]
                rms_threshold = thr if np.sum(ring_rms_values > thr) > 0 else fallback

    # --- Automatic SNR threshold ---
    if snr_threshold is None:
        med_snr = np.nanmedian(snr_values)
        std_snr = np.sqrt(np.nanmean((snr_values - med_snr) ** 2))
        snr_threshold = med_snr - std_snr

    # Flag noisy: low‐SNR or high‐RMS, but ignore ordinary high‐SNR outliers.
    # A separate extreme off-disk RMS guard catches high-band artifact frames
    # with coherent bright stripes that can look high-SNR yet ruin the merge.
    snr_limited = snr_values <= 3 * snr_threshold
    rms_bad = (ring_rms_values > rms_threshold) & snr_limited
    finite_ring_rms = ring_rms_values[np.isfinite(ring_rms_values) & (ring_rms_values > 0)]
    if finite_ring_rms.size > 0:
        rms_median = np.nanmedian(finite_ring_rms)
    else:
        rms_median = np.nan
    extreme_rms_bad = np.zeros_like(rms_bad, dtype=bool)
    if np.isfinite(rms_median) and rms_median > 0 and extreme_rms_factor is not None:
        extreme_rms_bad = ring_rms_values > (float(extreme_rms_factor) * rms_median)
    snr_bad = (snr_values < snr_threshold) & snr_limited
    noisy = rms_bad | extreme_rms_bad | snr_bad

    stripe_bad = np.zeros(images.shape[0], dtype=bool)
    stripe_scores = np.ones(images.shape[0], dtype=float)
    grad_scores = np.ones(images.shape[0], dtype=float)

    if stripe_qa:
        stripe_bad, stripe_scores, grad_scores = detect_stripe_artifacts(
            images,
            stripe_threshold=stripe_threshold,
            grad_aniso_threshold=grad_aniso_threshold)
        noisy = noisy | stripe_bad
        # Log the full per-frame score distribution so the real per-band scale
        # is visible in the pipeline log and the thresholds can be calibrated.
        with np.printoptions(precision=1, suppress=True, linewidth=200):
            log_print('INFO',
                      f"Stripe QA (n={images.shape[0]}): "
                      f"stripe_scores={np.asarray(stripe_scores)}; "
                      f"median={float(np.nanmedian(stripe_scores)):.2f}; "
                      f"grad_scores={np.asarray(grad_scores)}; "
                      f"abs_thr={stripe_threshold:.3g}; grad_thr={grad_aniso_threshold:.3g}; "
                      f"n_flagged={int(np.count_nonzero(stripe_bad))}")
        if stripe_bad.size > 0 and stripe_bad.all():
            log_print('WARNING',
                      'Stripe/artifact QA: artifact dominated band/day; all frames exceeded '
                      'stripe thresholds. Keeping the 2 least-striped frames for merge stability.')

    # Ensure we keep at least two frames
    keep = ~noisy
    min_keep = min(2, keep.size)
    if keep.sum() < min_keep:
        snr_rank = np.nan_to_num(snr_values, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        if stripe_qa:
            stripe_rank = np.nan_to_num(stripe_scores, nan=np.inf, posinf=np.inf, neginf=np.inf)
            order = np.lexsort((-snr_rank, stripe_rank))
        else:
            order = np.argsort(-snr_rank)
        needed = min_keep - keep.sum()
        candidates = order[~keep[order]]
        keep[candidates[:needed]] = True

    for idx in np.where(~keep)[0]:
        reasons = []
        if rms_bad[idx]:
            reasons.append('RMS')
        if extreme_rms_bad[idx]:
            reasons.append('EXTREME_RMS')
        if snr_bad[idx]:
            reasons.append('SNR')
        if stripe_bad[idx]:
            reasons.append('STRIPE')
        reason_text = '+'.join(reasons) if reasons else 'UNKNOWN'
        log_print('INFO',
                  f"Rejected image index {idx}: reason={reason_text}; "
                  f"stripe_score={stripe_scores[idx]:.3g}; "
                  f"grad_aniso={grad_scores[idx]:.3g}; "
                  f"stripe_thr={stripe_threshold:.3g}; "
                  f"grad_thr={grad_aniso_threshold:.3g}")

    # Diagnostic plots: RMS on top, SNR below
    if showplt:
        fig, (ax_rms, ax_snr, ax_ring_disk) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

        ax_rms.plot(ring_rms_values, 'bo-', label='RMS Noise')
        ax_rms.axhline(rms_threshold, color='r', ls='--',
                       label=f'RMS thr = {rms_threshold:.1f}')
        ax_rms.set_ylabel('RMS')
        ax_rms.set_title('Diagnostics: RMS & SNR')
        ax_rms.legend()

        ax_snr.plot(snr_values, 'go-', label='SNR')
        ax_snr.axhline(snr_threshold, color='r', ls='--',
                       label=f'SNR thr = {snr_threshold:.1f}')
        ax_snr.set_xlabel('Image Index')
        ax_snr.set_ylabel('SNR')
        ax_snr.legend()

        ax_ring_disk.plot(ring_rms_values, 'ro-', label='Ring RMS')
        ax_ring_disk.plot(disk_rms_values, 'mo-', label='Disk RMS')
        ax_ring_disk.set_ylabel('RMS Value')
        ax_ring_disk.set_xlabel('Image Index')
        ax_ring_disk.legend()

        plt.tight_layout()
        plt.show()

    return keep, ring_rms_values, snr_values


def detect_stripe_artifacts(data_stack,
                            stripe_threshold=50.0,
                            grad_aniso_threshold=12.0,
                            stripe_rel_factor=8.0,
                            stripe_rel_floor=30.0,
                            bg_sigma=8.0,
                            n_angle_bins=45,
                            freq_inner_frac=0.06,
                            freq_outer_frac=0.95):
    """
    Detect coherent directional stripe/artifact frames in a stack of
    DISK-FREE solar images (pre-disk-addition synoptic arrays).

    Band-agnostic: every metric is a scale-invariant ratio, so it works for
    low, mid, and high band image scales without a solar-disk model.  Each
    frame is high-pass filtered by subtracting a broad Gaussian background.
    Frames with negligible high-pass residual structure retain neutral scores
    to avoid classifying smooth radial emission as a stripe.  Remaining
    residuals are apodized with a two-dimensional Hann window and scored by
    the angular concentration of Fourier power in a mid-frequency annulus.
    A companion directional-gradient ratio catches strong stripe-like
    residuals in the image domain.

    A frame is flagged when its Fourier stripe score clears EITHER an absolute
    backstop (``stripe_threshold``, which catches uniformly artifact-dominated
    bands such as the highest EOVSA spectral windows) OR a per-stack relative
    cut (``stripe_rel_factor`` x the stack median, gated by ``stripe_rel_floor``
    so an ordinary band of mildly anisotropic frames is never trimmed). Genuine
    coherent stripes score one to several orders of magnitude above ordinary
    dirty-beam anisotropy, so this two-sided rule rejects ruined frames while
    leaving clean low/mid-band frames untouched. The gradient ratio is a weak
    secondary check held at a high threshold to avoid false positives.

    :param data_stack: Stack of disk-free image frames. Accepted shapes include
                       ``(n_images, ny, nx)``, ``(n_images, 1, ny, nx)``, and a
                       single ``(ny, nx)`` frame.
    :type data_stack: numpy.ndarray
    :param stripe_threshold: Absolute Fourier angular concentration ratio above
                             which a frame is flagged regardless of the rest of
                             the stack.
    :type stripe_threshold: float
    :param grad_aniso_threshold: Directional gradient anisotropy ratio above
                                 which a frame is flagged.
    :type grad_aniso_threshold: float
    :param stripe_rel_factor: Multiple of the stack-median stripe score above
                              which a frame is treated as a within-band outlier.
    :type stripe_rel_factor: float
    :param stripe_rel_floor: Minimum absolute stripe score for the relative
                             outlier path to fire, so faint uniform anisotropy
                             never trips it.
    :type stripe_rel_floor: float
    :param bg_sigma: Gaussian sigma, in pixels, for the broad background
                     subtraction.
    :type bg_sigma: float
    :param n_angle_bins: Number of folded Fourier-angle bins over ``[0, pi)``.
                         Kept deliberately coarse so a handful of compact bright
                         sources (active regions) average out and do not mimic a
                         stripe, while a single coherent stripe direction still
                         dominates one bin.
    :type n_angle_bins: int
    :param freq_inner_frac: Inner normalized radius of the Fourier annulus.
    :type freq_inner_frac: float
    :param freq_outer_frac: Outer normalized radius of the Fourier annulus.
    :type freq_outer_frac: float

    :returns: Tuple containing the boolean stripe mask, Fourier stripe scores,
              and gradient anisotropy scores.  A True stripe mask value means
              the frame cleared the absolute or relative stripe cut, or the
              gradient anisotropy threshold.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    stack = np.asarray(data_stack)
    if stack.ndim == 2:
        frames = stack[np.newaxis, :, :]
    else:
        frames = stack

    n_images = frames.shape[0]
    stripe_scores = np.ones(n_images, dtype=float)
    grad_aniso_scores = np.ones(n_images, dtype=float)
    n_bins = max(1, int(n_angle_bins))

    for idx, frame in enumerate(frames):
        img = np.asarray(np.squeeze(frame), dtype=float)
        if img.ndim != 2:
            continue

        img = np.where(np.isfinite(img), img, 0.0)
        ny, nx = img.shape
        if ny == 0 or nx == 0:
            continue

        resid = img - ndimage.gaussian_filter(img, bg_sigma)
        img_std = np.std(img)
        resid_std = np.std(resid)
        if resid_std <= 0 or not np.isfinite(resid_std):
            continue
        if img_std > 0 and resid_std / img_std < 0.2:
            continue

        window = np.outer(np.hanning(ny), np.hanning(nx))
        resid_w = resid * window
        power = np.abs(np.fft.fftshift(np.fft.fft2(resid_w))) ** 2

        ky = np.arange(ny) - ny // 2
        kx = np.arange(nx) - nx // 2
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        half_short_axis = max(min(ny, nx) / 2.0, 1.0)
        r_norm = np.sqrt(kx_grid ** 2 + ky_grid ** 2) / half_short_axis
        annulus = (r_norm > freq_inner_frac) & (r_norm < freq_outer_frac)

        annulus_power = power[annulus]
        if annulus_power.size > 0:
            theta = np.mod(np.arctan2(ky_grid[annulus], kx_grid[annulus]), np.pi)
            bin_index = np.floor(theta / np.pi * n_bins).astype(int)
            bin_index = np.clip(bin_index, 0, n_bins - 1)
            angle_power = np.bincount(bin_index, weights=annulus_power, minlength=n_bins)
            angle_count = np.bincount(bin_index, minlength=n_bins)
            populated_power = angle_power[angle_count > 0]

            if populated_power.size >= 2:
                median_power = np.median(populated_power)
                if median_power > 0:
                    stripe_scores[idx] = np.max(populated_power) / median_power

        gy, gx = np.gradient(resid)
        diag1 = np.roll(np.roll(resid, -1, axis=0), -1, axis=1) - resid
        diag2 = np.roll(np.roll(resid, -1, axis=0), 1, axis=1) - resid
        dir_powers = np.array([
            np.mean(gx ** 2),
            np.mean(gy ** 2),
            np.mean(diag1 ** 2),
            np.mean(diag2 ** 2),
        ], dtype=float)
        dir_powers = np.where(np.isfinite(dir_powers), dir_powers, 0.0)
        median_dir_power = np.median(dir_powers)
        if median_dir_power > 0:
            grad_aniso_scores[idx] = np.max(dir_powers) / median_dir_power

    # Two-sided stripe decision: an absolute backstop catches uniformly
    # artifact-dominated bands, while a within-stack relative-outlier cut
    # (gated by an absolute floor) catches a bad frame inside an otherwise
    # clean band without trimming a band of merely mildly anisotropic frames.
    finite = np.isfinite(stripe_scores)
    median_score = float(np.median(stripe_scores[finite])) if finite.any() else 1.0
    if not np.isfinite(median_score) or median_score <= 0:
        median_score = 1.0
    relative_cut = max(stripe_rel_floor, stripe_rel_factor * median_score)
    stripe_hit = (stripe_scores > stripe_threshold) | (stripe_scores > relative_cut)
    grad_hit = grad_aniso_scores > grad_aniso_threshold
    is_striped = stripe_hit | grad_hit
    return is_striped, stripe_scores, grad_aniso_scores


def extract_time_flags_from_ms(msfile, threshold=0.75, plotflag=False, return_flag=False, pols='XX'):
    """
    Extracts time steps and their corresponding flag status from a Measurement Set (MS).

    This function processes the flag data from all spectral windows (SPWs) in the MS file
    to compute the average flag status across baselines and channels. A time step is
    considered flagged if the mean flag value across all data exceeds a specified threshold.

    :param msfile: Path to the Measurement Set (MS) file.
    :type msfile: str
    :param threshold: Threshold for considering a time step as flagged, defaults to 0.75.
                      A time step is flagged if the mean flag fraction > threshold.
    :type threshold: float, optional
    :param plotflag: If True, generates a plot of the flag data and mean flag fraction, defaults to False.
    :type plotflag: bool, optional
    :return: A tuple containing:
             - `timedata` (numpy.ndarray): Array of time steps from the MS.
             - `timeflag` (numpy.ndarray): Boolean array where `True` indicates
                                           flagged time steps based on the threshold.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    t0 = datetime.now()

    ms.open(msfile)

    # Get spectral window information
    spwinfo = ms.getspectralwindowinfo()

    # Initialize storage for flag data
    flagdata_list = []
    timedata = None  # Initialize timedata to None for clarity

    pol_index = 1 if str(pols).upper() == 'YY' else 0

    # Loop through all spectral windows
    for spw_index, spw in enumerate(spwinfo.keys()):
        ms.selectinit(datadescid=0, reset=True)
        ms.selectinit(datadescid=spw_index)
        # Extract time data once (assume it's the same for all SPWs)
        if timedata is None:
            timedata = ms.getdata(['time'], ifraxis=True)['time']

        if spw_index == 0 and not return_flag:
            ms.close()
            t1 = datetime.now()
            log_print('INFO',
                      f"Extracting time flags from {os.path.basename(msfile)} took {(t1 - t0).total_seconds():.1f} seconds")
            return timedata, None
        data = ms.getdata(['flag'], ifraxis=True)
        flagdata_spw = data['flag']  # Shape: (npol, nchan, nbl, ntim)

        flag_pol_index = pol_index
        if flag_pol_index >= flagdata_spw.shape[0] and flagdata_spw.shape[0] == 1:
            flag_pol_index = 0
        if flag_pol_index >= flagdata_spw.shape[0]:
            raise ValueError(f"Requested polarization {pols} is not available in {msfile}")
        flagdata_mean = np.nanmean(flagdata_spw[flag_pol_index], axis=1)  # Shape: (nbl, ntim)
        flagdata_list.append(flagdata_mean)

    # Combine flag data from all SPWs
    flagdata_combined = np.vstack(flagdata_list)

    # Compute time flags based on the threshold
    flagdata_combined_mean = np.nanmean(flagdata_combined, axis=0)
    timeflag = flagdata_combined_mean > threshold

    # Plot results if required
    if plotflag:
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axs[0].pcolormesh(flagdata_combined, cmap='viridis', vmin=0, vmax=1)
        axs[0].set_title('Flag spectrogram')
        axs[0].set_ylabel('Frequency chan #')
        axs[1].plot(flagdata_combined_mean, label='Mean Flag Fraction', color='blue')
        axs[1].axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        axs[1].set_title('Mean Flag Fraction Per Time Step')
        axs[1].set_xlabel('Time Index')
        axs[1].set_ylabel('Mean Flag Fraction')
        axs[1].legend()
        plt.tight_layout()
        plt.show()

    ms.close()
    t1 = datetime.now()
    log_print('INFO',
              f"Extracting time flags from {os.path.basename(msfile)} took {(t1 - t0).total_seconds():.1f} seconds")
    return timedata, timeflag


def get_timedelta(tim):
    """
    Calculates the most frequent time difference (delta) in minutes from a sequence of time values.

    This function computes the differences between consecutive Modified Julian Dates (MJDs)
    in the provided `tim` object, converts them to minutes, and identifies the most frequently
    occurring time difference.

    :param tim: An object with an `.mjd` attribute containing time values in Modified Julian Date (MJD) format.
    :type tim: object
    :return: The most frequently occurring time difference(s) in minutes.
    :rtype: numpy.ndarray
    """
    # Calculate time differences in minutes
    time_deltas = np.round(np.diff(tim.mjd) * 24 * 60)

    # Get unique values and their counts
    unique_values, counts = np.unique(time_deltas, return_counts=True)

    # Identify the most frequent time difference(s)
    most_frequent_deltas = unique_values[counts == counts.max()]

    return most_frequent_deltas[0]


def plot_style(func):
    """
    A decorator to apply common plotting styles to the plotting functions.
    """

    @wraps(func)
    def wrapper(self, filename=None, *args, **kwargs):
        # Create a figure with two rows
        fig, axs = plt.subplots(nrows=2, figsize=(6, 5), sharex=True)

        # Call the original plotting function
        func(self, axs, *args, **kwargs)

        # Apply common styling
        ax_major = axs[0]
        ax_minor = axs[1]

        # Customize the major intervals plot
        ax_major.set_ylim(0 - 0.3, 1 + 0.3)
        ax_major.set_yticks([0, 1])
        ax_major.set_yticklabels(['False', 'True'])
        ax_major.set_title('Time Intervals for WSClean Imaging')
        ax_major.legend(loc='best')

        # Customize the minor intervals plot
        ax_minor.set_title('Time Intervals for Solar Differential Rotation')
        ax_minor.set_ylim(0, self.nintervals_minor + 1)
        ax_minor.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax_minor.set_ylabel('Sub-Interval Index')
        ax_minor.set_xlabel('Time [UT]')
        ax_minor.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return wrapper


class WSCleanTimeIntervals:
    def __init__(self, msfile, combine_scans=True, interval_length=50, pols='XX'):
        """
        A class to manage and compute time intervals for WSClean imaging from a Measurement Set (MS).

        This class can process each scan individually or treat all scans as a single entity.

        :param msfile: Path to the Measurement Set (MS) file.
        :type msfile: str
        :param combine_scans: If True, treat all scans as a single entity. If False, process each scan individually.
        :type combine_scans: bool
        """
        self.msfile = msfile
        self.combine_scans = combine_scans
        self.pols = pols
        self.scans = self._get_scans()
        self.scan_results = []
        self.tim = None
        self.tim_flag = None
        self.tdur = None
        self.tdt = None
        self.intervals_indices = None
        self.time_intervals = None
        self.nintervals_major = None
        self.nintervals_minor = 6  # Default value for minor intervals
        self.time_intervals_major_avg = None
        self.time_intervals_minor_avg = None
        self.interval_length = interval_length  # Length of each major interval in minutes

        if self.combine_scans:
            # Extract time data and flags for the entire MS
            self.tim, self.tim_flag = extract_time_flags_from_ms(self.msfile, pols=self.pols)
            self.tim = Time(self.tim / 24 / 3600, format='mjd')
            self.tdt = get_timedelta(self.tim)  # The most frequent time delta
            self.ntim = len(self.tim)
            self.tdur = self.tdt * self.ntim  # Total duration in minutes

    def _get_scans(self):
        """
        Extract scan information from the MS file.

        Returns:
            dict: Dictionary containing scan information.
        """
        ms.open(self.msfile)
        scans = ms.getscansummary()
        ms.close()
        return scans

    def _get_nintervals_major(self, tdur):
        """
        Calculate the number of major intervals for WSClean.

        Args:
            tdur (float): Total duration of the valid time range for WSClean, in minutes.

        Returns:
            int: Number of major intervals.
        """
        thour = tdur / self.interval_length
        nintervals_major = int(np.round(thour))
        if nintervals_major == 0:
            nintervals_major = 1
            print(
                f'WARNING: The time duration for WSClean is less than {self.interval_length:.0f} min. Setting nintervals_major to 1.')
        return nintervals_major

    def compute_intervals(self, nintervals_minor=6):
        """
        Compute time intervals for WSClean imaging.

        Args:
            nintervals_minor (int, optional): Number of minor intervals within each major interval. Default is 6.
        """
        self.nintervals_minor = nintervals_minor
        if self.combine_scans:
            # Process all scans as a single entity
            self._compute_intervals_combined(nintervals_minor)
        else:
            # Process each scan individually
            self._compute_intervals_individual(nintervals_minor)

    def _compute_intervals_combined(self, nintervals_minor):
        """
        Compute time intervals for WSClean imaging, treating all scans as a single entity.
        """
        # Calculate number of major intervals
        self.nintervals_major = self._get_nintervals_major(self.tdur)

        # Divide the valid time range into equal intervals
        tim_indices = np.arange(self.ntim)
        self.intervals_indices = np.array_split(tim_indices, self.nintervals_major)
        self.time_intervals = [self.tim[seg] for seg in self.intervals_indices]
        self.time_intervals_major_avg = Time([np.nanmean(seg.mjd) for seg in self.time_intervals], format='mjd')

        # Compute minor intervals
        nintervals_all = self.nintervals_major * nintervals_minor
        if nintervals_all > self.ntim:
            print(
                f'WARNING: The full time span only has {self.ntim} time steps. The number of intervals for solar differential rotation correction is {nintervals_all}. Setting nintervals_all to {self.ntim}.')
            nintervals_all = self.ntim

        intervals_all_indices = np.array_split(tim_indices, nintervals_all)
        self.time_intervals_minor = []
        self.time_intervals_minor_avg = []
        for interval_idx in range(self.nintervals_major):
            start_idx = interval_idx * nintervals_minor
            end_idx = min((interval_idx + 1) * nintervals_minor, len(intervals_all_indices))
            interval_minor = [self.tim[seg] for seg in intervals_all_indices[start_idx:end_idx]]
            self.time_intervals_minor.append(interval_minor)
            self.time_intervals_minor_avg.append(Time([np.nanmean(seg.mjd) for seg in interval_minor], format='mjd'))

    def _compute_intervals_individual(self, nintervals_minor):
        """
        Compute time intervals for WSClean imaging for each scan individually.
        """
        for scan_id, scan_info in self.scans.items():
            for subscan_id, subscan_data in scan_info.items():
                # Extract scan information
                begin_time = subscan_data['BeginTime']
                end_time = subscan_data['EndTime']
                integration_time = subscan_data['IntegrationTime']

                # Calculate time indices for the scan
                tim = np.arange(begin_time, end_time, integration_time / 86400)  # Convert seconds to days
                tim = Time(tim, format='mjd')
                ntim = len(tim)
                tdur = (end_time - begin_time) * 1440  # Convert days to minutes

                # Calculate number of major intervals
                nintervals_major = self._get_nintervals_major(tdur)

                # Divide the valid time range into equal intervals
                tim_indices = np.arange(ntim)
                intervals_indices = np.array_split(tim_indices, nintervals_major)
                time_intervals = [tim[seg] for seg in intervals_indices]
                time_intervals_major_avg = Time([np.nanmean(seg.mjd) for seg in time_intervals], format='mjd')

                # Compute minor intervals
                nintervals_all = nintervals_major * nintervals_minor
                if nintervals_all > ntim:
                    print(
                        f'WARNING: The full time span only has {ntim} time steps. The number of intervals for solar differential rotation correction is {nintervals_all}. Setting nintervals_all to {ntim}.')
                    nintervals_all = ntim

                intervals_all_indices = np.array_split(tim_indices, nintervals_all)
                time_intervals_minor = []
                time_intervals_minor_avg = []
                for interval_idx in range(nintervals_major):
                    start_idx = interval_idx * nintervals_minor
                    end_idx = min((interval_idx + 1) * nintervals_minor, len(intervals_all_indices))
                    interval_minor = [tim[seg] for seg in intervals_all_indices[start_idx:end_idx]]
                    time_intervals_minor.append(interval_minor)
                    time_intervals_minor_avg.append(Time([np.nanmean(seg.mjd) for seg in interval_minor], format='mjd'))

                # Store results for this scan
                self.scan_results.append({
                    'ScanId': scan_id,
                    'SubScanId': subscan_id,
                    'BeginTime': begin_time,
                    'EndTime': end_time,
                    'IntegrationTime': integration_time,
                    'tdur': tdur,
                    'nintervals_major': nintervals_major,
                    'nintervals_minor': nintervals_minor,
                    'intervals_indices': intervals_indices,
                    'time_intervals': time_intervals,
                    'time_intervals_major_avg': time_intervals_major_avg,
                    'time_intervals_minor': time_intervals_minor,
                    'time_intervals_minor_avg': time_intervals_minor_avg
                })

    @plot_style
    def _plot_combined(self, axs):
        """
        Plot the time intervals for the entire MS file.
        """
        # Plot major intervals
        cmap = plt.cm.get_cmap('tab20', self.nintervals_major)
        for sidx, seg_times in enumerate(self.time_intervals):
            color = cmap(sidx)
            axs[0].plot(seg_times.plot_date, [1] * len(seg_times), marker='.', linestyle='', color=color,
                        label=f'Major Interval {sidx + 1}')

        # Plot minor intervals
        cmap2 = plt.cm.get_cmap('tab20', self.nintervals_major)
        y = np.arange(1, self.nintervals_minor + 1)
        for sidx, interval_minor in enumerate(self.time_intervals_minor):
            color = cmap2(sidx)
            for sridx, seg_times in enumerate(interval_minor):
                axs[1].plot(seg_times.plot_date, [y[sridx]] * len(seg_times), marker='.', linestyle='', color=color)

    @plot_style
    def _plot_individual(self, axs):
        """
        Plot the time intervals for each scan individually.
        """
        # Generate a colormap for major intervals
        nintervals_major_total = sum([scan_result['nintervals_major'] for scan_result in self.scan_results])
        cmap = plt.cm.get_cmap('tab20', nintervals_major_total)

        color_idx = 0  # Index to track colors across major and minor intervals
        for scan_result in self.scan_results:
            time_intervals = scan_result['time_intervals']
            time_intervals_minor = scan_result['time_intervals_minor']
            nintervals_major = scan_result['nintervals_major']
            nintervals_minor = scan_result['nintervals_minor']

            # Plot major intervals
            for sidx, seg_times in enumerate(time_intervals):
                color = cmap(color_idx)
                axs[0].plot(seg_times.plot_date, [1] * len(seg_times), marker='.', linestyle='', color=color,
                            label=f'Major Interval {color_idx + 1}')
                interval_minor = time_intervals_minor[sidx]
                # Plot minor intervals
                for sridx, seg_times_minor in enumerate(interval_minor):
                    axs[1].plot(seg_times_minor.plot_date, [sridx + 1] * len(seg_times_minor), marker='.',
                                linestyle='',
                                color=color)
                color_idx += 1

    def plot(self, filename=None):
        """
        Plot the time intervals and optionally save the plot to a file.

        Args:
            filename (str, optional): Path to save the plot. If None, the plot is displayed.
        """
        if self.combine_scans:
            if self.time_intervals is None:
                raise ValueError("Intervals have not been computed. Call `compute_intervals` first.")
            self._plot_combined(filename)
        else:
            if not self.scan_results:
                raise ValueError("Intervals have not been computed. Call `compute_intervals` first.")
            self._plot_individual(filename)

    def results(self):
        """
        Returns the computed intervals and related information.

        :return: If combine_scans=True, a dictionary containing:
                 - `intervals_indices` (list of lists): Indices of time steps in each interval.
                 - `time_intervals` (list of `astropy.Time`): Time values for each interval.
                 - `tdur` (float): Total duration of the valid time range for WSClean, in minutes.
                 - `tdt` (float): The most frequent time delta, in minutes.
                 - `nintervals_major` (int): Number of major intervals.
                 - `nintervals_minor` (int): Number of minor intervals within each major interval.
                 - `time_intervals_major_avg` (`astropy.Time`): Average time for each major interval.
                 - `time_intervals_minor_avg` (list of `astropy.Time`): Average time for each minor interval within major intervals.
                 If combine_scans=False, a list of dictionaries, each containing:
                 - `ScanId`: The ID of the scan.
                 - `SubScanId`: The ID of the sub-scan.
                 - `BeginTime`: The start time of the scan.
                 - `EndTime`: The end time of the scan.
                 - `IntegrationTime`: The integration time of the scan.
                 - `tdur`: Total duration of the valid time range for WSClean, in minutes.
                 - `nintervals_major`: Number of major intervals.
                 - `nintervals_minor`: Number of minor intervals within each major interval.
                 - `intervals_indices`: Indices of time steps in each major interval.
                 - `time_intervals`: Time values for each major interval.
                 - `time_intervals_major_avg`: Average time for each major interval.
                 - `time_intervals_minor`: Time values for each minor interval.
                 - `time_intervals_minor_avg`: Average time for each minor interval within major intervals.
        :rtype: dict or list of dict
        """
        if self.combine_scans:
            return {
                'tdur': self.tdur,
                'tdt': self.tdt,
                'intervals_indices': self.intervals_indices,
                'time_intervals': self.time_intervals,
                'nintervals_major': self.nintervals_major,
                'nintervals_minor': self.nintervals_minor,
                'time_intervals_major_avg': self.time_intervals_major_avg,
                'time_intervals_minor_avg': self.time_intervals_minor_avg
            }
        else:
            return self.scan_results


def trange2timerange(trange):
    """
    Convert a time range tuple in datetime format to a string representation.

    :param trange: A tuple containing start and end times as datetime objects.
    :type trange: tuple
    :return: A string representation of the time range.
    :rtype: str
    """

    sttime, edtime = trange
    if isinstance(sttime, str):
        sttime = pd.to_datetime(sttime)
    if isinstance(edtime, str):
        edtime = pd.to_datetime(edtime)
    timerange = f"{sttime.strftime('%Y/%m/%d/%H:%M:%S')}~{edtime.strftime('%Y/%m/%d/%H:%M:%S')}"
    return timerange


def rotateimage(data, xc_centre, yc_centre, p_angle):
    """
    Rotate an image around a specified point (xc_centre, yc_centre) by a given angle.

    :param data: The image data.
    :type data: numpy.ndarray
    :param xc_centre: The x-coordinate of the rotation center.
    :type xc_centre: int
    :param yc_centre: The y-coordinate of the rotation center.
    :type yc_centre: int
    :param p_angle: The rotation angle in degrees.
    :type p_angle: float
    :return: The rotated image.
    :rtype: numpy.ndarray
    """

    padX = [data.shape[1] - xc_centre, xc_centre]
    padY = [data.shape[0] - yc_centre, yc_centre]
    imgP = np.pad(data, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, p_angle, reshape=False, order=0, prefilter=False)
    return imgR[padY[0]:-padY[1], padX[0]:-padX[1]]


def solar_diff_rot_heliofits(in_fits, newtime, out_fits, in_time=None, template_fits=None, showplt=False,
                             overwrite_prev=True):
    """
    Reproject a FITS file to account for solar differential rotation to a new observation time.

    Parameters
    ----------
    in_fits : str
        Path to the input FITS file to be reprojected
    newtime : astropy.time.Time
        The new time to which the map is reprojected
    out_fits : str
        Path for the output FITS file
    in_time : astropy.time.Time, optional
        The reference time for the input map, defaults to the middle of the exposure time. This is useful when exposure time is not provided in the FITS header.
    template_fits : str, optional
        Path to template FITS file for output. If None, uses in_fits as template
    showplt : bool, optional
        Show plots of the original and reprojected maps, defaults to False

    Returns
    -------
    str
        Path to the output FITS file

    Notes
    -----
    The output header's DATE-OBS/DATE record the reprojection target epoch
    (the epoch of the output WCS), not the acquisition time. The true
    acquisition window of the input image is preserved in STARTOBS/ENDOBS,
    and EXPTIME is carried over unchanged.
    """

    # Convert FITS to SunPy Map
    in_map = smap.Map(in_fits)

    # Calculate reference time and output time
    if in_time is None:
        in_time = in_map.date  # + in_map.exposure_time / 2
    out_time = in_map.date + (Time(newtime) - in_time)
    # out_time = Time(newtime)

    # Set up output frame
    out_frame = Helioprojective(observer=in_map.observer_coordinate,
                                obstime=out_time,
                                rsun=in_map.coordinate_frame.rsun)

    # Define output center and reference pixel
    out_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=out_frame)
    out_ref_pixel = [in_map.reference_pixel.x.value,
                     in_map.reference_pixel.y.value] * in_map.reference_pixel.x.unit

    # Create output header
    out_header = smap.make_fitswcs_header(in_map.data.shape,
                                          out_center,
                                          reference_pixel=out_ref_pixel,
                                          scale=u.Quantity(in_map.scale))
    out_wcs = WCS(out_header)

    # Perform reprojection
    with propagate_with_solar_surface():
        out_map, footprint = in_map.reproject_to(out_wcs, return_footprint=True)

    # Fill missing data from original map
    out_data = out_map.data
    out_data[footprint == 0] = in_map.data[footprint == 0]

    # Set NaN values to 0
    out_data[np.isnan(out_data)] = 0

    # Create new map with reprojected data
    out_map = smap.Map(out_data, out_map.meta)
    out_map.meta['p_angle'] = in_map.meta['p_angle']

    # Display plots if requested
    if showplt:
        fig = plt.figure(figsize=(12, 4))

        ax1 = fig.add_subplot(121, projection=in_map)
        in_map.plot(axes=ax1, title='Original map')
        plt.colorbar()

        ax2 = fig.add_subplot(122, projection=out_map)
        out_map.plot(axes=ax2,
                     title=f"Reprojected to an Earth observer {(Time(newtime) - in_time).to('day')} later")
        plt.colorbar()
        plt.show()

    # Save to FITS file using template if provided
    if template_fits is None:
        template_fits = in_fits

    # Load template header
    template_map = smap.Map(template_fits)
    template_header = template_map.meta.copy()

    # Update necessary header information
    template_header.update({
        'DATE-OBS': out_map.date.iso,
        'DATE': out_map.date.iso,
        'CRVAL1': out_map.reference_coordinate.Tx.value,
        'CRVAL2': out_map.reference_coordinate.Ty.value,
        'CRPIX1': out_map.reference_pixel.x.value,
        'CRPIX2': out_map.reference_pixel.y.value,
        'PC1_1': out_map.rotation_matrix[0, 0],
        'PC1_2': out_map.rotation_matrix[0, 1],
        'PC2_1': out_map.rotation_matrix[1, 0],
        'PC2_2': out_map.rotation_matrix[1, 1]
    })

    # DATE-OBS above is the reprojection target epoch; keep the true
    # acquisition window of the input image in STARTOBS/ENDOBS.
    startobs = in_map.meta.get('startobs') or in_map.meta.get('date-obs')
    if startobs:
        template_header['startobs'] = Time(startobs).isot
        endobs = in_map.meta.get('endobs')
        exptime = in_map.meta.get('exptime')
        if not endobs and exptime:
            endobs = (Time(startobs) + float(exptime) * u.s).isot
        if endobs:
            template_header['endobs'] = Time(endobs).isot

    # Create new map with template header and save to FITS
    final_map = smap.Map(out_data, template_header)
    outdir = os.path.dirname(out_fits)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    final_map.save(out_fits, overwrite=overwrite_prev)

    return out_fits


def sunpyfits_to_j2000fits(in_fits, out_fits, template_fits=None, overwrite_prev=True):
    """
    Rotate a solar FITS file from helioprojective to RA-DEC coordinates and save to a new FITS file.
    Originally written by Dr. Peijin Zhang
    Parameters
    ----------
    in_fits : str
        Path to input FITS file in helioprojective coordinates
    out_fits : str
        Path for output FITS file in RA-DEC coordinates
    template_fits : str, optional
        Path to template FITS file for output format. If None, uses in_fits as template
    overwrite_prev : bool, optional
        If True, overwrites existing output file. Defaults to True

    Returns
    -------
    str
        Path to the output FITS file
    """
    import astropy.units as u
    from astropy.io import fits

    # Load input map
    in_map = smap.Map(in_fits)
    p_ang = in_map.meta['p_angle'] * u.deg

    # Get reference pixel coordinates
    ref_x = int(in_map.reference_pixel.x.value)
    ref_y = int(in_map.reference_pixel.y.value)

    # Perform rotation
    data_rot = rotateimage(in_map.data, ref_x, ref_y, -p_ang.to('deg').value)

    # Set NaN values to 0
    data_rot[np.isnan(data_rot)] = 0

    # Use template if provided, otherwise use input file
    if template_fits is None:
        template_fits = in_fits

    # Read template header
    with fits.open(template_fits) as hdul:
        template_header = hdul[0].header.copy()

    # Update necessary header information
    template_header.update({
        'DATE-OBS': in_map.date.iso,
        'DATE': in_map.date.iso,
    })

    fits.writeto(out_fits, data_rot, template_header, overwrite=overwrite_prev)

    return out_fits


def get_beam_kernel(header=None, bmaj=None, bmin=None, bpa=0.0, pixel_scale_arcsec=None):
    """
    Returns a normalized 2D elliptical Gaussian kernel representing the beam.

    The beam can be specified in two ways:

    1. By providing a FITS header (via the 'header' parameter). In this case, if not explicitly provided,
       the beam parameters are extracted from the header using the following keywords:
         - 'BMAJ' (FWHM of the major axis, in degrees)
         - 'BMIN' (FWHM of the minor axis, in degrees)
         - 'BPA'  (beam position angle in degrees, measured counterclockwise from North)
       In addition, if 'pixel_scale_arcsec' is not provided, it is determined from the header using:
         - 'CDELT1' (pixel scale, in degrees or arcseconds)
         - 'CUNIT1' (unit of CDELT1)

    2. By directly providing the beam parameters and pixel scale:
         - bmaj: FWHM of the major axis in degrees.
         - bmin: FWHM of the minor axis in degrees.
         - bpa:  Beam position angle in degrees (counterclockwise from North).
         - pixel_scale_arcsec: Pixel scale in arcseconds per pixel.

    The function converts the FWHM values (in degrees) to arcseconds and then to pixels using the provided
    pixel scale. It then converts the FWHM to sigma (standard deviation) via sigma = FWHM / 2.35482. The beam
    position angle is converted to a rotation angle (theta) relative to the x-axis via theta = radians(90 - BPA).

    :param header: FITS header containing the beam and pixel scale information, defaults to None.
    :type header: astropy.io.fits.header.Header, optional
    :param bmaj: FWHM of the beam major axis in degrees, defaults to None.
    :type bmaj: float, optional
    :param bmin: FWHM of the beam minor axis in degrees, defaults to None.
    :type bmin: float, optional
    :param bpa: Beam position angle in degrees (counterclockwise from North), defaults to None.
    :type bpa: float, optional
    :param pixel_scale_arcsec: Pixel scale in arcseconds per pixel, defaults to None.
    :type pixel_scale_arcsec: float, optional
    :raises KeyError: If a header is provided but required keywords ('BMAJ', 'BMIN', 'BPA', 'CDELT1', or 'CUNIT1') are missing.
    :raises ValueError: If neither a header nor the required beam parameters and pixel_scale_arcsec are provided, or if 'CUNIT1' is unrecognized.
    :return: A 2D numpy array representing the normalized elliptical Gaussian beam kernel.
    :rtype: numpy.ndarray
    """
    if header is not None:
        # If a header is provided, extract beam parameters and pixel scale first.
        try:
            if header["BMAJ"] > 0 and header["BMIN"] > 0:
                bmaj = header["BMAJ"]
                bmin = header["BMIN"]
                bpa = header["BPA"]
            else:
                print(f"Beam parameters 'BMAJ' and 'BMIN' must be positive. Use the provided values: ")
        except KeyError as e:
            raise KeyError(
                "Beam parameters 'BMAJ', 'BMIN', and 'BPA' must be provided either as inputs or in the header.") from e
        if pixel_scale_arcsec is None:
            try:
                cdelt1 = header["CDELT1"]
                cunit1 = header["CUNIT1"].strip().lower()
            except KeyError as e:
                raise KeyError(
                    "When header is provided, it must contain 'CDELT1' and 'CUNIT1' for pixel scale conversion.") from e
            if cunit1 == "deg":
                pixel_scale_arcsec = abs(cdelt1) * 3600.0
            elif cunit1 in ("arcsec", "arcsec."):
                pixel_scale_arcsec = abs(cdelt1)
            else:
                raise ValueError(f"Unrecognized CUNIT1: expected 'deg' or 'arcsec', got '{cunit1}'.")
    else:
        # If no header is provided, all beam parameters must be explicitly given.
        if bmaj is None or bmin is None or bpa is None or pixel_scale_arcsec is None:
            raise ValueError("Either provide a header or explicitly supply bmaj, bmin, bpa, and pixel_scale_arcsec.")

    # Convert FWHM from degrees to arcseconds.
    bmaj_arcsec = bmaj * 3600.0
    bmin_arcsec = bmin * 3600.0

    # Convert FWHM from arcseconds to pixels.
    fwhm_major_pixels = bmaj_arcsec / pixel_scale_arcsec
    fwhm_minor_pixels = bmin_arcsec / pixel_scale_arcsec

    # Convert FWHM to standard deviation (sigma) in pixels.
    sigma_major = fwhm_major_pixels / 2.35482
    sigma_minor = fwhm_minor_pixels / 2.35482

    # Convert beam position angle (BPA) to rotation angle (theta) relative to the x-axis.
    # BPA is measured counterclockwise from North (the y-axis), so theta = radians(90 - BPA).
    theta = np.deg2rad(90.0 - bpa)

    # Determine an appropriate kernel size (e.g., 8 sigma to capture most of the Gaussian)
    size = int(np.ceil(8 * max(sigma_major, sigma_minor)))
    if size % 2 == 0:
        size += 1  # ensure the kernel size is odd

    # Create the Gaussian kernel using astropy.convolution.Gaussian2DKernel.
    # The kernel is normalized so that its integral (sum) is unity.
    kernel = Gaussian2DKernel(x_stddev=sigma_major, y_stddev=sigma_minor, theta=theta,
                              x_size=size, y_size=size, mode='oversample')
    return kernel.array


# Function to compute disk image given a header, a disk radius value (in arcsec),
# and the uniform disk value (fdn_val).
def compute_disk_image(header, dsz_value, fdn_value, add_limb=False, bmaj=None):
    # Extract pixel scale from header.
    try:
        cdelt1 = header["CDELT1"]
        cunit1 = header["CUNIT1"].strip().lower()
    except KeyError as e:
        raise KeyError("Header must contain 'CDELT1' and 'CUNIT1'.") from e
    if cunit1 == "deg":
        pixel_scale_arcsec = abs(cdelt1) * 3600.0
    elif cunit1 in ("arcsec", "arcsec."):
        pixel_scale_arcsec = abs(cdelt1)
    else:
        raise ValueError(f"Unrecognized CUNIT1: expected 'deg' or 'arcsec', got '{cunit1}'.")

    # Calculate disk radius in pixels.
    disk_radius_pixels = dsz_value / pixel_scale_arcsec

    # Get image dimensions.
    ny, nx = header["NAXIS2"], header["NAXIS1"]

    # Get center coordinates (FITS convention: 1-indexed).
    try:
        crpix1 = header["CRPIX1"]
        crpix2 = header["CRPIX2"]
    except KeyError as e:
        raise KeyError("Header must contain 'CRPIX1' and 'CRPIX2'.") from e
    x_center = crpix1 - 1
    y_center = crpix2 - 1

    # Create coordinate grid.
    y_indices, x_indices = np.ogrid[:ny, :nx]
    distance = np.sqrt((x_indices - x_center) ** 2 + (y_indices - y_center) ** 2)

    # Create disk mask and compute uniform disk value per pixel so that total flux is fdn_value.
    disk_mask = distance <= disk_radius_pixels
    n_disk_pixels = np.count_nonzero(disk_mask)
    # disk_value = fdn_value / n_disk_pixels
    disk_img = np.zeros((ny, nx), dtype=np.float64)
    disk_img[disk_mask] = 1.0

    if add_limb:
        # Compute limb width: 6 times the beam FWHM in pixels.
        if 'BMAJ' in header:
            if header["BMAJ"] > 0:
                bmaj = header["BMAJ"]
            else:
                if bmaj is None:
                    raise KeyError("Header must contain 'BMAJ' for beam FWHM computation.")
                else:
                    print(f"Beam parameters 'BMAJ' must be positive. Use the provided value: {bmaj}")
        beam_fwhm_arcsec = bmaj * 3600.0
        beam_fwhm_pixels = beam_fwhm_arcsec / pixel_scale_arcsec
        limb_width = 6 * beam_fwhm_pixels
        # Limb is defined as annulus: (disk_radius_pixels - limb_width) < distance <= disk_radius_pixels.
        limb_mask = (distance > (disk_radius_pixels - limb_width)) & (distance <= disk_radius_pixels)
        disk_img[limb_mask] *= 1.6
    return disk_img


class FluxComparison(NamedTuple):
    infile: str  # the input FITS file
    data_max: float  # the maximum value of the data
    data_p9999: float  # the 99.99 percentile of the data
    model_max: float  # the maximum value of the model
    flux_data: float = None  # the flux of the data
    flux_model: float = None  # the flux of the model
    flux_input: float = None  # the flux used to create the model
    tbmax_data: float = None  # the brightness temperature of the data [kK]
    tbmax_model: float = None  # the brightness temperature of the model [kK]
    rms_data: float = None  # the RMS of the ondisk data
    snr_data: float = None  # the SNR of the ondisk data


def add_convolved_disk_to_fits(
        in_fits: Union[str, List[str]], out_fits: Union[str, List[str]],
        dsz: Union[float, List[float]], fdn: Union[float, List[float]],
        ignore_data: bool = False, toTb: bool = False, rfreq: str = None,
        add_limb: bool = False, bmaj: float = None, no_negative: bool = False,
        doconvolve: bool = True,
        create_mask: bool = False,
) -> List[FluxComparison]:
    """
    Add a uniform disk to FITS images, convolve with beam, and save.

    Both dsz and fdn accept float or list. Lists are averaged internally.
    Returns TbComparison for each file for downstream analysis.

    Parameters
    ----------
    in_fits : str or list
        Input FITS filename(s).
    out_fits : str or list
        Output FITS filename(s).
    dsz : float or list
        Disk radius(es) in arcseconds.
    fdn : float or list
        Total disk flux(es).
    ignore_data : bool
        If True, output only the disk model.
    toTb : bool
        If True, convert output to brightness temperature.
    rfreq : str
        Reference frequency string (e.g., '5.5GHz') for Tb conversion.
    add_limb : bool
        If True, add a bright limb to the disk.
    bmaj : float
        Beam FWHM (deg) for limb width calculation.

    Returns
    -------
    stats : list of FluxComparison
    """
    # normalize inputs
    files_in = [in_fits] if isinstance(in_fits, str) else list(in_fits)
    files_out = [out_fits] if isinstance(out_fits, str) else list(out_fits)
    # open header
    with fits.open(files_in[0]) as hdulist:
        for idx, hdu in enumerate(hdulist):
            if 'CDELT1' in hdu.header:
                hdr = hdu.header
                dateobs = Time(hdr['DATE-OBS'])
                rsun_obs = sun.angular_radius(dateobs).value
                break
    if bmaj is None:
        bmaj = hdr.get('BMAJ', None)
        bmin = hdr.get('BMIN', None)
    else:
        bmin = bmaj  # assume circular beam if bmaj is provided
    beam_area = np.pi * np.abs(np.deg2rad(bmaj)) * np.abs(np.deg2rad(bmin)) / (4 * np.log(2))
    pix_area = ((np.abs(hdr['CDELT1']) * u.Unit(hdr['CUNIT1'])).to(u.rad).value * (
            np.abs(hdr['CDELT2']) * u.Unit(hdr['CUNIT2']))).to(u.rad).value
    # determine mean values
    radius = float(np.mean(dsz)) if not isinstance(dsz, numbers.Number) else dsz
    flux = float(np.mean(fdn)) if not isinstance(fdn, numbers.Number) else fdn
    # build disk model
    disk_img = compute_disk_image(hdr, radius, flux, add_limb, bmaj)

    if create_mask:
        mask_img = np.zeros_like(disk_img, dtype=bool)
        mask_img[:] = False  # initialize mask
        crpix1, crpix2 = hdr['CRPIX1'], hdr['CRPIX2']
        ny, nx = disk_img.shape
        y_indices, x_indices = np.ogrid[:ny, :nx]
        rsun_pix = rsun_obs / np.abs(hdr['CDELT1']) / 3600
        dist = np.sqrt((x_indices - crpix1) ** 2 + (y_indices - crpix2) ** 2)
        mask_img[dist <= rsun_pix * 1.1] = True
        mask_outf = files_in[0].replace('image.fits', 'mask.fits')
        fits.writeto(mask_outf, mask_img.astype(np.float32), hdr, overwrite=True)
    if doconvolve:
        # convolve
        beam = get_beam_kernel(header=hdr, bmaj=bmaj, bmin=bmin)
        diskmodel_conv = fftconvolve(disk_img, beam, mode='same')
        # diskmodel_conv = diskmodel_conv / np.nansum(diskmodel_conv) * (flux*beam_area/pix_area)  # normalize to flux and beam area
        diskmodel_conv = diskmodel_conv / np.nansum(diskmodel_conv) * (flux)  # normalize to flux and beam area
    else:
        diskmodel_conv = disk_img / np.nansum(disk_img) * (flux)  # normalize to flux and beam area
    diskmodel_max = np.nanmax(diskmodel_conv)
    # diskmodel_intergal = np.nansum(diskmodel_conv) * pix_area / beam_area
    diskmodel_intergal = np.nansum(diskmodel_conv)

    # print(f'disk flux: {np.nansum(disk_conv)*pix_area/beam_area:.0f} Jy. intended flux: {flux:.0f} Jy')
    factor_area = pix_area / beam_area
    # Tb conversion factor
    nu = float(rfreq.rstrip('GHz')) * 1e9
    factor = (2 * constants.k / constants.c ** 2) * nu ** 2
    jy_to_si = 1e-26  # Jy to SI units (W/m^2/Hz)
    jy2tb = jy_to_si / pix_area / factor
    # print(np.nanmax(diskmodel_conv) * jy2tb, 'K')
    tbmax_model = np.nanmax(diskmodel_conv) * jy2tb / 1e3  # convert to kK
    # print(f'Beam area: {beam_area} deg^2, pixel area: {pix_area} deg^2, max disk Tb: {tbmax_model:.3f} kK')
    if toTb:
        diskmodel_conv *= jy2tb

    # process each file
    stats = []
    for fidx, (f_in, outf) in enumerate(zip(files_in, files_out)):
        with fits.open(f_in) as hdulist:
            for idx, hdu in enumerate(hdulist):
                if 'CDELT1' in hdu.header:
                    hdr = hdu.header
                    dat = hdu.data.copy()
                    break
        dat_squeeze = np.squeeze(dat)
        if fidx == 0:
            dateobs = Time(hdr['DATE-OBS'])
            rsun_obs = sun.angular_radius(dateobs).value
        rsun_pix = rsun_obs / np.abs(hdr['CDELT1']) / 3600
        crpix1, crpix2 = hdr['CRPIX1'], hdr['CRPIX2']
        ny, nx = dat_squeeze.shape
        mask_ring, mask_disk = get_solar_radius_mask(rsun_pix, crpix1, crpix2, ny, nx, rsun_ratio=[1.2, 1.3])
        if no_negative:
            dat[dat < 0] = 0
        out_data = diskmodel_conv if ignore_data else dat + diskmodel_conv
        fits.writeto(outf, out_data, hdr, overwrite=True)

        dat_disk = dat_squeeze[mask_disk]
        p100, p9999, p50 = np.nanpercentile(dat_disk, [100, 99.99, 50])
        # rms_data = np.sqrt(np.nanmean(dat_disk[dat_disk < p50] ** 2))
        rms_data = np.sqrt(np.nanmean(dat_squeeze[mask_ring] ** 2))
        dataintegral = np.nansum(dat_disk)
        tbmax_data = p9999 * factor_area * jy2tb / 1e3  # convert to kK
        stats.append(
            FluxComparison(f_in, p100 * factor_area, p9999 * factor_area, diskmodel_max, dataintegral * factor_area,
                           diskmodel_intergal, flux, tbmax_data, tbmax_model, rms_data * factor_area, p9999 / rms_data))
    if len(stats) == 1:
        return stats[0]
    else:
        return stats


def disk_size_function(v, c1, alpha1):
    """
    Analytic model for disk size as a function of frequency.

    R(v) = c1 / v^(alpha1) + c2 / v^(alpha2)

    Parameters:
      v      : float or array_like
               Frequency (e.g., in GHz).
      c1     : float
               Coefficient for the first term.
      alpha1 : float
               Exponent for the first term.

    Returns:
      Disk size at frequency v.
    """
    return c1 * v ** (-alpha1)


def flux_function(v, c1, alpha1, c2, alpha2, c3, alpha3):
    """
    Analytic model for disk flux as a function of frequency.

    F(v) = c1 * v^(alpha1) + c2 * v^(alpha2) + c3 * v^(alpha3)

    Parameters:
      v      : float or array_like
               Frequency (e.g., in GHz).
      c1     : float
               Coefficient for the first term.
      alpha1 : float
               Exponent for the first term.
      c2     : float
               Coefficient for the second term.
      alpha2 : float
               Exponent for the second term.
    c3      : float
               Coefficient for the third term.
    alpha3 : float
               Exponent for the third term.

    Returns:
      Disk flux at frequency v.
    """
    return c1 * v ** (alpha1) + c2 * v ** (alpha2) + c3 * v ** (alpha3)


def calc_diskmodel(tim, nbands, freq):
    from astropy.time import Time
    ## The following disk size and flux density are derived using diskfitter based on 22 days of quiescent Sun data in 2019/09-2019/10.
    ## The disk sizes of all days are rescaled to the size on 2019/09/01.
    ## fitting params for disk size. [8.99295515e+02, 3.26324866e-01, 3.59103477e+02, -1.88429536e-01]
    ## fitting params for disk flux density. [60.9254115   0.69705383  0.60279497  2.29533769]
    ## Note: the lastest diskfitter is on /data1/sjyu/eovsa/ipynb_scripts, not in local. The parameters can be found in eovsa_disk_fitting.ipynb
    # params_disksize = [899.295515, 0.326324866, 359.103477, -0.188429536]
    params_disksize = [9.91363494e+02, 7.96091910e-03]
    # params_flux = [60.9254115, 0.69705383, 0.60279497, 2.29533769]
    params_flux = [58.76705446, 0.71404115, 1.01304963, 2.11645297, 16.32953077, -1.41759891]

    # Get current solar distance and modify the default size accordingly
    try:
        from sunpy.coordinates.sun import earth_distance
        fac = earth_distance('2019/09/01') / earth_distance(tim)
    except Exception:
        import sunpy.coordinates.ephemeris as eph
        fac = eph.get_sunearth_distance('2019/09/01') / eph.get_sunearth_distance(tim)

    if nbands == 34:
        if tim.mjd < Time('2018-03-13').mjd:
            # Interpolate size to 31 spectral windows (bands 4-34 -> spw 0-30)
            newsize = disk_size_function(freq[3:], *params_disksize) * fac.to_value()
        else:
            # Dates between 2018-03-13 have 33 spectral windows
            newsize = disk_size_function(freq[[0] + list(range(2, 34))], *params_disksize) * fac.to_value()
    else:
        newsize = disk_size_function(freq, *params_disksize) * fac.to_value()
    dsize = np.array([str(i)[:5] + 'arcsec' for i in newsize], dtype='U12')

    if nbands == 34:
        if tim.mjd < Time('2018-03-13').mjd:
            # Interpolate size to 31 spectal windows (bands 4-34 -> spw 0-30)
            fdens = flux_function(freq[3:], *params_flux)
        else:
            # Dates between 2018-03-13 have 33 spectral windows
            fdens = flux_function(freq[[0] + list(range(2, 34))], *params_flux)
    else:
        fdens = flux_function(freq, *params_flux)
    return dsize, fdens * 1e4


def ant_trange(vis):
    ''' Figure out nominal times for tracking of old EOVSA antennas, and return time
        range in CASA format
    '''
    from eovsapy import eovsa_array as ea
    from astropy.time import Time

    # Get timerange from the visibility file
    # msinfo = dict.fromkeys(['vis', 'scans', 'fieldids', 'btimes', 'btimestr', 'inttimes', 'ras', 'decs', 'observatory'])
    ms.open(vis)
    # metadata = ms.metadata()
    scans = ms.getscansummary()
    ms.close()
    sk = sorted(list(scans.keys()))
    vistrange = np.array([scans[sk[0]]['0']['BeginTime'], scans[sk[-1]]['0']['EndTime']])

    # Get the Sun transit time, based on the date in the vis file name (must have UDByyyymmdd in the name)
    aa = ea.eovsa_array()
    date = vis.split('UDB')[-1][:8]
    slashdate = date[:4] + '/' + date[4:6] + '/' + date[6:8]
    aa.date = slashdate
    sun = aa.cat['Sun']
    mjd_transit = Time(aa.next_transit(sun).datetime(), format='datetime').mjd
    # Construct timerange limits based on +/- 3h55m from transit time (when all dishes are nominally tracking)
    # and clip the visibility range not to exceed those limits
    mjdrange = np.clip(vistrange, mjd_transit - 0.1632, mjd_transit + 0.1632)
    trange = Time(mjdrange[0], format='mjd').iso[:19] + '~' + Time(mjdrange[1], format='mjd').iso[:19]
    trange = trange.replace('-', '/').replace(' ', '/')
    return trange


def format_spw(spw):
    """
    Formats the spectral window (spw) string for file naming, ensuring start and end values are separated by a dash and zero-padded to two digits.

    :param str spw: The spectral window in the format "start~end", where start and end are integers.
    :return: A formatted string with start and end values zero-padded to two digits, separated by a dash.
    :rtype: str
    """
    return '-'.join(['{:02d}'.format(int(sp_)) for sp_ in spw.split('~')])


def _spw_range_bounds(spw):
    parts = str(spw).split('~')
    if len(parts) == 1:
        start = end = int(parts[0])
    elif len(parts) == 2:
        start, end = (int(part) for part in parts)
    else:
        raise ValueError(f"Invalid SPW range {spw!r}")
    if end < start:
        raise ValueError(f"Invalid descending SPW range {spw!r}")
    return start, end


def _spw_indices_for_range(spw):
    start, end = _spw_range_bounds(spw)
    return ','.join(str(sp) for sp in range(start, end + 1))


def _fine_spectral_spws_for_range(spw):
    """Split one coarse SPW range into 2-band chunks, using a leading 3-band chunk for odd counts."""
    start, end = _spw_range_bounds(spw)
    nspw = end - start + 1
    if nspw <= 3:
        return [f'{start}~{end}']

    chunks = []
    cursor = start
    if nspw % 2:
        chunks.append(f'{cursor}~{cursor + 2}')
        cursor += 3
    while cursor <= end:
        chunks.append(f'{cursor}~{min(cursor + 1, end)}')
        cursor += 2
    return chunks


def _normalize_spw_list(spws):
    """Return validated SPW range strings from CLI or Python inputs."""
    if spws is None:
        return None
    if isinstance(spws, str):
        tokens = spws.replace(',', ' ').split()
    else:
        tokens = [str(spw).strip() for spw in spws]
    normalized = []
    for token in tokens:
        if not token:
            continue
        start, end = _spw_range_bounds(token)
        normalized.append(f'{start}~{end}')
    if not normalized:
        raise ValueError("custom_spws was provided but no valid SPW ranges were found")
    return normalized


def _resolve_fine_spectral_spw_mode(custom_spws, fine_spectral_imaging,
                                    fine_spectral_bootstrap=False):
    """Resolve scratch versus parent-bootstrapped fine-SPW processing.

    Operators historically passed the 24 requested output ranges through
    ``custom_spws`` and self-calibrated every range independently. Preserve
    that safe behavior unless parent bootstrapping is explicitly enabled.
    With both fine imaging and bootstrapping enabled, recognize the exact plan
    as requested child outputs and keep the default seven parent groups.

    :returns: ``(processing_custom_spws, requested_fine_spws)``.
    :rtype: tuple(list(str) or None, list(str) or None)
    """
    normalized = _normalize_spw_list(custom_spws)
    if fine_spectral_imaging and fine_spectral_bootstrap and normalized is not None:
        requested_bounds = [_spw_range_bounds(spw) for spw in normalized]
        exact_bounds = [_spw_range_bounds(spw) for spw in FINE_SPECTRAL_SPWS_52BAND]
        if requested_bounds == exact_bounds:
            return None, normalized
    return normalized, None


def _fine_spectral_children_for_parent(parent_spw, requested_spws=None):
    """Return unique fine ranges strictly contained by one coarse parent."""
    candidates = (_fine_spectral_spws_for_range(parent_spw)
                  if requested_spws is None else _normalize_spw_list(requested_spws))
    parent_start, parent_end = _spw_range_bounds(parent_spw)
    children = []
    seen = set()
    for child in candidates:
        child_start, child_end = _spw_range_bounds(child)
        child_bounds = (child_start, child_end)
        if child_bounds == (parent_start, parent_end):
            continue
        if parent_start <= child_start <= child_end <= parent_end and child_bounds not in seen:
            children.append(child)
            seen.add(child_bounds)
    return children


def _remove_fine_spectral_products(imgoutdir, date_str, fine_spwstr, fits_tag=''):
    """Remove prior products for one fine range before a new attempt or skip."""
    tag = f'.{fits_tag}' if fits_tag else ''
    interval_dates = [date_str]
    try:
        next_date = (datetime.strptime(date_str, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        interval_dates.append(next_date)
    except ValueError:
        pass
    patterns = [
        os.path.join(
            imgoutdir,
            f'eovsa.synoptic{tag}.{interval_date}T??????Z.s{fine_spwstr}.tb*.fits')
        for interval_date in interval_dates
    ]
    patterns.append(os.path.join(
        imgoutdir,
        f'eovsa.synoptic_daily{tag}.{date_str}T200000Z.s{fine_spwstr}.tb*.fits'))
    for pattern in patterns:
        for product in glob(pattern):
            try:
                os.remove(product)
            except OSError:
                log_print('WARNING',
                          f"[fine_bootstrap] failed to remove stale product {product}.\n"
                          f"{traceback.format_exc()}")


def _s00_route_identity(fits_tag=''):
    """Return exact and filesystem-safe identifiers for one s00 route.

    ``fits_tag`` is the exact route identifier available to the imaging
    pipeline.  The namespace includes a digest so tags that sanitize to the
    same text cannot share masks, QA artifacts, or archived source models.

    :param fits_tag: FITS/output tag identifying the calibration route.
    :type fits_tag: str
    :returns: Mapping with exact ``fits_tag``/``route_id`` and safe
        ``namespace`` values.
    :rtype: dict
    """
    exact_tag = str(fits_tag or '')
    route_id = exact_tag or 'production'
    safe_label = re.sub(r'[^A-Za-z0-9._-]+', '-', route_id).strip('._-') or 'route'
    if exact_tag:
        digest = hashlib.sha256(exact_tag.encode('utf-8')).hexdigest()[:12]
        namespace = f'{safe_label[:48]}-{digest}'
    else:
        namespace = 'production'
    return {
        'fits_tag': exact_tag,
        'route_id': route_id,
        'namespace': namespace,
    }


def _remove_s00_route_products(imgoutdir, date_str, spwstr='00-01', fits_tag=''):
    """Remove stale s00 science products for one exact output route.

    The observing day can produce interval files on either UTC date around
    the local-day boundary, so both ``date_str`` and the following UTC date
    are checked.  Products with another ``fits_tag`` are never matched.

    :param imgoutdir: Synoptic FITS output directory.
    :type imgoutdir: str
    :param date_str: Pipeline date in ``YYYYMMDD`` form.
    :type date_str: str
    :param spwstr: Formatted s00 spectral-window label.
    :type spwstr: str
    :param fits_tag: Exact output-route tag.
    :type fits_tag: str
    :returns: Paths successfully removed.
    :rtype: list(str)
    """
    tag = f'.{fits_tag}' if fits_tag else ''
    interval_dates = [str(date_str)]
    try:
        next_date = (
            datetime.strptime(str(date_str), '%Y%m%d') + timedelta(days=1)
        ).strftime('%Y%m%d')
        interval_dates.append(next_date)
    except ValueError:
        pass
    patterns = [
        os.path.join(
            imgoutdir,
            f'eovsa.synoptic{tag}.{interval_date}T??????Z.s{spwstr}.tb*.fits')
        for interval_date in interval_dates
    ]
    patterns.append(os.path.join(
        imgoutdir,
        f'eovsa.synoptic_daily{tag}.{date_str}T200000Z.s{spwstr}.tb*.fits'))

    removed = []
    failures = []
    for product in sorted({path for pattern in patterns for path in glob(pattern)}):
        try:
            os.remove(product)
            removed.append(product)
        except OSError as exc:
            failures.append((product, str(exc)))
            log_print('WARNING',
                      f"[s00_policy] failed to remove stale route product {product}.\n"
                      f"{traceback.format_exc()}")
    if failures:
        failed_paths = ', '.join(path for path, _reason in failures)
        raise RuntimeError(
            f"Unable to withhold stale s00 route product(s): {failed_paths}")
    return removed


def _begin_s00_attempt(s00_policy_by_key, synfits_manifest, sidx,
                       imgoutdir, date_str, spwstr, fits_tag=''):
    """Mark an adaptive s00 attempt before any fallible preparation step.

    Membership in ``s00_policy_by_key`` makes post-processing use only the
    current-run manifest.  Initializing that manifest to empty before policy
    preparation prevents a failed attempt from falling back to the generic
    glob of prior interval FITS.

    :param s00_policy_by_key: Current-run adaptive-s00 policy mapping.
    :type s00_policy_by_key: dict
    :param synfits_manifest: Current-run interval FITS manifest.
    :type synfits_manifest: dict
    :param sidx: Coarse spectral-window index.
    :type sidx: int
    :param imgoutdir: Synoptic FITS output directory.
    :type imgoutdir: str
    :param date_str: Pipeline date in ``YYYYMMDD`` form.
    :type date_str: str
    :param spwstr: Formatted s00 spectral-window label.
    :type spwstr: str
    :param fits_tag: Exact output-route tag.
    :type fits_tag: str
    :returns: Stale same-route products removed before the attempt.
    :rtype: list(str)
    """
    s00_policy_by_key[sidx] = None
    synfits_manifest[sidx] = []
    return _remove_s00_route_products(
        imgoutdir, date_str, spwstr=spwstr, fits_tag=fits_tag)


def _is_default_52_parent_plan(spws):
    """Return whether ``spws`` is the unchanged seven-group 52-band plan."""
    return ([_spw_range_bounds(spw) for spw in spws]
            == [_spw_range_bounds(spw) for spw in SPWS_52BAND])


def _adaptive_s00_enabled(adaptive_s00, spw, tr_series_time):
    """Return whether the explicit adaptive s00 route may be used.

    The historical segmented full-track route remains the default.  The
    adaptive route is deliberately restricted to an explicit opt-in, the
    s00-01 group, and an absent imaging time-range filter.

    :param adaptive_s00: Explicit request for the adaptive s00 route.
    :type adaptive_s00: bool
    :param spw: Spectral-window group or range understood by
        :func:`_spw_range_bounds`.
    :type spw: str or tuple or list
    :param tr_series_time: Parsed imaging time-range filter, or ``None``.
    :type tr_series_time: list or None
    :returns: Whether adaptive s00 imaging is enabled for this group.
    :rtype: bool
    """
    return bool(
        adaptive_s00
        and _spw_range_bounds(spw) == (0, 1)
        and tr_series_time is None
    )


def _write_fine_bootstrap_parent_marker(msfile):
    """Mark an archive as containing the seven trustworthy parent checkpoints."""
    marker = os.path.join(msfile, FINE_BOOTSTRAP_PARENT_MARKER)
    with open(marker, 'w') as marker_file:
        marker_file.write('schema=1\nparents=' + ','.join(SPWS_52BAND) + '\n')
    return marker


def _has_fine_bootstrap_parent_marker(msfile):
    """Return whether an MS was archived from the seven-parent checkpoint path."""
    return os.path.isfile(os.path.join(msfile, FINE_BOOTSTRAP_PARENT_MARKER))


def _spw_id_list(spw_index):
    """Return integer SPW ids from a comma-separated id list or range string."""
    ids = []
    for token in str(spw_index).replace(',', ' ').split():
        start, end = _spw_range_bounds(token)
        ids.extend(range(start, end + 1))
    return sorted(set(ids))


def _data_description_ids_for_spws(msfile, spw_ids):
    """Return DATA_DESC_ID rows whose SPECTRAL_WINDOW_ID is in ``spw_ids``."""
    spw_set = set(int(spw_id) for spw_id in spw_ids)
    data_description_table = os.path.join(msfile, 'DATA_DESCRIPTION')
    tb.open(data_description_table)
    try:
        spw_col = np.asarray(tb.getcol('SPECTRAL_WINDOW_ID')).astype(int).ravel()
        return [ddid for ddid, spw_id in enumerate(spw_col) if int(spw_id) in spw_set]
    finally:
        tb.close()


def _archive_spw_index_remap(msfile, freq_setup):
    """Map original (pre-``split``) SPW indices onto the row indices of an archive MS.

    ``fine_spectral_only``/``imaging_only`` resume from an archive MS built (by a
    prior full run) via per-coarse-group ``split(datacolumn='corrected')`` +
    ``concat``. CASA's ``split`` renumbers SPW ids from 0 in the output, so a
    SCOPED archive (built from a subset of the original coarse groups, e.g.
    ``5~10,31~43,44~49``) ends up with SPW ids ``0..N-1`` that do NOT match the
    ORIGINAL band indices the rest of this module computes from
    :class:`FrequencySetup` (``spws_indices``, ``fine_sp_index``, etc.). A
    full 7-group archive tiles ``0~49`` contiguously, so its renumbering is the
    identity map -- which is why this was never observed before scoped
    archives were introduced.

    This reads the archive's ``SPECTRAL_WINDOW`` subtable and matches each row's
    ``REF_FREQUENCY`` back to the nearest :attr:`FrequencySetup.eofreq` band
    center to recover, for every archive row ``i``, which original band index
    ``j`` it corresponds to.

    :param msfile: Path to the (possibly scoped) archive MS to inspect.
    :type msfile: str
    :param freq_setup: Active :class:`FrequencySetup` instance providing the
        original per-band center frequencies (``eofreq``, GHz) and band width
        (``bandwidth``, GHz).
    :type freq_setup: FrequencySetup
    :returns: Mapping ``{original_index: archive_row_index}``, or ``None`` if
        no remapping is needed (identity map).
    :rtype: dict or None
    :raises RuntimeError: If the mapping cannot be determined unambiguously.
        Aborting is deliberate: falling back to identity on a scoped archive
        would silently image the wrong SPW rows (all-NaN/garbage products).
    """
    spw_table = msfile + '/SPECTRAL_WINDOW'
    tb.open(spw_table)
    try:
        ref_freqs = np.asarray(tb.getcol('REF_FREQUENCY')).ravel()
    finally:
        tb.close()

    eofreq = np.asarray(freq_setup.eofreq)
    bandwidth = float(freq_setup.bandwidth)

    remap = {}
    for i, ref_freq_hz in enumerate(ref_freqs):
        # EOVSA MS SPW REF_FREQUENCY is the band's LOWER EDGE (== CHAN_FREQ[0]),
        # while FrequencySetup.eofreq holds band CENTERS — offset by exactly
        # bandwidth/2 (verified on a real archive: eofreq lands midway between
        # adjacent REF_FREQUENCY values, making raw nearest-match ambiguous).
        row_center_ghz = float(ref_freq_hz) / 1e9 + bandwidth / 2.0
        diffs = np.abs(eofreq - row_center_ghz)
        j = int(np.argmin(diffs))
        min_diff = float(diffs[j])
        if min_diff >= bandwidth / 2.0:
            raise RuntimeError(
                f"[archive_spw_remap] archive row {i} (band center "
                f"{row_center_ghz:.4f} GHz) does not match any original band "
                f"center within {bandwidth / 2.0:.4f} GHz "
                f"(closest: band {j} at {eofreq[j]:.4f} GHz, "
                f"diff={min_diff:.4f} GHz) in {msfile!r}. Aborting resume: "
                f"identity fallback would image the wrong SPW rows.")
        if j in remap:
            raise RuntimeError(
                f"[archive_spw_remap] ambiguous remap for {msfile!r}: original "
                f"band {j} matches both archive row {remap[j]} and row {i}. "
                f"Aborting resume: identity fallback would image the wrong SPW rows.")
        remap[j] = i

    n_rows = len(ref_freqs)
    if all(j == remap.get(j) for j in range(n_rows)) and len(remap) == n_rows:
        log_print('INFO',
                  f"[archive_spw_remap] {msfile}: {n_rows} row(s), identity mapping "
                  f"(no SPW remapping needed).")
        return None

    # Summarize as compact "orig a-b -> rows c-d" ranges for one INFO line.
    ordered_js = sorted(remap.keys())
    ranges = []
    run_start_j = ordered_js[0]
    run_start_i = remap[run_start_j]
    prev_j = run_start_j
    prev_i = run_start_i
    for j in ordered_js[1:]:
        i = remap[j]
        if j == prev_j + 1 and i == prev_i + 1:
            prev_j, prev_i = j, i
            continue
        ranges.append((run_start_j, prev_j, run_start_i, prev_i))
        run_start_j, run_start_i = j, i
        prev_j, prev_i = j, i
    ranges.append((run_start_j, prev_j, run_start_i, prev_i))
    summary = '; '.join(
        f"orig {j0}-{j1} -> rows {i0}-{i1}" if j0 != j1 else f"orig {j0} -> row {i0}"
        for j0, j1, i0, i1 in ranges
    )
    log_print('INFO',
              f"[archive_spw_remap] {msfile}: {n_rows} row(s), remapping ({summary}).")
    return remap


def _remap_sp_index_str(sp_index, remap):
    """Remap a comma-separated string of original SPW indices to archive-row indices.

    :param sp_index: Comma-separated original SPW indices (as produced by
        :func:`_spw_indices_for_range`), e.g. ``'5,6,7,8,9,10'``.
    :type sp_index: str
    :param remap: Mapping ``{original_index: archive_row_index}`` as returned by
        :func:`_archive_spw_index_remap`.
    :type remap: dict
    :returns: Comma-separated archive-row indices in the same order as the
        input, or ``None`` if any input index is missing from ``remap``.
    :rtype: str or None
    """
    orig_indices = [int(tok) for tok in str(sp_index).split(',')]
    missing = [j for j in orig_indices if j not in remap]
    if missing:
        log_print('ERROR',
                  f"[archive_spw_remap] original SPW index(es) {missing} not present in "
                  f"the archive SPW remap; cannot remap sp_index={sp_index!r}.")
        return None
    return ','.join(str(remap[j]) for j in orig_indices)


def _unflagged_row_mask(flag_values, nrow):
    """Return rows with at least one unflagged correlation/channel sample."""
    flags = np.asarray(flag_values, dtype=bool)
    if flags.ndim == 0:
        return np.full(nrow, not bool(flags), dtype=bool)
    if flags.shape[-1] == nrow:
        return np.any(~flags.reshape(-1, nrow), axis=0)
    if flags.shape[0] == nrow:
        return np.any(~flags.reshape(nrow, -1), axis=1)
    if flags.size == nrow:
        return ~flags.reshape(nrow)
    return np.full(nrow, bool(np.any(~flags)), dtype=bool)


def _caltable_has_unflagged_solutions(caltable):
    """Return whether a CASA calibration table contains a usable solution row."""
    if not os.path.exists(caltable):
        return False
    try:
        tb.open(caltable)
        try:
            nrow = tb.nrows()
            if nrow < 1 or 'FLAG' not in tb.colnames():
                return False
            return bool(np.any(_unflagged_row_mask(tb.getcol('FLAG'), nrow)))
        finally:
            tb.close()
    except Exception:
        log_print('WARNING',
                  f"[fine_bootstrap] could not validate phase solutions in {caltable}.\n"
                  f"{traceback.format_exc()}")
        return False


def unflagged_solar_antenna_ids(msfile, spw_index, n_ant_total, row_chunk=4096):
    """Return one-based solar antennas with unflagged data in the imaging SPWs.

    Antenna indices are CASA zero-based.  Solar antennas are indices
    ``0..n_ant_total-1``; the 27-m calibration antenna is intentionally excluded.
    """
    if n_ant_total is None:
        return None
    try:
        ddids = _data_description_ids_for_spws(msfile, _spw_id_list(spw_index))
        if not ddids:
            return None

        available = set()
        tb.open(msfile)
        try:
            for ddid in ddids:
                subtable = tb.query('DATA_DESC_ID == {}'.format(int(ddid)))
                try:
                    nrow = subtable.nrows()
                    for startrow in range(0, nrow, row_chunk):
                        nread = min(row_chunk, nrow - startrow)
                        if nread <= 0:
                            continue
                        ant1 = np.asarray(subtable.getcol('ANTENNA1', startrow, nread), dtype=int)
                        ant2 = np.asarray(subtable.getcol('ANTENNA2', startrow, nread), dtype=int)
                        flags = subtable.getcol('FLAG', startrow, nread)
                        unflagged_rows = _unflagged_row_mask(flags, nread)
                        for ant in np.concatenate([ant1[unflagged_rows], ant2[unflagged_rows]]):
                            ant = int(ant)
                            if 0 <= ant < int(n_ant_total):
                                available.add(ant)
                finally:
                    subtable.close()
        finally:
            tb.close()
        return [ant + 1 for ant in sorted(available)]
    except Exception:
        log_print('WARNING',
                  f"Unable to count unflagged solar antennas for imaging label: "
                  f"{_format_log_state(msfile=msfile, spw_index=spw_index, n_ant_total=n_ant_total)}\n"
                  f"{traceback.format_exc()}")
        return None


def count_unflagged_solar_antennas(msfile, spw_index, n_ant_total, row_chunk=4096):
    """Count solar antennas with at least one unflagged sample in the imaging SPWs."""
    antenna_ids = unflagged_solar_antenna_ids(
        msfile, spw_index, n_ant_total, row_chunk=row_chunk)
    return None if antenna_ids is None else len(antenna_ids)


def _default_config_index_for_spw(spw, default_spws):
    """Map a custom SPW range to the overlapping default coarse-band config."""
    start, end = _spw_range_bounds(spw)
    best_idx = 0
    best_overlap = -1
    for idx, default_spw in enumerate(default_spws):
        default_start, default_end = _spw_range_bounds(default_spw)
        overlap = max(0, min(end, default_end) - max(start, default_start) + 1)
        if overlap > best_overlap:
            best_idx = idx
            best_overlap = overlap
    return best_idx


class FrequencySetup:
    """
    Manages frequency setup based on observation date for radio astronomy imaging.

    This class is initialized with an observation time and calculates
    essential frequency parameters such as effective observing frequencies (eofreq)
    and spectral windows (spws) based on the observation date. It provides methods
    to calculate reference frequency and bandwidth for given spectral windows.

    :param tim: Observation time used to determine the frequency setup.
    :type tim: astropy.time.Time

    Attributes:
    - tim (astropy.time.Time): The observation time.
    - spw2band (numpy.ndarray): An array mapping spectral window indices to band numbers.
    - bandwidth (float): The bandwidth in GHz.
    - defaultfreq (numpy.ndarray): The default effective observing frequencies in GHz.
    - nbands (int): Number of bands.
    - eofreq (numpy.ndarray): Effective observing frequencies based on the observation time.
    - spws (list of str): Spectral window selections based on the observation time.

    :Example:

    >>> from astropy.time import Time
    >>> tim = Time('2022-01-01T00:00:00', format='isot')
    >>> freq_setup = FrequencySetup(tim)
    >>> crval, cdelt = freq_setup.get_reffreq_and_cdelt('5~10')
    >>> print(crval, cdelt)
    """

    def __init__(self, tim=None, spws=None):
        if tim is None:
            tim = Time.now()
        self.tim = tim
        self.spw2band = np.array([0, 1] + list(range(4, 52)))
        self.bandwidth = 0.325  # 325 MHz
        self.defaultfreq = 1.1 + self.bandwidth * (self.spw2band + 0.5)

        if self.tim.mjd > SPW_EPOCH_SPLIT_MJD:
            self.eofreq = self.defaultfreq
            self.spws = list(SPWS_52BAND)
            self.nbands = len(self.spw2band)
        else:
            self.bandwidth = 0.5  # 500 MHz
            self.nbands = 34
            self.eofreq = 1.419 + np.arange(self.nbands) * self.bandwidth
            self.spws = list(SPWS_34BAND)

        self.default_spws = list(self.spws)
        custom_spws = _normalize_spw_list(spws)
        if custom_spws is not None:
            for spw in custom_spws:
                start, end = _spw_range_bounds(spw)
                if start < 0 or end >= self.nbands:
                    raise ValueError(
                        f"Custom SPW range {spw!r} is outside the valid 0~{self.nbands - 1} range")
            self.spws = custom_spws
        self.spw_config_indices = [
            _default_config_index_for_spw(spw, self.default_spws)
            for spw in self.spws
        ]

        self.spws_indices = []
        for sp in self.spws:
            s1, s2 = sp.split('~')
            self.spws_indices.append(','.join(map(str, range(int(s1), int(s2) + 1))))

    def get_reffreq_and_cdelt(self, spw, return_bmsize=False):
        """
        Calculates the reference frequency (CRVAL) and the frequency delta (CDELT)
        for a given spectral window range.

        This method takes a spectral window selection and computes the mean of the effective
        observing frequencies (eofreq) within that range as the reference frequency. It also
        calculates the bandwidth covered by the spectral window range as the frequency delta.

        :param spw: Spectral window selection, specified as a range 'start~end' or a single value.
        :type spw: str
        :param return_bmsize: If True, return the beam size based on the reference frequency. Defaults to False.
        :type return_bmsize: bool

        :return: A tuple containing the reference frequency and frequency delta, both in GHz.
        :rtype: (str, str)

        :Example:

        >>> crval, cdelt = freq_setup.get_reffreq_and_cdelt('5~10')
        >>> print(crval, cdelt)
        """
        sp_st, sp_ed = (int(s) for s in spw.split('~')) if '~' in spw else (int(spw), int(spw))
        crval = f'{np.mean(self.eofreq[sp_st:sp_ed + 1]):.4f}GHz'
        nband = (self.eofreq[sp_ed] - self.eofreq[sp_st]) / self.bandwidth + 1
        cdelt = f'{nband * self.bandwidth:.4f}GHz'

        if return_bmsize:
            bmsz = self.get_bmsize(float(crval.rstrip('GHz')), minbmsize=5.0)
            return crval, cdelt, bmsz
        else:
            return crval, cdelt

    def get_bmsize(self, cfreq, refbmsize=75.0, reffreq=1.0, minbmsize=4.0):
        """
        Calculate the beam size at given frequencies based on a reference beam size at a reference frequency.
        This function supports both single frequency values and lists of frequencies.

        :param cfreq: Input frequencies in GHz, can be a float or a list of floats.
        :type cfreq: float or list
        :param refbmsize: Reference beam size in arcsec, defaults to 60.0.
        :type refbmsize: float, optional
        :param reffreq: Reference frequency in GHz, defaults to 1.0.
        :type reffreq: float, optional
        :param minbmsize: Minimum beam size in arcsec, defaults to 4.0.
        :type minbmsize: float, optional
        :return: Beam size at the given frequencies, same type as input (float or numpy array).
        :rtype: float or numpy.ndarray
        """
        # Ensure cfreq is an array for uniform processing
        cfreq = np.array(cfreq, dtype=float)

        # Calculate beam size
        bmsize = refbmsize * reffreq / cfreq

        # Enforce minimum beam size
        bmsize = np.maximum(bmsize, minbmsize)

        # If the original input was a single float, return a single float
        if bmsize.size == 1:
            return bmsize.item()  # Convert numpy scalar to Python float
        else:
            return bmsize


def merge_FITSfiles(fitsfilesin, outfits, snr_weight=None, deselect_index=None,
                    overwrite=True, snr_threshold=None, rms_threshold=None, showplt=False,
                    reftime=None, band_stripe_threshold=12.0):
    """
    Merges multiple FITS files into a single output file by calculating the mean of stacked data.

    This function stacks the data from a list of input FITS files along a new axis, optionally applying exposure time
    weighting and suppression of on-disk residuals based on a threshold relative to the disk brightness temperature.
    The mean of the stacked data is then written to a specified output FITS file.

    Parameters
    ----------
    fitsfilesin : list of str
        List of file paths to the input FITS files to be merged.
    outfits : str
        File path for the output FITS file where the merged data will be saved.
    snr_weight : list of float, optional
        List of signal-to-noise ratios (SNRs) to use as weights for calculating the mean of the stacked data.
        If both exposure time and SNR weights are provided, the weights will be the product of the two. Defaults to None.
    deselect_index : list of int, optional
        index of the images to be deselected. Defaults to None.
    overwrite : bool, optional
        If True, the output file is overwritten if it already exists. Defaults to True.
    snr_threshold : float, optional
        SNR threshold for selecting images to be deselected. Defaults to 10.
    reftime : astropy.time.Time or str, optional
        Reference epoch stamped into the merged header's DATE-OBS/DATE-AVG.
        If None, defaults to 20:00:00 UT on the observing date taken from the
        first input header.
    band_stripe_threshold : float, optional
        Stripe score above which the *merged* daily diskless image is flagged as
        artifact dominated. Per-frame stripe scores cannot separate a genuinely
        ruined high band from a clean one (beam elongation inflates both), but
        coherent striping reinforces in the merge while incoherent residuals
        average down, so the merged image is the reliable band-level
        discriminator. When exceeded, a WARNING is logged and the merged header
        records ``STRPSCOR``/``STRPDOM``; disk addition is not changed.
        Defaults to 12.0.

    Raises
    ------
    ValueError
        If any of the input FITS files cannot be read.

    Returns
    -------
    None

    Notes
    -----
    The suppression of on-disk residuals is achieved by applying a sigmoid function to pixels within a specified range
    of the disk brightness temperature, effectively reducing the contribution of residuals while preserving the overall
    structure.

    Time convention: the input images have been differentially rotated to a
    common daily reference epoch, so the merged header's DATE-OBS and DATE-AVG
    are set to that epoch (``reftime``; default 20:00 UT on the observing
    date) to keep the timestamp consistent with the WCS for coalignment.
    The true observing window is preserved in STARTOBS/ENDOBS, and EXPTIME
    records the total integration time summed over the merged images.

    Examples
    --------
    Merge three FITS files without exposure time weighting and with on-disk residual suppression:

    >>> merge_FITSfiles(['sun_01.fits', 'sun_02.fits', 'sun_03.fits'], 'sun_merged.fits',
    ...                 overwrite=True)
    """
    from astropy.io import fits
    from astropy.time import Time

    exptimes = []
    startobs_list = []
    endobs_list = []
    data = []
    for fidx, file in enumerate(fitsfilesin):
        with fits.open(file) as hdulist:
            for idx, hdu in enumerate(hdulist):
                if 'CDELT1' in hdu.header:
                    header = hdu.header
                    data.append(np.squeeze(hdu.data))
                    exptimes.append(header['EXPTIME'])
                    if 'STARTOBS' in header:
                        startobs = Time(header['STARTOBS'])
                        startobs_list.append(startobs)
                        endobs = header.get('ENDOBS')
                        if endobs is None:
                            endobs = (startobs + float(header['EXPTIME']) * u.s).isot
                        endobs_list.append(Time(endobs))
                    break
    data_stack = np.array(data)
    if deselect_index is None:
        rsun_pix = header['RSUN_OBS'] / header['CDELT1']
        crpix1, crpix2 = header['CRPIX1'], header['CRPIX2']
        select_indices, rms, snr = detect_noisy_images(data_stack, rsun_pix=rsun_pix, crpix1=crpix1, crpix2=crpix2,
                                                       rsun_ratio=[1.2, 1.3], showplt=showplt,
                                                       snr_threshold=snr_threshold, rms_threshold=rms_threshold)
        deselect_index = ~select_indices
        snr_weight = snr
        print(snr)
    if snr_weight is not None:
        weights = np.array(snr_weight)
    else:
        weights = np.ones(data_stack.shape[0])
    weights[deselect_index] = 0
    weights = weights / np.sum(weights)
    weights = weights[:, np.newaxis, np.newaxis]
    date_merged = np.nansum(data_stack * weights, axis=0)
    newheader = header.copy()
    exptime = np.nansum(exptimes)
    if reftime is None:
        # Daily reference epoch: 20:00 UT on the observing date. The inputs
        # were differentially rotated to this epoch, so DATE-OBS must match
        # it (not the integration mid-point) for time-based coalignment.
        reftime = Time(Time(header['date-obs']).datetime.replace(
            hour=20, minute=0, second=0, microsecond=0))
    reftime_str = Time(reftime).isot
    newheader.set('EXPTIME', exptime, 'total integration time of merged images [s]')
    newheader.set('DATE-OBS', reftime_str, 'daily reference epoch; see STARTOBS/ENDOBS')
    newheader.set('DATE-AVG', reftime_str, 'daily reference epoch (WCS)')
    if startobs_list:
        newheader.set('STARTOBS', min(startobs_list).isot, 'start of true observing window')
    if endobs_list:
        newheader.set('ENDOBS', max(endobs_list).isot, 'end of true observing window')
    newheader['HISTORY'] = 'Merged from multiple images'
    newheader['HISTORY'] = 'DATE-OBS set to daily reference epoch; acquisition window in STARTOBS/ENDOBS'

    # Band-level artifact-dominated guard on the MERGED diskless image.
    # Per-frame stripe scores overlap between ruined high bands and clean ones
    # (dirty-beam elongation rises at track endpoints and toward high
    # frequency), so they cannot separate the two. Coherent striping, however,
    # reinforces under the merge while incoherent residuals average down, so the
    # merged image is the reliable discriminator. This only flags/logs; it does
    # not alter disk addition or delete the product.
    try:
        merged_stripe_score = float(detect_stripe_artifacts(date_merged[np.newaxis])[1][0])
    except Exception:
        merged_stripe_score = float('nan')
    artifact_dominated = bool(np.isfinite(merged_stripe_score) and
                              merged_stripe_score > band_stripe_threshold)
    newheader.set('STRPSCOR', merged_stripe_score if np.isfinite(merged_stripe_score) else -1.0,
                  'merged diskless directional stripe score')
    newheader.set('STRPDOM', artifact_dominated,
                  'merged image flagged artifact/stripe dominated')

    fits.writeto(outfits, date_merged, newheader, overwrite=overwrite)
    log_print('INFO',
              f'{np.count_nonzero(deselect_index)} out of {len(fitsfilesin)} images (index{np.where(deselect_index)}) are not selected for merging due to low SNR.')
    if artifact_dominated:
        log_print('WARNING',
                  f'Merged daily image {os.path.basename(outfits)} is ARTIFACT/STRIPE DOMINATED: '
                  f'merged stripe score={merged_stripe_score:.1f} > {band_stripe_threshold:.1f}. '
                  f'This band/day daily synoptic is unreliable; coherent striping survives the '
                  f'merge and per-frame deselection cannot recover it (likely upstream '
                  f'high-band calibration/imaging quality).')
    else:
        log_print('INFO',
                  f'Merged daily stripe score for {os.path.basename(outfits)}='
                  f'{merged_stripe_score:.1f} (thr={band_stripe_threshold:.1f}).')
    return outfits


class MSselfcal:
    """
    A class for imaging using WSClean and selfcal visibility data.

    This class facilitates the creation of images by imaging major
    intervals for WSClean imaging and applying solar differential rotation corrections to minor intervals (new model).
    the model will be written to the visibility data for selfcalibration.

    :param msfile: Path to the Measurement Set (MS) file.
    :type msfile: str
    :param time_intervals: List of major time intervals.
    :type time_intervals: list
    :param time_intervals_major_avg: Average times for major intervals.
    :type time_intervals_major_avg: list
    :param time_intervals_minor_avg: Average times for minor intervals within major intervals.
    :type time_intervals_minor_avg: list
    :param spw: Spectral window index or range.
    :type spw: str
    :param sp_index: String representation of the spectral window indices.
    :type sp_index: str
    :param N1: Number of major intervals.
    :type N1: int
    :param N2: Number of minor intervals within each major interval.
    :type N2: int
    :param workdir: Directory for storing intermediate and output files.
    :type workdir: str
    :param image_marker: Optional marker to add to output image names, defaults to ''
    :type image_marker: str, optional
    :param niter: Maximum number of iterations for WSClean, defaults to 5000.
    :type niter: int, optional
    :param data_column: Column name in the MS file to process, defaults to "CORRECTED_DATA".
    :type data_column: str, optional
    :param beam_size: Beam size for imaging, defaults to None.
    :type beam_size: float, optional
    :param reftime_daily: Reference time for daily rotation corrections, defaults to None.
    :type reftime_daily: `astropy.Time`, optional

    :raises OSError: If any file operation fails.
    :raises Exception: For errors during heliocentric frame registration or rotation corrections.

    :return: Path to the final model name string.
    :rtype: str
    """

    def __init__(self,
                 msfile,
                 time_intervals,
                 time_intervals_major_avg,
                 time_intervals_minor_avg,
                 spw,
                 N1,
                 N2,
                 workdir,
                 image_marker='',
                 niter=5000,
                 gain=0.1,
                 briggs=0.0,
                 auto_mask=6,
                 auto_threshold=3,
                 maxuv_l=None,
                 minuv_l=None,
                 maxuvw_m=None,
                 minuvw_m=None,
                 data_column="CORRECTED_DATA",
                 pols='XX',
                 beam_size=None,
                 circular_beam=True,
                 no_negative=True,
                 fits_mask=None,
                 theoretic_beam=False,
                 reftime_daily=None,
                 sp_index=None):
        self.msfile = msfile
        self.time_intervals = time_intervals
        self.time_intervals_major_avg = time_intervals_major_avg
        self.time_intervals_minor_avg = time_intervals_minor_avg
        self.spw = str(spw)
        if sp_index is not None:
            self.sp_index = str(sp_index)
        elif '~' in spw:
            try:
                start_str, end_str = spw.split('~')
                start = int(start_str)
                end = int(end_str)
                self.sp_index = ','.join(str(i) for i in range(start, end + 1))
            except Exception as e:
                raise ValueError(f"Invalid spw range format: '{spw}'. Expected format 'start~end'.") from e
        else:
            self.sp_index = spw
        self.N1 = N1
        self.N2 = N2
        self.workdir = workdir
        self.image_marker = image_marker
        self.niter = niter
        self.gain = gain
        self.pols = pols
        self.data_column = data_column
        self.beam_size = beam_size
        self.reftime_daily = reftime_daily
        self.model_minor_name_str = None
        self.model_ref_name_str = None
        self.briggs = briggs
        self.maxuv_l = maxuv_l
        self.minuv_l = minuv_l
        self.maxuvw_m = maxuvw_m
        self.minuvw_m = minuvw_m
        self.auto_mask = auto_mask
        self.auto_threshold = auto_threshold
        self.succeeded = True
        self.no_negative = no_negative
        self.circular_beam = circular_beam
        self.fits_mask = fits_mask
        self.theoretic_beam = theoretic_beam

        # self.intervals_out = None

    @ww.runtime_report
    def run(self, imaging_only=False, predict=True, gen_minor_model=True, clearcache=True):
        """
        Processes the visibility data and generates model images using WSClean.

        This method performs the following steps:
        1. Initial imaging to generate rough model images.
        2. Rotate each model image to the corresponding minor time intervals. This step converts N1 images to N1*N2 images.
           This model will be used for self-calibration and will be subtracted from the visibility data after self-calibration.
        3. Rotate each model image to 20 UT of the day (if the reference time daily is provided; otherwise, skip this step). This model will be added back to the visibility data.
        4. Predict visibilities based on the rotated models.

        :param imaging_only: If True, only perform the initial imaging step and skip the rotation and prediction steps. Default is False.
        :type imaging_only: bool
        :param predict: If True, predict visibilities based on the rotated models. Default is True.
        :type predict: bool
        :param gen_minor_model: If True, generate model images for minor time intervals. Default is True. If False, skip the rotation and prediction steps.
        :type gen_minor_model: bool

        :raises OSError: If any file operation fails.
        :raises Exception: For errors during heliocentric frame registration or rotation corrections.

        :return: Path to the final model name string.
        :rtype: str
        """
        if not gen_minor_model:
            predict = False
        spwstr = format_spw(self.spw)
        msfilename = os.path.basename(self.msfile)
        imname_strlist = ["eovsa", "major", f"{msfilename}", f"sp{spwstr}"]
        if self.image_marker:
            imname_strlist.append(self.image_marker)
        imname = '-'.join(imname_strlist)
        imname_minor = imname.replace('major', 'minor')

        for f in glob(os.path.join(self.workdir, f"{imname}*.fits")):
            os.remove(f)
        clean_obj = ww.WSClean(self.msfile)
        clean_obj.setup(size=1024, scale="2.5asec", weight_briggs=self.briggs, pol=self.pols,
                        niter=self.niter,
                        mgain=0.85,
                        gain=self.gain,
                        data_column=self.data_column,
                        name=os.path.join(self.workdir, imname),
                        multiscale=True,
                        multiscale_scale_bias=0.6,
                        # multiscale_gain=0.3,
                        maxuv_l=self.maxuv_l, minuv_l=self.minuv_l,
                        maxuvw_m=self.maxuvw_m, minuvw_m=self.minuvw_m,
                        auto_mask=self.auto_mask,
                        auto_threshold=self.auto_threshold,
                        fits_mask=self.fits_mask,
                        intervals_out=self.N1,
                        no_negative=self.no_negative, quiet=True,
                        spws=self.sp_index,
                        beam_size=self.beam_size,
                        theoretic_beam=self.theoretic_beam,
                        circular_beam=self.circular_beam)
        clean_obj.run(dryrun=False)

        if clearcache:
            extensions = ['dirty', 'psf', 'residual']
            for ext in extensions:
                for f in glob(os.path.join(self.workdir, f"{imname}-*{ext}.fits")):
                    if os.path.exists(f):
                        os.remove(f)
        if imaging_only:
            return
        # Step 2: Rotate each model image to the corresponding minor time intervals
        if self.N1 == 1:
            fitsname = self.workdir + f"{imname}-t0000-model.fits"
            os.rename(fitsname.replace('-t0000', ''), fitsname)
            modelfitsfiles = [fitsname]
        else:
            modelfitsfiles = sorted(glob(self.workdir + f"{imname}-t*-model.fits"))

        model_dir = os.path.join(self.workdir,
                                 f"modelrot_{msfilename}_{self.image_marker}") if self.image_marker else os.path.join(
            self.workdir, f"modelrot_{msfilename}")

        if not os.path.exists(model_dir): os.makedirs(model_dir)

        self.model_minor_name_str = os.path.join(model_dir, imname_minor)
        for f in glob(f"{self.model_minor_name_str}*.fits"):
            os.remove(f)
        if self.reftime_daily is not None:
            self.model_ref_name_str = os.path.join(model_dir, f"{imname}-ref_daily")
        for idx_major, fitsname in enumerate(modelfitsfiles):
            timerange_ = trange2timerange([self.time_intervals[idx_major][0], self.time_intervals[idx_major][-1]])
            heliofitsname = fitsname.replace("model.fits", "model.helio.fits")
            try:
                hf.imreg(vis=self.msfile, imagefile=fitsname, fitsfile=heliofitsname, timerange=timerange_, )
            except Exception as e:
                log_print('ERROR', f'Failed to register {fitsname} to heliocentric frame due to {e}')
                self.succeeded = False
                continue

            if self.reftime_daily is not None:
                log_print('INFO',
                          f"Rotating {fitsname} to daily reftime at {self.reftime_daily.datetime.strftime('%H:%M:%S')} UT...")
                heliorotname_ref_daily = fitsname.replace("model.fits", "model.helio.ref_daily.fits")
                solar_diff_rot_heliofits(heliofitsname, self.reftime_daily,
                                         heliorotname_ref_daily, in_time=self.time_intervals_major_avg[idx_major])
                new_model_file_ref_daily = f"{self.model_ref_name_str}-t{idx_major:04d}-model.fits"
                sunpyfits_to_j2000fits(heliorotname_ref_daily, new_model_file_ref_daily,
                                       template_fits=fitsname, overwrite_prev=True)

            if gen_minor_model:
                log_print('INFO', f"Rotating {fitsname} to {self.N2} minor time intervals ...")
                for idx_minor, time_interval_minor_avg in enumerate(self.time_intervals_minor_avg[idx_major]):
                    heliorotname = heliofitsname.replace('major', 'minor')
                    heliorotname = heliorotname.replace("model.helio.fits",
                                                        f"t{idx_major * self.N2 + idx_minor:04d}-model.helio.rot.fits")
                    heliorotname = os.path.join(model_dir, os.path.basename(heliorotname))
                    solar_diff_rot_heliofits(heliofitsname, time_interval_minor_avg,
                                             heliorotname, in_time=self.time_intervals_major_avg[idx_major])

                    new_model_file = f"{self.model_minor_name_str}-t{idx_major * self.N2 + idx_minor:04d}-model.fits"
                    sunpyfits_to_j2000fits(heliorotname, new_model_file,
                                           template_fits=fitsname, overwrite_prev=True)

        if predict:
            # Step 3: Predict visibilities for each model image
            cmd = f"{WSCLEAN_BIN} -predict -reorder -spws {self.sp_index} -pol {self.pols} -name {self.model_minor_name_str} -quiet -intervals-out {self.N1 * self.N2} {self.msfile}"
            log_print('INFO', f"Running wsclean predict: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
        return


def clean_junk(imname, junks=['dirty', 'model', 'psf', 'residual']):
    """Clears junk files generated during the imaging process.
    :param imname: The base name of the image files to clean.
    :type imname: str
    :param junks: List of junk file suffixes to remove, defaults to ['dirty', 'model', 'psf', 'residual'].
    :type junks: list of str, optional
    """

    clnjunks = junks
    for clnjunk in clnjunks:
        file2remove = glob(f'{imname}*{clnjunk}.fits')
        if len(file2remove) > 0:
            for f in file2remove:
                if os.path.exists(f):
                    os.remove(f)


def _central_wsclean_interval(wsclean_intervals, reftime, duration_hours,
                              min_coverage_fraction=0.90, max_gap_factor=5.0):
    """Build one end-exclusive WSClean interval around the daily reference time.

    :param wsclean_intervals: Time-grid manager for the active measurement set.
    :type wsclean_intervals: WSCleanTimeIntervals
    :param reftime: Desired center of the synthesis window.
    :type reftime: astropy.time.Time
    :param duration_hours: Requested full window duration in hours.
    :type duration_hours: float
    :param min_coverage_fraction: Minimum sampled/expected timestamp fraction.
    :type min_coverage_fraction: float
    :param max_gap_factor: Largest accepted internal gap in nominal cadences.
    :type max_gap_factor: float
    :returns: Candidate dictionary containing ``valid``, ``interval``, ``ri``,
        coverage metrics, and an explanatory ``reason``.
    :rtype: dict
    """
    tim = Time(wsclean_intervals.tim)
    center = Time(reftime)
    duration_hours = float(duration_hours)
    mjd = np.asarray(tim.mjd, dtype=float)
    result = s00qa.central_time_indices(
        mjd, center.mjd, duration_hours,
        min_coverage_fraction=min_coverage_fraction,
        max_gap_factor=max_gap_factor)
    result['ri'] = None
    selected = np.asarray(result.pop('indices'), dtype=int)
    if selected.size < 2:
        return result

    selected_time = tim[selected]
    midpoint = Time([float(np.nanmean(selected_time.mjd))], format='mjd')
    timerange = (
        selected_time[0].datetime.strftime('%Y/%m/%d/%H:%M:%S') + '~' +
        selected_time[-1].datetime.strftime('%Y/%m/%d/%H:%M:%S')
    )
    ri = {
        'N1': 1,
        'N2': 1,
        'time_intervals': [selected_time],
        'time_intervals_major_avg': midpoint,
        'time_intervals_minor_avg': [midpoint],
        'timeranges': [timerange],
    }
    result.update({
        'ri': ri,
    })
    return result


def _s00_source_coordinates(model_fits, reftime, peak_fraction=0.02,
                            threshold_sigma=7.0):
    """Return helioprojective centroids of strong positive init-model islands.

    The input is the shallow, long-integration CLEAN model at the daily
    reference epoch.  When its WCS or support cannot be interpreted, a disk
    center coordinate is returned so the subsequent Howard-motion gate remains
    conservative and deterministic.

    :param model_fits: Path to the native WSClean model FITS.
    :type model_fits: str
    :param reftime: Daily reference time for the helioprojective frame.
    :type reftime: astropy.time.Time
    :param peak_fraction: Minimum positive component relative to model peak.
    :type peak_fraction: float
    :param threshold_sigma: Robust significance threshold for nonzero support.
    :type threshold_sigma: float
    :returns: One or more helioprojective source coordinates.
    :rtype: astropy.coordinates.SkyCoord
    """
    center = Time(reftime)
    fallback = SkyCoord(
        0.0 * u.arcsec, 0.0 * u.arcsec,
        frame=Helioprojective(observer='earth', obstime=center),
    )
    if not model_fits or not os.path.exists(model_fits):
        return fallback
    try:
        with fits.open(model_fits, memmap=False) as hdul:
            image_hdu = next(hdu for hdu in hdul if hdu.data is not None and 'CDELT1' in hdu.header)
            data = np.asarray(image_hdu.data, dtype=float).squeeze()
            header = image_hdu.header.copy()
        if data.ndim != 2 or not np.any(np.isfinite(data)):
            return fallback
        finite = data[np.isfinite(data)]
        positive = finite[finite > 0]
        if positive.size == 0:
            return fallback
        median = float(np.nanmedian(finite))
        sigma = float(1.4826 * np.nanmedian(np.abs(finite - median)))
        peak = float(np.nanmax(positive))
        threshold = float(peak_fraction) * peak
        if np.isfinite(sigma) and sigma > 0:
            threshold = max(threshold, median + float(threshold_sigma) * sigma)
        seed = np.isfinite(data) & (data >= threshold)
        labels, nlabels = ndimage.label(seed)
        if nlabels < 1:
            return fallback

        centroids = []
        for label_id in range(1, nlabels + 1):
            island = labels == label_id
            weights = np.where(island, np.clip(data, 0.0, None), 0.0)
            if not np.any(weights > 0):
                continue
            y, x = ndimage.center_of_mass(weights)
            if np.isfinite(x) and np.isfinite(y):
                centroids.append((float(x), float(y)))
        if not centroids:
            return fallback

        xpix = np.asarray([item[0] for item in centroids], dtype=float)
        ypix = np.asarray([item[1] for item in centroids], dtype=float)
        # A registered helioprojective model carries the observer and solar
        # distance needed by SunPy.  Converting a native RA/Dec WSClean image
        # directly would yield unit-spherical ICRS coordinates and an invalid
        # heliocentric origin shift.
        model_map = smap.Map(data, header)
        hpc = model_map.pixel_to_world(xpix * u.pix, ypix * u.pix)
        hpc = _solar_rotate_howard(hpc, center)
        tx = np.asarray(hpc.Tx.to_value(u.arcsec), dtype=float)
        ty = np.asarray(hpc.Ty.to_value(u.arcsec), dtype=float)
        rsun = float(sun.angular_radius(center).to_value(u.arcsec))
        good = np.isfinite(tx) & np.isfinite(ty) & (np.hypot(tx, ty) <= 1.05 * rsun)
        return hpc[good] if np.any(good) else fallback
    except Exception:
        log_print('WARNING',
                  f"[s00_window] could not derive source coordinates from {model_fits}; "
                  f"using disk center for the motion gate.\n{traceback.format_exc()}")
        return fallback


def _solar_rotate_howard(coordinate, target_time):
    """Rotate a coordinate with Howard rates across supported SunPy APIs."""
    try:
        return solar_rotate_coordinate(
            coordinate, time=target_time, model='howard')
    except TypeError as exc:
        if "unexpected keyword argument 'model'" not in str(exc):
            raise
        return solar_rotate_coordinate(
            coordinate, time=target_time, rot_type='howard')


def _s00_howard_displacement_arcsec(source_coords, reftime, duration_hours):
    """Measure maximum endpoint-to-endpoint Howard rotation displacement.

    :param source_coords: Helioprojective source coordinates at ``reftime``.
    :type source_coords: astropy.coordinates.SkyCoord
    :param reftime: Center time of the synthesis window.
    :type reftime: astropy.time.Time
    :param duration_hours: Full synthesis duration in hours.
    :type duration_hours: float
    :returns: Largest projected displacement in arcseconds.
    :rtype: float
    """
    center = Time(reftime)
    half_window = float(duration_hours) * 0.5 * u.hour
    try:
        at_center = _solar_rotate_howard(source_coords, center)
        at_start = _solar_rotate_howard(at_center, center - half_window)
        at_end = _solar_rotate_howard(at_center, center + half_window)
        dx = (at_end.Tx - at_start.Tx).to_value(u.arcsec)
        dy = (at_end.Ty - at_start.Ty).to_value(u.arcsec)
        displacement = np.hypot(dx, dy)
        finite = np.asarray(displacement, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.nanmax(finite)) if finite.size else np.inf
    except Exception:
        log_print('WARNING',
                  f"[s00_window] Howard displacement failed; rejecting the candidate.\n"
                  f"{traceback.format_exc()}")
        return np.inf


def _compute_round_intervals(wsclean_intervals, multiplier, base_interval=50, nintervals_minor=3):
    """Compute time intervals for one self-cal round."""
    wsclean_intervals.interval_length = base_interval * multiplier
    wsclean_intervals.compute_intervals(nintervals_minor=nintervals_minor)
    res = wsclean_intervals.results()
    timeranges = []
    for trange in res['time_intervals']:
        timeranges.append(
            trange[0].datetime.strftime('%Y/%m/%d/%H:%M:%S') + '~' +
            trange[-1].datetime.strftime('%Y/%m/%d/%H:%M:%S'))
    return {
        'N1': res['nintervals_major'],
        'N2': res['nintervals_minor'],
        'time_intervals': res['time_intervals'],
        'time_intervals_major_avg': res['time_intervals_major_avg'],
        'time_intervals_minor_avg': res['time_intervals_minor_avg'],
        'timeranges': timeranges,
    }


def _run_gaincal_step(msfile, workdir, spw, spwstr, antenna, caltbs, tdur, N1, N2,
                      step_name, solint, calmode, refantmode='flex',
                      uvrange='', gaintype='G', minsnr=1.0,
                      overwrite_caltb=True):
    """Run a single gaincal step with standard boilerplate."""
    ext = 'pha' if calmode == 'p' else ('amp' if calmode == 'a' else 'sbd')
    caltb = os.path.join(workdir, f"caltb_{step_name}_sp{spwstr}.{ext}")
    if os.path.exists(caltb) and overwrite_caltb:
        shutil.rmtree(caltb, ignore_errors=True)
    if not os.path.exists(caltb):
        if solint == 'auto_minor':
            solint = f'{tdur * 60 / N1 / N2:.0f}s'
        gaincal(vis=msfile, caltable=caltb, selectdata=True,
                uvrange=uvrange, spw=spw,
                combine="scan", antenna=f'{antenna}&{antenna}',
                refant='0', solint=solint, refantmode=refantmode,
                gaintype=gaintype, gaintable=caltbs,
                minsnr=minsnr, calmode=calmode, append=False)
    if os.path.exists(caltb):
        caltbs.append(caltb)
    return caltbs


def _run_slfcal_round(round_def, msfile, ri, spw, spwstr,
                      workdir, antenna, caltbs, tdur, pols, bmsize, fits_mask_sidx,
                      datacolumn, overwrite_caltb, uvmin_l_str_sidx, do_sbdcal=False,
                      clearcache=True):
    """Execute one round of feature self-calibration: image, gaincal, applycal.

    :param round_def: Configuration dict from SELFCAL_ROUNDS.
    :param ri: Interval dict from _compute_round_intervals().
    :param datacolumn: Data column to use (may be overridden by round_def).
    :returns: (slfcal_obj, caltbs, datacolumn) tuple.
    """
    dc = round_def['data_column'] or datacolumn
    with pipeline_stage(
        f"selfcal:{round_def['name']}",
        spw=spw,
        spwstr=spwstr,
        datacolumn=dc,
        n_major=ri['N1'],
        n_minor=ri['N2'],
        do_sbdcal=do_sbdcal and round_def['name'] == 'round1',
    ):
        slfcal_obj = MSselfcal(
            msfile, ri['time_intervals'], ri['time_intervals_major_avg'],
            ri['time_intervals_minor_avg'], spw, ri['N1'], ri['N2'], workdir,
            image_marker=round_def['name'], niter=round_def['niter'],
            briggs=round_def['briggs'], auto_mask=round_def['auto_mask'],
            auto_threshold=round_def['auto_threshold'],
            fits_mask=fits_mask_sidx, pols=pols, beam_size=bmsize,
            circular_beam=round_def['circular_beam'], data_column=dc)
        slfcal_obj.run(clearcache=clearcache)

        if not slfcal_obj.succeeded:
            log_step('WARNING', f"selfcal:{round_def['name']}", "MSselfcal reported failure", spw=spw, spwstr=spwstr)
            return None, caltbs, datacolumn

        ncaltbs_before = len(caltbs)
        for step in round_def['gaincal_steps']:
            uvrange = uvmin_l_str_sidx if step.get('uvrange_key') == 'uvmin' else ''
            with pipeline_stage(
                f"gaincal:{round_def['name']}_{step['suffix']}",
                spw=spw,
                spwstr=spwstr,
                solint=step['solint'],
                calmode=step['calmode'],
                uvrange=uvrange or 'none',
            ):
                caltbs = _run_gaincal_step(
                    msfile, workdir, spw, spwstr, antenna, caltbs, tdur,
                    ri['N1'], ri['N2'],
                    f"{round_def['name']}_{step['suffix']}", step['solint'],
                    step['calmode'], step['refantmode'], uvrange=uvrange,
                    overwrite_caltb=overwrite_caltb)

        # Single-band delay calibration (only in round1)
        if do_sbdcal and round_def['name'] == 'round1':
            with pipeline_stage("gaincal:round1_sbd", spw=spw, spwstr=spwstr):
                caltb_k = os.path.join(workdir, f"caltb_inf_sp{spwstr}.sbd")
                if os.path.isdir(caltb_k):
                    shutil.rmtree(caltb_k, ignore_errors=True)
                gaincal(vis=msfile, caltable=caltb_k, solint='inf', combine='scan',
                        uvrange='', refant='0', gaintype='K', spw=spw,
                        gaintable=caltbs, calmode='p', refantmode='flex',
                        minsnr=2.0, minblperant=4, append=False)
                if os.path.isdir(caltb_k):
                    caltbs.append(caltb_k)

        if len(caltbs) > ncaltbs_before:
            with pipeline_stage("applycal:feature", spw=spw, spwstr=spwstr, interp=round_def['interp']):
                applycal(vis=msfile, selectdata=True, antenna=antenna,
                         spw=spw, gaintable=caltbs, interp=round_def['interp'],
                         calwt=False, applymode="calonly")
            datacolumn = "CORRECTED_DATA"
            log_print('INFO', f"{round_def['name']} Feature Selfcal for SPW {spw}: solution applied")
        else:
            log_print('WARNING',
                      f"{round_def['name']} Feature Selfcal for SPW {spw}: no new solutions, skipping applycal")

    return slfcal_obj, caltbs, datacolumn


def _compute_uv_ranges(spws_indices, spws, freq, spwidx2proc):
    """Compute per-band UV range strings for disk and feature self-calibration."""
    uvmax_l_list = []
    uvmin_l_list = []
    uvmax_d = UV_CONFIG['uvmax_baseline_m']
    uvmin_d_default = UV_CONFIG['uvmin_baseline_m']
    uvmin_d_band0 = UV_CONFIG['uvmin_baseline_m_band0']

    for sidx, sp_index in enumerate(spws_indices):
        if sidx not in spwidx2proc:
            uvmax_l_list.append(0)
            uvmin_l_list.append(0)
        else:
            uvmin_d = uvmin_d_band0 if sidx == 0 else uvmin_d_default
            sp_st = int(spws[sidx].split('~')[0])
            wavelength = 3e8 / (np.nanmean(freq[sp_st:sp_st + 1]) * 1e9)
            uvmax_l_list.append(uvmax_d / wavelength)
            uvmin_l_list.append(uvmin_d / wavelength)

    uvmax_l_list = np.array(uvmax_l_list, dtype=float)
    uvmin_l_list = np.array(uvmin_l_list, dtype=float)
    lo, hi = UV_CONFIG['uvmax_l_clamp']
    uvmax_l_list = np.clip(uvmax_l_list, lo, hi)
    uvmin_l_list[uvmin_l_list > UV_CONFIG['uvmin_l_clamp_max']] = UV_CONFIG['uvmin_l_clamp_max']
    uvmin_l_str = [f'>{l:.0f}lambda' for l in uvmin_l_list]
    return uvmin_l_str


def _validate_xx_yy_correlation_order(msfile):
    """Verify the working MS stores XX and YY in the first two correlation slots."""
    poltb = os.path.join(msfile, 'POLARIZATION')
    tb.open(poltb)
    try:
        for row in range(tb.nrows()):
            corr_type = tb.getcell('CORR_TYPE', row)
            if corr_type.shape[0] < 2 or int(corr_type[0]) != 9 or int(corr_type[1]) != 12:
                raise RuntimeError(
                    f"Expected CORR_TYPE [XX, YY, ...] in {poltb} row {row}; got {corr_type.tolist()}")
    finally:
        tb.close()


def _swap_xx_yy_data_columns(msfile):
    """Swap XX and YY data-like columns in-place without changing MS metadata."""
    _validate_xx_yy_correlation_order(msfile)
    swap_columns = [
        'DATA', 'CORRECTED_DATA', 'MODEL_DATA', 'FLAG',
        'WEIGHT', 'SIGMA', 'WEIGHT_SPECTRUM', 'SIGMA_SPECTRUM',
    ]
    tb.open(msfile, nomodify=False)
    try:
        columns = [col for col in swap_columns if col in tb.colnames()]
        ddids = sorted(set(int(x) for x in tb.getcol('DATA_DESC_ID')))
        log_print('INFO', f"Swapping XX/YY slots in {os.path.basename(msfile)} columns {columns}")
        for ddid in ddids:
            subtable = tb.query(f'DATA_DESC_ID == {ddid}')
            try:
                for col in columns:
                    values = subtable.getcol(col)
                    if values.shape[0] < 2:
                        continue
                    swapped = values.copy()
                    swapped[0] = values[1]
                    swapped[1] = values[0]
                    subtable.putcol(col, swapped)
            finally:
                subtable.close()
        tb.flush()
    finally:
        tb.close()


def _prepare_yy_as_xx_ms(msfile, workdir, msname, pols, overwrite=False):
    """Return a YY-as-XX full-correlation working MS for YY diagnostic runs."""
    pol = str(pols).upper().replace(',', '')
    if pol not in ('YY', 'YY_AS_XX'):
        return msfile, msname

    pol_msname = f'{msname}.YYasXX'
    pol_msfile = os.path.join(workdir, f'{pol_msname}.ms')
    swap_marker = os.path.join(pol_msfile, '.yy_as_xx_swap_done')
    if os.path.isdir(pol_msfile):
        if overwrite:
            shutil.rmtree(pol_msfile, ignore_errors=True)
        else:
            _validate_xx_yy_correlation_order(pol_msfile)
            if not os.path.exists(swap_marker):
                raise RuntimeError(
                    f"Existing YY-as-XX working MS is missing swap marker; rerun with overwrite=True: {pol_msfile}")
            log_print('INFO', f"Using existing YY-as-XX working MS {pol_msfile}")
            return pol_msfile, pol_msname

    log_print('INFO', f"Creating YY-as-XX working MS {pol_msfile} from {msfile}")
    shutil.copytree(msfile, pol_msfile)
    if not os.path.isdir(pol_msfile):
        raise RuntimeError(f"Failed to create YY-as-XX working MS: {pol_msfile}")
    _swap_xx_yy_data_columns(pol_msfile)
    with open(swap_marker, 'w') as marker:
        marker.write('XX and YY data-like columns swapped; metadata intentionally unchanged.\n')
    return pol_msfile, pol_msname


def _ms_has_column(msfile, column):
    tb.open(msfile)
    try:
        return column in tb.colnames()
    finally:
        tb.close()


def _require_ms_column(msfile, column, operation):
    """Fail before splitting when a requested MS data column is absent."""
    if _ms_has_column(msfile, column):
        return
    raise RuntimeError(
        f"{operation} requires {column} in {msfile}, but that column is absent. "
        "The calibration/imaging stage did not produce a self-calibrated "
        "measurement set; refusing to assemble an output from raw DATA."
    )


def _require_wsclean_available():
    """Fail before mutating a measurement set when WSClean is unavailable."""
    executable = WSCLEAN_BIN
    if os.path.isabs(executable):
        available = os.path.isfile(executable) and os.access(executable, os.X_OK)
    else:
        available = shutil.which(executable) is not None
    if not available:
        raise RuntimeError(
            f"WSClean executable is unavailable: {executable!r}. "
            "Install the pinned EOVSA WSClean runtime or make it available "
            "on PATH before running imaging."
        )
    # Keep commands built by wrap_wsclean and this module identical even when
    # a caller imported the wrapper before this module resolved its binary.
    ww.WSCLEAN_BIN = executable
    return executable


def _run_disk_selfcal(msfile, sidx, spw, spwstr, sp_index, workdir, antenna,
                      caltbs, slfcal_init_obj, imname_init_disk_strlist,
                      freq_setup, dsize, fdens, ri_init,
                      ri_final, tdur, uvmin_l_str_sidx,
                      overwrite_caltb, pols):
    """Run disk self-calibration: predict disk model, gaincal, applycal, flag, uvsub."""
    with pipeline_stage("disk_selfcal", spw=spw, spwstr=spwstr, has_feature_model=slfcal_init_obj is not None):
        delmod(vis=msfile)
        run_start = datetime.now()
        log_print('INFO', f"Starting Disk self-calibration for SPW {spw} ...")

        for sp in sp_index.split(','):
            reffreq, cdelt4_real, bmsize = freq_setup.get_reffreq_and_cdelt(f'{sp}~{sp}', return_bmsize=True)
            sp_int = int(sp)
            if slfcal_init_obj is None:
                model_imname = '-'.join(imname_init_disk_strlist + [f'sp{sp_int:02d}_adddisk'])
                cmd = f"{WSCLEAN_BIN} -predict -reorder -spws {sp} -pol {pols} -name {model_imname} -quiet -intervals-out 1 {msfile}"
            else:
                dsz = float(dsize[sp_int].rstrip('arcsec'))
                fdn = fdens[sp_int]
                in_fits_files = sorted(glob(slfcal_init_obj.model_minor_name_str + '-t????-model.fits'))
                model_dir_disk_name_str = slfcal_init_obj.model_minor_name_str.replace(
                    format_spw(spw), f'{sp_int:02d}') + '_adddisk'
                if len(in_fits_files) > 0:
                    for in_fits in in_fits_files:
                        out_fits = in_fits.replace(slfcal_init_obj.model_minor_name_str, model_dir_disk_name_str)
                        add_convolved_disk_to_fits(in_fits, out_fits, dsz, fdn, ignore_data=False,
                                                   bmaj=bmsize / 3600., rfreq=reffreq)
                    cmd = (f"{WSCLEAN_BIN} -predict -reorder -spws {sp} -pol {pols} -name {model_dir_disk_name_str} "
                           f"-quiet -intervals-out {ri_init['N1'] * ri_init['N2']} {msfile}")
                else:
                    log_print('WARNING',
                              f"No feature cal model files found for SPW {spw}. Using uniform disk model.")
                    model_imname = '-'.join(imname_init_disk_strlist + [f'sp{sp_int:02d}_adddisk'])
                    cmd = f"{WSCLEAN_BIN} -predict -reorder -spws {sp} -pol {pols} -name {model_imname} -quiet -intervals-out 1 {msfile}"
            log_print('INFO', f"Running WSClean predict: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

        elapsed = (datetime.now() - run_start).total_seconds() / 60
        log_print('INFO', f"Disk self-calibration for SPW {spwstr}: completed in {elapsed:.1f} minutes")

        ncaltbs_before = len(caltbs)
        # Disk phase calibration (inf)
        caltbs = _run_gaincal_step(msfile, workdir, spw, spwstr, antenna, caltbs, tdur,
                                   1, 1, 'disk_inf', 'inf', 'p', refantmode='strict',
                                   overwrite_caltb=overwrite_caltb)
        # Disk phase calibration (int)
        caltbs = _run_gaincal_step(msfile, workdir, spw, spwstr, antenna, caltbs, tdur,
                                   ri_final['N1'], ri_final['N2'],
                                   'disk_int', 'auto_minor', 'p', refantmode='strict',
                                   overwrite_caltb=overwrite_caltb)
        # Disk amplitude calibration
        caltb_amp = os.path.join(workdir, f"caltb_disk_int_sp{spwstr}.amp")
        if os.path.exists(caltb_amp) and overwrite_caltb:
            shutil.rmtree(caltb_amp, ignore_errors=True)
        if not os.path.exists(caltb_amp):
            gaincal(vis=msfile, caltable=caltb_amp, selectdata=True,
                    uvrange='', spw=spw, combine="scan",
                    antenna=f'{antenna}&{antenna}', refant='0',
                    solint=f'{tdur * 60 / ri_final["N1"]:.0f}s',
                    refantmode="flex", gaintype='G', gaintable=caltbs,
                    minsnr=1, calmode='a', append=False)
        if os.path.exists(caltb_amp):
            mstl.flagcaltboutliers(caltb_amp, limit=[0.125, 8.0])
            caltbs.append(caltb_amp)

        # Apply solutions and flag
        if len(caltbs) > ncaltbs_before:
            log_print('INFO', f"Applying selfcal solutions for SPW {spw}")
            applycal(vis=msfile, selectdata=True, antenna=antenna,
                     spw=spw, gaintable=caltbs, interp='linear',
                     calwt=False, applymode="calonly")
        else:
            log_print('WARNING', f"No Selfcal solutions for SPW {spw}, skipping applycal")

        flagdata(vis=msfile, mode="tfcrop", spw=spw, action='apply', display='',
                 timecutoff=6.0, freqcutoff=6.0, maxnpieces=2, flagbackup=False)

        # Subtract disk model
        run_start_sub = datetime.now()
        if slfcal_init_obj is not None:
            for sp in sp_index.split(','):
                log_print('INFO', f'Inserting disk model into the data for SPW {sp}')
                sp_int = int(sp)
                model_imname = '-'.join(imname_init_disk_strlist + [f'sp{sp_int:02d}_adddisk'])
                cmd = f"{WSCLEAN_BIN} -predict -reorder -spws {sp} -pol {pols} -name {model_imname} -quiet -intervals-out 1 {msfile}"
                subprocess.run(cmd, shell=True, check=True)
        log_print('INFO', f'Subtracting disk model from the data for SPW {spw}')
        uvsub(vis=msfile)
        elapsed_sub = (datetime.now() - run_start_sub).total_seconds() / 60
        log_print('INFO', f"Disk subtraction for SPW {spwstr}: completed in {elapsed_sub:.1f} minutes")

    return caltbs


def _fits_corner_noise_peak(fitsfile, corner_frac=0.15):
    """Compute a cheap off-source-noise / peak estimate from an image's corners.

    Reads the first HDU with image data, then measures the standard
    deviation over the four corner boxes (a crude, WCS-free proxy for
    off-source/off-disk noise) and the peak pixel value over the full image.

    :param fitsfile: Path to a ``.tb.fits``-style image FITS file.
    :type fitsfile: str
    :param corner_frac: Fraction of each axis used for each corner box.
    :type corner_frac: float
    :returns: ``(noise, peak)`` in the same units as the FITS data (K for
        ``.tb.fits`` brightness-temperature images).
    :rtype: tuple(float, float)
    """
    with fits.open(fitsfile, memmap=False) as hdul:
        data = None
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data
                break
        if data is None:
            raise ValueError(f"No image data found in {fitsfile}")
    arr = np.squeeze(np.asarray(data, dtype=float))
    if arr.ndim != 2:
        # Collapse any leading singleton/stokes/freq axes; keep the last two.
        arr = arr.reshape(arr.shape[-2], arr.shape[-1]) if arr.ndim > 2 else arr
    ny, nx = arr.shape[-2], arr.shape[-1]
    cy = max(1, int(round(ny * corner_frac)))
    cx = max(1, int(round(nx * corner_frac)))
    corners = np.concatenate([
        arr[:cy, :cx].ravel(),
        arr[:cy, nx - cx:].ravel(),
        arr[ny - cy:, :cx].ravel(),
        arr[ny - cy:, nx - cx:].ravel(),
    ])
    noise = float(np.nanstd(corners))
    peak = float(np.nanmax(arr))
    return noise, peak


def _fine_spectral_quality_gate(fine_fits, coarse_fits, cfg=None):
    """Decide whether a fine-spectral image is good enough to keep.

    Compares off-source noise / peak / dynamic-range of a fine-spectral
    sub-chunk image against its parent coarse-group image (imaged from the
    same visibility interval, just before this fine chunk, at the same
    ``sidx``). The fine chunk is rejected when it looks diverged relative to
    the coarse parent: the coarse product already covers the same frequency
    range, so rejecting a diverged fine product does not lose sky coverage.

    Any failure to read/compute metrics defaults to KEEP (returns
    ``keep=True``) so a metrics-read problem never blocks the pipeline.

    :param fine_fits: Path to the fine-chunk's ``.tb.fits`` image.
    :type fine_fits: str
    :param coarse_fits: Path to the parent coarse-group's ``.tb.fits`` image
        for the same time interval, or None/missing if unavailable.
    :type coarse_fits: str or None
    :param cfg: Config dict with keys ``fine_spectral_noise_factor``,
        ``fine_spectral_dr_floor``, ``fine_spectral_peak_tb_max``. Defaults
        to :data:`PIPELINE_CONFIG` when None.
    :type cfg: dict or None
    :returns: ``(keep, metrics, reason)`` where ``metrics`` has keys
        ``fine_noise``, ``fine_peak``, ``fine_dr``, ``coarse_noise`` (when
        available), and ``reason`` is a human-readable string describing the
        pass/fail criterion (empty when kept cleanly).
    :rtype: tuple(bool, dict, str)
    """
    cfg = cfg or PIPELINE_CONFIG
    noise_factor = cfg.get('fine_spectral_noise_factor', 5.0)
    dr_floor = cfg.get('fine_spectral_dr_floor', 40.0)
    peak_tb_max = cfg.get('fine_spectral_peak_tb_max', 3.0e6)
    peak_ratio_max = cfg.get('fine_spectral_peak_ratio_max', 3.0)

    metrics = {'fine_noise': None, 'fine_peak': None, 'fine_dr': None,
               'coarse_noise': None, 'coarse_peak': None}
    try:
        fine_noise, fine_peak = _fits_corner_noise_peak(fine_fits)
        fine_dr = fine_peak / fine_noise if fine_noise > 0 else float('inf')
        metrics.update(fine_noise=fine_noise, fine_peak=fine_peak, fine_dr=fine_dr)

        coarse_noise = None
        coarse_peak = None
        if coarse_fits and os.path.exists(coarse_fits):
            coarse_noise, coarse_peak = _fits_corner_noise_peak(coarse_fits)
            metrics['coarse_noise'] = coarse_noise
            metrics['coarse_peak'] = coarse_peak

        if coarse_noise is not None and coarse_noise > 0 and fine_noise > noise_factor * coarse_noise:
            return False, metrics, (
                f"fine_noise={fine_noise:.3g} > "
                f"{noise_factor:g}*coarse_noise({coarse_noise:.3g})")
        if fine_dr < dr_floor:
            return False, metrics, f"fine_dr={fine_dr:.3g} < dr_floor({dr_floor:g})"
        # Peak criterion is RELATIVE to the coarse parent when available: a fine
        # chunk of a real source cannot exceed the group's MFS peak by much more
        # than the spectral slope allows (~x2), while true divergence blows up
        # >~x10. An ABSOLUTE ceiling misfires on active days: genuinely bright
        # sources (2-10 MK at 3-5 GHz with healthy DR 80-200) were mass-rejected
        # by peak_tb_max=2e6 on 2026-07-01. The absolute ceiling is retained
        # only as a fallback when no coarse parent exists for the interval.
        if coarse_peak is not None and coarse_peak > 0:
            if fine_peak > peak_ratio_max * coarse_peak:
                return False, metrics, (
                    f"fine_peak={fine_peak:.3g} > "
                    f"{peak_ratio_max:g}*coarse_peak({coarse_peak:.3g})")
        elif fine_peak > peak_tb_max:
            return False, metrics, f"fine_peak={fine_peak:.3g} > peak_tb_max({peak_tb_max:.3g})"
        return True, metrics, ""
    except Exception:
        log_print('WARNING',
                  f"[fine_spectral_snr_gate] metric computation failed for {fine_fits}; "
                  f"defaulting to KEEP\n{traceback.format_exc()}")
        return True, metrics, "metric-read-failed"


def _apply_fine_spectral_quality_gate(fine_fitsfiles, coarse_fitsfiles, tag, cfg=None):
    """Run the fine-spectral quality gate over a fine chunk's FITS outputs.

    For each fine-chunk FITS file, finds the coarse-parent FITS for the same
    imaging interval (matched by list position, since both lists are built
    from the same ``ri_final`` time intervals in ``_run_final_imaging``),
    evaluates :func:`_fine_spectral_quality_gate`, and deletes+drops any
    rejected fine FITS file. Logs INFO on pass, WARNING on rejection.

    :param fine_fitsfiles: FITS paths just written for the fine chunk.
    :type fine_fitsfiles: list(str)
    :param coarse_fitsfiles: FITS paths already written for the parent
        coarse group (same ``sidx``, same time intervals).
    :type coarse_fitsfiles: list(str)
    :param tag: Short label (e.g. ``fine_key``) used in log messages.
    :type tag: str
    :param cfg: Config dict; defaults to :data:`PIPELINE_CONFIG` when None.
    :type cfg: dict or None
    :returns: The subset of ``fine_fitsfiles`` that passed the gate (or all
        of them, unchanged, if gating could not run).
    :rtype: list(str)
    """
    cfg = cfg or PIPELINE_CONFIG
    kept = []
    for idx, fine_fits in enumerate(fine_fitsfiles or []):
        coarse_fits = coarse_fitsfiles[idx] if coarse_fitsfiles and idx < len(coarse_fitsfiles) else (
            coarse_fitsfiles[0] if coarse_fitsfiles else None)
        keep, metrics, reason = _fine_spectral_quality_gate(fine_fits, coarse_fits, cfg=cfg)
        if keep:
            log_print('INFO',
                      f"[fine_spectral_snr_gate] KEEP {tag} {os.path.basename(fine_fits)} | "
                      f"{_format_log_state(**metrics)}")
            kept.append(fine_fits)
        else:
            log_print('WARNING',
                      f"[fine_spectral_snr_gate] REJECT {tag} {os.path.basename(fine_fits)} | "
                      f"{reason} | {_format_log_state(**metrics)}")
            try:
                if os.path.exists(fine_fits):
                    os.remove(fine_fits)
            except Exception:
                log_print('WARNING',
                          f"[fine_spectral_snr_gate] failed to remove rejected FITS {fine_fits}\n"
                          f"{traceback.format_exc()}")
    return kept


def _register_and_export_interval_fits(msfile, fitsfilefinal, ri_final, imgoutdir, spwstr,
                                       reftime_daily, date_str, is_segmented,
                                       n_ant_img, solar_antenna_total, antenna_ids=None,
                                       tr_series_time=None, fits_tag=''):
    """Register per-interval WSClean FITS to helioprojective and export synoptic FITS.

    Extracted verbatim from the tail of :func:`_run_final_imaging` (the loop that
    ran after ``clean_junk`` there) so that :func:`_run_fine_joint_imaging` can
    route its per-channel WSClean outputs through the exact same registration,
    naming, and antenna-keyword-stamping logic used for the existing per-chunk
    fine-imaging path. Behavior-preserving: no logic changes vs. the original
    inline block.

    :param msfile: Input measurement set path (passed through to callers; kept
        for signature parity/future use, matching the original inline block's
        available scope).
    :type msfile: str
    :param fitsfilefinal: Registered helioprojective FITS paths (one per
        imaging interval), already produced by ``hf.imreg`` (+
        ``solar_diff_rot_heliofits`` for the segmented case) upstream.
    :type fitsfilefinal: list(str)
    :param ri_final: Interval dict from :func:`_compute_round_intervals`.
    :type ri_final: dict
    :param imgoutdir: Output directory for synoptic FITS files.
    :type imgoutdir: str
    :param spwstr: Formatted SPW range string (e.g. from :func:`format_spw`).
    :type spwstr: str
    :param reftime_daily: Daily reference time used to stamp non-segmented products.
    :type reftime_daily: astropy.time.Time
    :param date_str: Date string (``YYYYMMDD``) used in non-segmented filenames.
    :type date_str: str
    :param is_segmented: Whether this is the segmented (multi-interval) imaging path.
    :type is_segmented: bool
    :param n_ant_img: Unflagged solar antenna count used for imaging, or None.
    :type n_ant_img: int or None
    :param solar_antenna_total: Total solar antenna count for this epoch, or None.
    :type solar_antenna_total: int or None
    :param tr_series_time: List of (start_Time, end_Time) tuples to filter imaging intervals.
        If provided, only processes intervals overlapping with these ranges.
    :type tr_series_time: list or None
    :param fits_tag: Optional infix spliced into the output FITS filenames.
    :type fits_tag: str
    :returns: List of synoptic FITS file paths written.
    :rtype: list(str)
    """
    synfitsfiles = []
    for eoidx, eofile in enumerate(fitsfilefinal):
        interval_time = ri_final['time_intervals_major_avg'][eoidx]
        # Filter by tr_series_time if specified
        if tr_series_time is not None:
            in_range = any(t0.mjd <= interval_time.mjd <= t1.mjd for t0, t1 in tr_series_time)
            if not in_range:
                continue
        datetimestr = interval_time.datetime.strftime('%Y%m%dT%H%M%SZ')
        tag = f'.{fits_tag}' if fits_tag else ''
        if is_segmented:
            synfitsfile = os.path.join(imgoutdir,
                                       f"eovsa.synoptic{tag}.{datetimestr}.s{spwstr}.tb.fits")
        else:
            synfitsfile = os.path.join(imgoutdir,
                                       f"eovsa.synoptic_daily{tag}.{date_str}T200000Z.s{spwstr}.tb.fits")
        synfitsfiles.append(synfitsfile)
        log_print('INFO', f"Writing compressed synoptic FITS {synfitsfile} from {eofile} ...")
        # Keep every WSClean-facing FITS uncompressed. WSClean cannot read the
        # tiled-compressed FITS files written below, and moving compression
        # earlier in the model/imaging path breaks later WSClean predict runs.
        _write_compressed_synoptic_fits(eofile, synfitsfile)
        _set_imaging_antenna_keywords(
            synfitsfile, n_ant_img, solar_antenna_total, antenna_ids=antenna_ids)
        _write_imaging_antenna_manifest(
            synfitsfile, solar_antenna_total, antenna_ids)
        if not is_segmented:
            # Non-segmented daily product: content is aligned to the daily
            # reference epoch, but imreg stamps DATE-OBS with the timerange
            # start. Restamp to the reference epoch.
            _set_daily_reference_time_keywords(synfitsfile, reftime_daily)

    return synfitsfiles


def _fits_image_and_header(path):
    """Read the first two-dimensional image and matching FITS header."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None or 'CDELT1' not in hdu.header:
                continue
            data = np.asarray(hdu.data, dtype=float).squeeze()
            if data.ndim == 2:
                return data, hdu.header.copy()
    raise ValueError(f"No two-dimensional image HDU in {path}")


def _write_s00_mask(mask, template_fits, output_fits):
    """Write a plain WSClean-compatible FITS mask on a template image grid."""
    _, header = _fits_image_and_header(template_fits)
    os.makedirs(os.path.dirname(output_fits) or '.', exist_ok=True)
    fits.writeto(output_fits, np.asarray(mask, dtype=np.float32), header, overwrite=True)
    return output_fits


def _fits_grids_match(path_a, path_b, atol=1.0e-7):
    """Return whether two FITS images share an exact WSClean mask grid."""
    data_a, header_a = _fits_image_and_header(path_a)
    data_b, header_b = _fits_image_and_header(path_b)
    if data_a.shape != data_b.shape:
        return False
    for key in ('CDELT1', 'CDELT2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2'):
        if key not in header_a or key not in header_b:
            return False
        if not np.isclose(float(header_a[key]), float(header_b[key]), rtol=0.0, atol=atol):
            return False
    return True


def _wsclean_product(prefix, kind):
    """Resolve one WSClean FITS product with or without a time infix."""
    exact = f"{prefix}-{kind}.fits"
    if os.path.exists(exact):
        return exact
    matches = sorted(glob(f"{prefix}-t*-{kind}.fits"))
    return matches[0] if len(matches) == 1 else None


def _slfcal_source_model_fits(slfcal_obj):
    """Resolve the trusted long-integration model from one self-cal round.

    The feature self-cal object only sets ``model_ref_name_str`` when its
    models were explicitly rotated to the daily reference epoch.  The normal
    pipeline's init round keeps the equally useful native WSClean model
    instead, so resolve that exact one-interval product before falling back to
    a new shallow seed CLEAN.
    """
    if slfcal_obj is None:
        return None
    reference_prefix = getattr(slfcal_obj, 'model_ref_name_str', None)
    if reference_prefix:
        exact = f"{reference_prefix}-t0000-model.fits"
        if os.path.exists(exact):
            return exact
        matches = sorted(glob(f"{reference_prefix}-t*-model.fits"))
        if len(matches) == 1:
            return matches[0]

    workdir = getattr(slfcal_obj, 'workdir', None)
    msfile = getattr(slfcal_obj, 'msfile', None)
    spw = getattr(slfcal_obj, 'spw', None)
    marker = getattr(slfcal_obj, 'image_marker', None)
    if not workdir or not msfile or spw is None:
        return None
    parts = ['eovsa', 'major', os.path.basename(msfile),
             f"sp{format_spw(str(spw))}"]
    if marker:
        parts.append(str(marker))
    native_prefix = os.path.join(workdir, '-'.join(parts))
    exact = f"{native_prefix}-t0000-model.fits"
    if os.path.exists(exact):
        return exact
    exact = f"{native_prefix}-model.fits"
    if os.path.exists(exact):
        return exact
    matches = sorted(glob(f"{native_prefix}-t*-model.fits"))
    return matches[0] if len(matches) == 1 else None


def _run_s00_wsclean(msfile, prefix, sp_index, interval, briggs_val, bmsize,
                     pols, data_column, fits_mask=None, niter=20000,
                     no_update_model=False, auto_threshold=1,
                     make_psf=False):
    """Run one isolated, single-window s00 WSClean image.

    :returns: Mapping from standard WSClean product names to FITS paths, with
        integer ``status``.
    :rtype: dict
    """
    clean_junk(prefix, junks=['dirty', 'model', 'psf', 'residual', 'image'])
    clean_obj = ww.WSClean(msfile)
    # WSClean requires the automask threshold to be strictly higher than the
    # final auto-threshold.  The shallow source-model pass deliberately uses a
    # higher final threshold than the science/review passes, so derive the
    # automask threshold here instead of silently creating an invalid pair.
    auto_mask = max(2.0, float(auto_threshold) + 1.0)
    clean_obj.setup(
        size=1024, scale="2.5asec", pol=pols,
        weight_briggs=briggs_val,
        niter=int(niter), mgain=0.85, gain=0.2,
        data_column=data_column,
        name=prefix,
        multiscale=bool(niter > 0), multiscale_gain=0.3,
        multiscale_scale_bias=0.6,
        auto_mask=auto_mask, auto_threshold=auto_threshold,
        minuv_l=200,
        interval=list(interval), intervals_out=1,
        no_negative=False, quiet=True,
        circular_beam=True, beam_size=bmsize,
        fits_mask=fits_mask,
        no_update_model=bool(no_update_model),
        make_psf=bool(make_psf),
        spws=sp_index,
    )
    status = clean_obj.run(dryrun=False)
    products = {'status': int(status), 'prefix': prefix}
    for kind in ('dirty', 'model', 'psf', 'residual', 'image'):
        products[kind] = _wsclean_product(prefix, kind)
    return products


def _s00_disk_clip_mask(template_fits, disk_mask_fits, reftime):
    """Load the broad disk mask or recreate its 1.1-Rsun geometry."""
    data, header = _fits_image_and_header(template_fits)
    if disk_mask_fits and os.path.exists(disk_mask_fits):
        try:
            disk, _ = _fits_image_and_header(disk_mask_fits)
            if disk.shape == data.shape:
                return disk > 0
        except Exception:
            log_print('WARNING',
                      f"[s00_mask] could not read broad disk mask {disk_mask_fits}; "
                      f"recreating it from WCS.\n{traceback.format_exc()}")
    pixel_arcsec = abs(float(header['CDELT1'])) * 3600.0
    rsun_pix = float(sun.angular_radius(Time(reftime)).to_value(u.arcsec)) / pixel_arcsec
    crpix1 = float(header.get('CRPIX1', data.shape[1] / 2.0 + 1.0)) - 1.0
    crpix2 = float(header.get('CRPIX2', data.shape[0] / 2.0 + 1.0)) - 1.0
    yy, xx = np.indices(data.shape)
    return np.hypot(xx - crpix1, yy - crpix2) <= 1.1 * rsun_pix


def _prepare_s00_final_policy(msfile, msname, sp_index, workdir, imgoutdir,
                              wsclean_intervals, reftime_daily, model_fits,
                              disk_mask_fits, briggs_val, bmsize, pols,
                              data_column, fits_tag='', config=None):
    """Select a central s00 window and construct its compact support mask.

    :param fits_tag: Exact output-route tag used to isolate work and QA state.
    :type fits_tag: str
    :returns: Policy dictionary consumed by :func:`_run_final_imaging`, or
        ``None`` when no conservative source support can be constructed.
    :rtype: dict or None
    """
    cfg = dict(S00_FINAL_CONFIG)
    if config:
        cfg.update(config)
    route = _s00_route_identity(fits_tag)
    route_namespace = route['namespace']
    artifact_stem = f"eovsa-{msname}-sp00-01-s00-{route_namespace}"
    qa_dir = os.path.join(imgoutdir, 'qa', 's00', route_namespace)
    os.makedirs(qa_dir, exist_ok=True)
    seed_motion_model_fits = None

    if not model_fits or not os.path.exists(model_fits):
        archived_models = [
            path for path in sorted(glob(os.path.join(
                qa_dir, f"*{msname}*init*model.fits")))
            if not path.endswith('model.helio.fits')
        ]
        if len(archived_models) == 1:
            model_fits = archived_models[0]
            log_print(
                'INFO',
                f"[s00_policy] reusing archived long-integration source model "
                f"{model_fits} for imaging-only QA.")

    if not model_fits or not os.path.exists(model_fits):
        seed_candidate = _central_wsclean_interval(
            wsclean_intervals, reftime_daily, max(cfg['candidate_hours']),
            min_coverage_fraction=0.0, max_gap_factor=np.inf)
        if not seed_candidate.get('interval'):
            log_print('ERROR', '[s00_policy] no time support for a shallow source model.')
            return None
        seed_prefix = os.path.join(workdir, f"{artifact_stem}-seed")
        seed = _run_s00_wsclean(
            msfile, seed_prefix, sp_index, seed_candidate['interval'],
            briggs_val, bmsize, pols, data_column,
            niter=100, no_update_model=True, auto_threshold=3)
        model_fits = seed.get('model') or seed.get('image')
        if seed['status'] != 0 or not model_fits:
            log_print('ERROR', '[s00_policy] shallow source-model run failed; skipping s00.')
            return None

        if model_fits.endswith('model.fits'):
            seed_motion_model_fits = model_fits.replace(
                'model.fits', 'model.helio.fits')
        else:
            seed_motion_model_fits = model_fits.replace(
                '.fits', '.helio.fits')
        try:
            hf.imreg(
                vis=msfile, imagefile=model_fits,
                fitsfile=seed_motion_model_fits,
                timerange=seed_candidate['ri']['timeranges'][0])
        except Exception:
            seed_motion_model_fits = None
            log_print(
                'WARNING',
                f"[s00_window] could not register the shallow source model; "
                f"using the conservative disk-center motion estimate.\n"
                f"{traceback.format_exc()}")

    motion_model_fits = (seed_motion_model_fits or
                         model_fits.replace('model.fits', 'model.helio.fits'))
    if not os.path.exists(motion_model_fits):
        motion_model_fits = model_fits
    source_coords = _s00_source_coordinates(
        motion_model_fits, reftime_daily,
        peak_fraction=cfg['support_peak_fraction'],
        threshold_sigma=cfg['support_sigma'])
    candidates = []
    probe_prefixes = []
    for hours in cfg['candidate_hours']:
        candidate = _central_wsclean_interval(
            wsclean_intervals, reftime_daily, hours,
            min_coverage_fraction=cfg['min_coverage_fraction'],
            max_gap_factor=cfg['max_gap_factor'])
        motion_arcsec = _s00_howard_displacement_arcsec(source_coords, reftime_daily, hours)
        candidate['motion_arcsec'] = motion_arcsec
        candidate['motion_beams'] = motion_arcsec / float(bmsize)
        candidate['psf_sidelobe'] = np.inf
        candidate['probe_psf'] = None
        if candidate.get('interval'):
            probe_prefix = os.path.join(
                workdir, f"{artifact_stem}-probe-{float(hours):g}h")
            probe = _run_s00_wsclean(
                msfile, probe_prefix, sp_index, candidate['interval'],
                briggs_val, bmsize, pols, data_column,
                niter=0, no_update_model=True, make_psf=True)
            probe_prefixes.append(probe_prefix)
            if probe['status'] == 0 and probe.get('psf'):
                psf_data, psf_header = _fits_image_and_header(probe['psf'])
                beam_pix = float(bmsize) / (abs(float(psf_header['CDELT1'])) * 3600.0)
                candidate['psf_sidelobe'] = s00qa.measure_psf_sidelobe(psf_data, beam_pix)
                candidate['probe_psf'] = probe['psf']
        candidate['coverage_ok'] = bool(candidate['valid'])
        candidate['motion_ok'] = bool(candidate['motion_beams'] < cfg['smear_beam_max'])
        candidate['psf_ok'] = bool(candidate['psf_sidelobe'] <= cfg['psf_sidelobe_max'])
        candidate['accepted'] = bool(
            candidate['coverage_ok'] and candidate['motion_ok'] and candidate['psf_ok'])
        candidates.append(candidate)

    selected = next((item for item in candidates if item['accepted']), None)
    precondition_state = None
    if selected is None:
        usable = [item for item in reversed(candidates) if item.get('interval')]
        if not usable:
            log_print('ERROR', '[s00_policy] no central candidate contains usable samples.')
            return None
        selected = usable[0]
        if not selected['coverage_ok']:
            precondition_state = s00qa.FAIL_COVERAGE
        elif not selected['motion_ok']:
            precondition_state = s00qa.FAIL_MOTION
        elif not selected['psf_ok']:
            precondition_state = s00qa.FAIL_PSF
        else:
            precondition_state = s00qa.FAIL_COVERAGE

    if selected.get('probe_psf') and not _fits_grids_match(model_fits, selected['probe_psf']):
        log_print(
            'ERROR',
            f"[s00_policy] init model grid does not match the selected final-imaging "
            f"grid; refusing to resample a CLEAN mask ({model_fits} vs "
            f"{selected['probe_psf']}).")
        return None

    model_data, model_header = _fits_image_and_header(model_fits)
    pixel_arcsec = abs(float(model_header['CDELT1'])) * 3600.0
    beam_pix = float(bmsize) / pixel_arcsec
    disk_clip = _s00_disk_clip_mask(model_fits, disk_mask_fits, reftime_daily)
    try:
        support_mask, support_metrics = s00qa.build_support_mask(
            model_data, beam_pix,
            motion_pix=float(selected['motion_arcsec']) / pixel_arcsec,
            threshold_sigma=cfg['support_sigma'],
            peak_fraction=cfg['support_peak_fraction'],
            dilation_beams=cfg['support_dilation_beams'],
            clip_mask=disk_clip)
    except Exception:
        log_print('ERROR',
                  f"[s00_policy] compact support-mask construction failed; skipping s00.\n"
                  f"{traceback.format_exc()}")
        return None

    support_mask_fits = os.path.join(
        workdir, f"{artifact_stem}-support-mask.fits")
    _write_s00_mask(support_mask, model_fits, support_mask_fits)
    disk_clip_fits = os.path.join(
        workdir, f"{artifact_stem}-disk-clip-mask.fits")
    _write_s00_mask(disk_clip, model_fits, disk_clip_fits)
    log_print(
        'INFO',
        f"[s00_policy] selected {selected['duration_hours']:.1f} h central window | "
        f"coverage={selected['coverage_fraction']:.3f}, "
        f"psf={selected['psf_sidelobe']:.3f}, "
        f"motion={selected['motion_beams']:.3f} beam, "
        f"precondition_state={precondition_state or 'pass'}")
    return {
        'config': cfg,
        'fits_tag': route['fits_tag'],
        'route_id': route['route_id'],
        'route_namespace': route_namespace,
        'artifact_stem': artifact_stem,
        'qa_dir': qa_dir,
        'interval': selected['interval'],
        'ri': selected['ri'],
        'selected': selected,
        'candidates': candidates,
        'precondition_state': precondition_state,
        'support_mask': support_mask,
        'support_mask_fits': support_mask_fits,
        'disk_clip_mask': disk_clip,
        'disk_clip_mask_fits': disk_clip_fits,
        'support_metrics': support_metrics,
        'source_model_fits': model_fits,
        'motion_model_fits': motion_model_fits,
        'beam_fwhm_pix': beam_pix,
        'probe_prefixes': probe_prefixes,
        'artifact_prefixes': [],
    }


def _json_safe(value):
    """Convert nested pipeline QA state into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()
                if key not in ('ri', 'support_mask', 'disk_clip_mask')}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Time):
        return value.isot.tolist() if not value.isscalar else value.isot
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _copy_s00_products(prefixes, extra_files, qa_dir):
    """Copy the permanent s00 model/PSF/residual review bundle."""
    os.makedirs(qa_dir, exist_ok=True)
    copied = []
    for prefix in prefixes:
        for kind in ('dirty', 'model', 'psf', 'residual', 'image'):
            source = _wsclean_product(prefix, kind)
            if not source or not os.path.exists(source):
                continue
            destination = os.path.join(qa_dir, os.path.basename(source))
            shutil.copy2(source, destination)
            copied.append(destination)
    for source in extra_files:
        if source and os.path.exists(source):
            destination = os.path.join(qa_dir, os.path.basename(source))
            if os.path.abspath(source) != os.path.abspath(destination):
                shutil.copy2(source, destination)
            copied.append(destination)
    return copied


def _stamp_s00_qa(fitsfile, qa):
    """Stamp compact deterministic s00 QA fields into a synoptic FITS."""
    if not fitsfile or not os.path.exists(fitsfile):
        return
    selected = qa.get('selected') or {}
    state = str(qa.get('state') or s00qa.FAIL_AMBIGUOUS_SOURCE)
    def finite_metric(key):
        value = float(selected.get(key, -1.0))
        return value if np.isfinite(value) else -1.0

    with fits.open(fitsfile, mode='update') as hdul:
        for hdu in hdul:
            if 'CDELT1' not in hdu.header:
                continue
            hdr = hdu.header
            hdr.set('QASTATE', state, 'deterministic s00 imaging QA state')
            hdr.set('S00WINH', finite_metric('duration_hours'),
                    'selected central s00 window [h]')
            hdr.set('S00PSF', finite_metric('psf_sidelobe'),
                    'max abs PSF sidelobe outside 2 FWHM')
            hdr.set('S00SMEAR', finite_metric('motion_beams'),
                    'Howard source displacement / restoring beam')
            hdr.set('S00EXP', bool(state == s00qa.PASS_EXPANDED_MASK),
                    'support mask expanded after held-out QA')
            hdr.set('S00RNS', str(qa.get('route_namespace', 'production')),
                    'route-isolated s00 QA namespace')
            hdr.add_history('s00 CLEAN constrained by deterministic source-support mask')
            break


def _evaluate_s00_residual_policy(msfile, base_products, policy, sp_index,
                                  briggs_val, bmsize, pols, data_column):
    """Run held-out residual tests and at most one rollback-safe mask retry."""
    cfg = policy['config']
    required = ('residual', 'psf', 'model', 'image')
    if any(not base_products.get(key) for key in required):
        policy['state'] = s00qa.FAIL_AMBIGUOUS_SOURCE
        policy['assessment'] = {'triggered': True, 'reason': 'missing_base_products'}
        return base_products['prefix'], policy

    residual, _ = _fits_image_and_header(base_products['residual'])
    psf, _ = _fits_image_and_header(base_products['psf'])
    model, _ = _fits_image_and_header(base_products['model'])
    restored, _ = _fits_image_and_header(base_products['image'])
    support = policy['support_mask']
    source_peak = float(np.nanmax(np.abs(restored[support]))) if np.any(support) else 0.0
    assessment = s00qa.detect_residual_candidates(
        residual, support, psf, policy['beam_fwhm_pix'],
        source_model=model, in_mask_source_peak=source_peak,
        snr_trigger=cfg['residual_snr_trigger'],
        peak_ratio_trigger=cfg['residual_peak_ratio_trigger'])
    policy['assessment'] = assessment
    precondition = policy.get('precondition_state')
    if precondition:
        policy['state'] = precondition
        return base_products['prefix'], policy
    if not assessment['triggered']:
        policy['state'] = s00qa.PASS_BASE_MASK
        return base_products['prefix'], policy

    i0, i1 = [int(value) for value in policy['interval']]
    midpoint = i0 + (i1 - i0) // 2
    if midpoint <= i0 or midpoint >= i1:
        policy['state'] = s00qa.FAIL_AMBIGUOUS_SOURCE
        policy['repeatability'] = {'validated': [], 'reason': 'interval_too_short'}
        return base_products['prefix'], policy

    diagnostic_products = []
    half_residuals = []
    half_psfs = []
    for label, interval in (('half-a', [i0, midpoint]), ('half-b', [midpoint, i1])):
        prefix = f"{base_products['prefix']}-{label}"
        products = _run_s00_wsclean(
            msfile, prefix, sp_index, interval, briggs_val, bmsize,
            pols, data_column, fits_mask=policy['support_mask_fits'],
            niter=cfg['qa_clean_niter'], no_update_model=True)
        diagnostic_products.append(products)
        policy['artifact_prefixes'].append(prefix)
        if (products['status'] == 0 and products.get('residual')
                and products.get('psf')):
            half_residuals.append(_fits_image_and_header(products['residual'])[0])
            half_psfs.append(_fits_image_and_header(products['psf'])[0])

    frequency_residuals = []
    frequency_psfs = []
    spw_ids = [item for item in str(sp_index).split(',') if item]
    for spw_id in spw_ids[:2]:
        prefix = f"{base_products['prefix']}-spw{spw_id}"
        products = _run_s00_wsclean(
            msfile, prefix, spw_id, policy['interval'], briggs_val, bmsize,
            pols, data_column, fits_mask=policy['support_mask_fits'],
            niter=cfg['qa_clean_niter'], no_update_model=True)
        diagnostic_products.append(products)
        policy['artifact_prefixes'].append(prefix)
        if (products['status'] == 0 and products.get('residual')
                and products.get('psf')):
            frequency_residuals.append(_fits_image_and_header(products['residual'])[0])
            frequency_psfs.append(_fits_image_and_header(products['psf'])[0])

    repeatability = s00qa.validate_candidate_repeatability(
        assessment['candidates'], half_residuals, support,
        policy['beam_fwhm_pix'], frequency_residuals=frequency_residuals,
        time_half_psfs=half_psfs, frequency_psfs=frequency_psfs,
        source_model=model, required_frequency_count=2,
        sigma_min=cfg['half_snr_min'],
        centroid_beam_fraction=cfg['centroid_beam_fraction'])
    policy['repeatability'] = repeatability
    if not repeatability['validated']:
        policy['state'] = s00qa.REVIEW_PSF_RESIDUAL
        return base_products['prefix'], policy

    expanded_mask, growth = s00qa.expand_support_mask(
        support, repeatability['validated'], policy['beam_fwhm_pix'],
        max_growth_fraction=cfg['max_mask_growth_fraction'],
        max_added_beams_per_candidate=cfg['max_added_beams_per_candidate'],
        max_candidate_islands=cfg['max_candidate_islands'],
        max_image_fraction=cfg['max_mask_image_fraction'],
        clip_mask=policy['disk_clip_mask'])
    policy['mask_growth'] = growth
    if not growth['accepted']:
        policy['state'] = s00qa.FAIL_AMBIGUOUS_SOURCE
        return base_products['prefix'], policy
    expanded_mask_fits = policy['support_mask_fits'].replace('.fits', '-expanded.fits')
    _write_s00_mask(expanded_mask, base_products['image'], expanded_mask_fits)
    policy['expanded_mask_fits'] = expanded_mask_fits

    retry_prefix = f"{base_products['prefix']}-retry"
    retry = _run_s00_wsclean(
        msfile, retry_prefix, sp_index, policy['interval'], briggs_val, bmsize,
        pols, data_column, fits_mask=expanded_mask_fits, niter=20000,
        no_update_model=True)
    policy['artifact_prefixes'].append(retry_prefix)
    if retry['status'] != 0 or any(not retry.get(key) for key in required):
        policy['state'] = s00qa.FAIL_AMBIGUOUS_SOURCE
        policy['retry'] = {'accepted': False, 'reason': 'retry_imaging_failed'}
        return base_products['prefix'], policy

    retry_result = s00qa.validate_retry(
        residual, _fits_image_and_header(retry['residual'])[0],
        model, _fits_image_and_header(retry['model'])[0],
        support, expanded_mask, repeatability['validated'],
        policy['beam_fwhm_pix'],
        residual_drop_min=cfg['retry_residual_drop_min'],
        energy_drop_min=cfg['retry_energy_drop_min'],
        rms_worsen_max=cfg['retry_rms_worsen_max'],
        source_flux_change_max=cfg['retry_source_flux_change_max'])
    policy['retry'] = retry_result
    if retry_result['accepted']:
        policy['state'] = s00qa.PASS_EXPANDED_MASK
        policy['support_mask'] = expanded_mask
        return retry_prefix, policy
    policy['state'] = s00qa.FAIL_AMBIGUOUS_SOURCE
    return base_products['prefix'], policy


def _run_final_imaging(msfile, sidx, spw, spwstr, sp_index, workdir, imgoutdir,
                       msname, ri_final, briggs_val, bmsize, pols,
                       reftime_daily, viz_timerange, date_str,
                       is_segmented, imaging_objs, freq_setup,
                       tr_series_time=None, fits_tag='', data_column='CORRECTED_DATA',
                       solar_antenna_total=None, s00_policy=None):
    """Run final imaging (segmented or non-segmented) and return output FITS paths.

    :param tr_series_time: List of (start_Time, end_Time) tuples to filter imaging intervals.
        If provided, only processes intervals overlapping with these ranges.
    :param fits_tag: Optional infix (e.g. ``'test'``) spliced after ``eovsa.synoptic``
        / ``eovsa.synoptic_daily`` in the output FITS filenames so alternate-source
        runs (e.g. calwidget_v2 NPZ calibration) do not collide with production
        artefacts.
    :param s00_policy: Optional deterministic central-window/support-mask plan.
        It is accepted only by the segmented s00 path; all other callers pass
        ``None`` and retain the historical behavior.
    :type s00_policy: dict or None
    """
    gain = 0.2
    reffreq, cdelt4_real, _ = freq_setup.get_reffreq_and_cdelt(spw, return_bmsize=True)
    antenna_ids = unflagged_solar_antenna_ids(msfile, sp_index, solar_antenna_total)
    n_ant_img = None if antenna_ids is None else len(antenna_ids)
    if n_ant_img is not None and solar_antenna_total is not None:
        log_print('INFO',
                  f"SPW {spwstr}: {n_ant_img}/{int(solar_antenna_total)} solar antennas "
                  f"have unflagged data available for final imaging.")

    with pipeline_stage("final_imaging", spw=spw, spwstr=spwstr, segmented=is_segmented):
        if is_segmented:
            imname_strlist = ["eovsa", "major", f"{msname}", f"sp{spwstr}", 'final']
            if s00_policy is not None:
                imname_strlist.append(f"s00-{s00_policy['route_namespace']}")
            imname = '-'.join(imname_strlist)
            final_prefix = os.path.join(workdir, imname)
            clean_junk(final_prefix, junks=['dirty', 'model', 'psf', 'residual', 'image'])
            clean_obj = ww.WSClean(msfile)
            clean_obj.setup(size=1024, scale="2.5asec", pol=pols,
                            weight_briggs=briggs_val,
                            niter=20000, mgain=0.85, gain=gain,
                            data_column=data_column,
                            name=final_prefix,
                            multiscale=True, multiscale_gain=0.3,
                            multiscale_scale_bias=0.6,
                            auto_mask=2, auto_threshold=1,
                            minuv_l=200,
                            interval=(s00_policy['interval'] if s00_policy else []),
                            intervals_out=(1 if s00_policy else ri_final['N1']),
                            no_negative=False, quiet=True,
                            circular_beam=True, beam_size=bmsize,
                            fits_mask=(s00_policy['support_mask_fits']
                                       if s00_policy else None),
                            spws=sp_index)
            wsclean_status = clean_obj.run(dryrun=False)
            if wsclean_status != 0:
                log_print('ERROR',
                          f"[final_imaging] WSClean failed with status {wsclean_status}; "
                          f"SPW {spw} has no current final model.")
                clean_junk(final_prefix)
                return [], imaging_objs

            if s00_policy is not None:
                base_products = {'status': 0, 'prefix': final_prefix}
                for kind in ('dirty', 'model', 'psf', 'residual', 'image'):
                    base_products[kind] = _wsclean_product(final_prefix, kind)
                s00_policy['artifact_prefixes'].append(final_prefix)
                chosen_prefix, s00_policy = _evaluate_s00_residual_policy(
                    msfile, base_products, s00_policy, sp_index,
                    briggs_val, bmsize, pols, data_column)
                final_prefix = chosen_prefix
                imname = os.path.basename(chosen_prefix)
                qa_prefixes = (
                    list(s00_policy.get('probe_prefixes', [])) +
                    list(s00_policy.get('artifact_prefixes', [])))
                qa_files = [
                    s00_policy.get('support_mask_fits'),
                    s00_policy.get('disk_clip_mask_fits'),
                    s00_policy.get('expanded_mask_fits'),
                    s00_policy.get('source_model_fits'),
                    s00_policy.get('motion_model_fits'),
                ]
                copied = _copy_s00_products(
                    qa_prefixes, qa_files, s00_policy['qa_dir'])
                s00_policy['qa_files'] = copied
                qa_json = os.path.join(
                    s00_policy['qa_dir'],
                    f"{msname}-s00-{s00_policy['route_namespace']}-imaging-qa.json")
                with open(qa_json, 'w') as outfile:
                    json.dump(_json_safe(s00_policy), outfile, indent=2, sort_keys=True)
                s00_policy['qa_json'] = qa_json

            fitsname = sorted(glob(final_prefix + '-t*-image.fits'))
            if not fitsname:
                # wsclean omits the '-t????-' infix when intervals_out=1;
                # fall back to the non-time-indexed output name.
                fitsname = sorted(glob(final_prefix + '-image.fits'))
            fitsname_helio = [f.replace('image.fits', 'image.helio.fits') for f in fitsname]
            fitsname_helio_ref_daily = [f.replace('image.fits', 'image.helio.ref_daily.fits') for f in fitsname]
            try:
                hf.imreg(vis=msfile, imagefile=fitsname, fitsfile=fitsname_helio,
                         timerange=ri_final['timeranges'], toTb=True)
                for eoidx, (fits_helio, fits_helio_ref_daily) in enumerate(
                        zip(fitsname_helio, fitsname_helio_ref_daily)):
                    solar_diff_rot_heliofits(fits_helio, reftime_daily, fits_helio_ref_daily,
                                             in_time=ri_final['time_intervals_major_avg'][eoidx])
                fitsfilefinal = fitsname_helio_ref_daily
            except Exception:
                log_print('ERROR', f"[final_imaging] helioprojective registration failed | "
                                   f"{_format_log_state(spw=spw, spwstr=spwstr, n_fits=len(fitsname))}\n"
                                   f"{traceback.format_exc()}")
                fitsfilefinal = []
            log_print('INFO', f"Final imaging for SPW {spw}: completed")
        else:
            clean_obj = ww.WSClean(msfile)
            imaging_obj = MSselfcal(msfile, ri_final['time_intervals'], ri_final['time_intervals_major_avg'],
                                    ri_final['time_intervals_minor_avg'],
                                    spw, ri_final['N1'], ri_final['N2'], workdir, image_marker='temp',
                                    niter=10000, briggs=briggs_val, gain=gain,
                                    minuv_l=200,
                                    auto_mask=2, auto_threshold=1,
                                    data_column=data_column, pols=pols,
                                    no_negative=False, circular_beam=False,
                                    theoretic_beam=True,
                                    reftime_daily=reftime_daily,
                                    sp_index=sp_index)
            imaging_obj.run()
            imaging_objs[sidx] = imaging_obj

            if not imaging_obj.succeeded:
                log_print('ERROR', f"Final imaging for SPW {spw} failed when creating viz model. Skipping...")
                clean_junk('-'.join(["eovsa", "major", f"{msname}", f"sp{spwstr}", 'final']))
                return [], imaging_objs

            uvsub(vis=msfile)
            delmod(vis=msfile)

            model_dir_name_str = imaging_objs[sidx].model_ref_name_str
            cmd = (f"{WSCLEAN_BIN} -predict -reorder -spws {sp_index} "
                   f"-pol {pols} -name {model_dir_name_str} -quiet -intervals-out {ri_final['N1']} {msfile}")
            subprocess.run(cmd, shell=True, check=True)
            uvsub(vis=msfile, reverse=True)

            imname_strlist = ["eovsa", "major", f"{msname}", f"sp{spwstr}", 'final']
            imname = '-'.join(imname_strlist)
            final_prefix = os.path.join(workdir, imname)
            clean_junk(final_prefix, junks=['dirty', 'model', 'psf', 'residual', 'image'])
            clean_obj.setup(size=1024, scale="2.5asec", pol=pols,
                            weight_briggs=briggs_val,
                            niter=20000, mgain=0.85, gain=gain,
                            data_column=data_column,
                            name=final_prefix,
                            multiscale=True, multiscale_gain=0.3,
                            multiscale_scale_bias=0.6,
                            auto_mask=2, auto_threshold=1,
                            minuv_l=200,
                            intervals_out=1,
                            no_negative=False, quiet=True,
                            theoretic_beam=True, circular_beam=False,
                            spws=sp_index)
            wsclean_status = clean_obj.run(dryrun=False)
            if wsclean_status != 0:
                log_print('ERROR',
                          f"[final_imaging] WSClean failed with status {wsclean_status}; "
                          f"SPW {spw} has no current final model.")
                clean_junk(final_prefix)
                return [], imaging_objs
            log_print('INFO', f"Final imaging for SPW {spw}: completed")
            fitsname = sorted(glob(os.path.join(workdir, imname + '*image.fits')))
            fitsname_helio = [f.replace('image.fits', 'image.helio.fits') for f in fitsname]
            try:
                hf.imreg(vis=msfile, imagefile=fitsname, fitsfile=fitsname_helio,
                         timerange=[viz_timerange], toTb=True)
                fitsfilefinal = fitsname_helio
            except Exception:
                log_print('ERROR', f"[final_imaging] helioprojective registration failed | "
                                   f"{_format_log_state(spw=spw, spwstr=spwstr, n_fits=len(fitsname))}\n"
                                   f"{traceback.format_exc()}")
                fitsfilefinal = []

    clean_junk('-'.join(["eovsa", "major", f"{msname}", f"sp{spwstr}", 'final']))
    if s00_policy is not None:
        for prefix in set(s00_policy.get('probe_prefixes', []) +
                          s00_policy.get('artifact_prefixes', [])):
            # The permanent QA copies were written above.  Keep only a selected
            # restored image until registration/export finishes below.
            if prefix != final_prefix:
                clean_junk(prefix, junks=['dirty', 'model', 'psf', 'residual', 'image'])
            else:
                clean_junk(prefix)

    synfitsfiles = _register_and_export_interval_fits(
        msfile, fitsfilefinal, ri_final, imgoutdir, spwstr,
        reftime_daily, date_str, is_segmented,
        n_ant_img, solar_antenna_total, antenna_ids=antenna_ids,
        tr_series_time=tr_series_time, fits_tag=fits_tag)

    if s00_policy is not None:
        for synfitsfile in synfitsfiles:
            _stamp_s00_qa(synfitsfile, s00_policy)

    return synfitsfiles, imaging_objs


def _run_bootstrapped_fine_product(*, parent_msfile, source_sp_index,
                                   fine_spw, fine_spwstr, fine_msfile,
                                   workdir, antenna, parent_data_column,
                                   final_imaging_kwargs):
    """Image one fine product from an isolated copy of its coarse parent model.

    ``splitX`` copies both the parent data column used for coarse imaging and
    the coarse final image's ``MODEL_DATA``.  The child then gets exactly one
    phase-only refinement before final imaging.  The parent MS is never
    calibrated or imaged by this helper.
    """
    local_spw_count = len(_spw_id_list(source_sp_index))
    if local_spw_count < 1:
        log_print('ERROR',
                  f"[fine_bootstrap] no source SPWs for fine product {fine_spw}; skipping.")
        return None
    local_casa_spw = '0' if local_spw_count == 1 else f'0~{local_spw_count - 1}'
    local_sp_index = ','.join(str(idx) for idx in range(local_spw_count))
    caltb = os.path.join(workdir, f'caltb_fine_bootstrap_sp{fine_spwstr}.pha')
    parent_column = str(parent_data_column).upper()
    if parent_column.startswith('CORRECTED'):
        split_data_column = 'corrected'
    elif parent_column == 'DATA':
        split_data_column = 'data'
    else:
        log_print('ERROR',
                  f"[fine_bootstrap] unsupported parent data column {parent_data_column!r}; "
                  f"skipping fine product {fine_spw}.")
        return None

    def cleanup_scratch():
        for path in (fine_msfile, f'{fine_msfile}.flagversions', caltb,
                     f'{parent_msfile}.tmpms'):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)

    cleanup_scratch()
    fine_result = None
    try:
        with pipeline_stage('fine_bootstrap:split', spw=fine_spw, spwstr=fine_spwstr):
            mstl.splitX(parent_msfile, outputvis=fine_msfile,
                        spw=source_sp_index, datacolumn=split_data_column,
                        datacolumn2='MODEL_DATA')
        if not os.path.isdir(fine_msfile):
            raise RuntimeError(f'fine child MS was not created: {fine_msfile}')

        with pipeline_stage('fine_bootstrap:gaincal', spw=fine_spw, spwstr=fine_spwstr):
            gaincal(vis=fine_msfile, caltable=caltb, selectdata=True,
                    uvrange='', spw=local_casa_spw, combine='scan',
                    antenna=f'{antenna}&{antenna}', refant='0', solint='inf',
                    refantmode='flex', gaintype='G', gaintable=[],
                    minsnr=1.0, calmode='p', append=False)
        if not _caltable_has_unflagged_solutions(caltb):
            raise RuntimeError(f'fine phase caltable has no unflagged solutions: {caltb}')

        with pipeline_stage('fine_bootstrap:applycal', spw=fine_spw, spwstr=fine_spwstr):
            applycal(vis=fine_msfile, selectdata=True, antenna=antenna,
                     spw=local_casa_spw, gaintable=[caltb], interp='linear',
                     calwt=False, applymode='calonly')

        final_kwargs = dict(final_imaging_kwargs)
        final_kwargs.update(msfile=fine_msfile, spw=fine_spw,
                            spwstr=fine_spwstr, sp_index=local_sp_index,
                            data_column='CORRECTED_DATA')
        fine_result = _run_final_imaging(**final_kwargs)
    except Exception:
        log_print('ERROR',
                  f"[fine_bootstrap] failed for fine SPW {fine_spw}; skipping without "
                  f"from-scratch fallback.\n{traceback.format_exc()}")
    finally:
        cleanup_scratch()
    return fine_result


_FINE_JOINT_FALLBACK = object()  # sentinel: caller must fall back to the legacy per-chunk loop


def _fine_joint_channel_division_frequencies(freq_setup, chunks):
    """Compute wsclean ``-channel-division-frequencies`` boundaries (Hz) for one group.

    For chunks ``k=1..N-1``, the boundary is the midpoint (in Hz) between the
    highest frequency of chunk ``k-1`` (the upper edge of its highest-index SPW)
    and the lowest frequency of chunk ``k`` (the lower edge of its lowest-index
    SPW). Frequencies are sourced from :class:`FrequencySetup`'s ``eofreq``
    (per-SPW-index band-center frequency, GHz) and ``bandwidth`` (per-band width,
    GHz); these are already used elsewhere in this module
    (:meth:`FrequencySetup.get_reffreq_and_cdelt`) to derive per-SPW frequency
    coverage, so no separate MS SPW-table read is needed.

    :param freq_setup: The active :class:`FrequencySetup` instance.
    :type freq_setup: FrequencySetup
    :param chunks: Ordered list of fine chunks (lowest to highest frequency),
        each a dict with at least key ``'spw'`` (range string, e.g. ``'44~45'``).
    :type chunks: list(dict)
    :returns: Boundary frequencies in Hz, strictly increasing, length ``len(chunks) - 1``.
    :rtype: list(float)
    :raises AssertionError: If the computed boundaries are not strictly increasing.
    """
    half_bw_ghz = freq_setup.bandwidth / 2.0
    edges = []
    for chunk in chunks:
        start, end = _spw_range_bounds(chunk['spw'])
        low_ghz = freq_setup.eofreq[start] - half_bw_ghz
        high_ghz = freq_setup.eofreq[end] + half_bw_ghz
        edges.append((low_ghz, high_ghz))

    boundaries_hz = []
    for k in range(1, len(chunks)):
        prev_high_ghz = edges[k - 1][1]
        cur_low_ghz = edges[k][0]
        boundary_ghz = (prev_high_ghz + cur_low_ghz) / 2.0
        boundaries_hz.append(boundary_ghz * 1.0e9)

    if len(boundaries_hz) > 1:
        assert all(b2 > b1 for b1, b2 in zip(boundaries_hz[:-1], boundaries_hz[1:])), (
            f"[fine_joint_imaging] channel-division-frequencies are not strictly "
            f"increasing: {boundaries_hz} (from chunks={[c['spw'] for c in chunks]})")
    return boundaries_hz


def _restore_image_at_table_beam(image_fits, bmsize):
    """Re-restore one WSClean per-channel image at a forced circular beam.

    The joint fine-spectral WSClean run (:func:`_run_fine_joint_imaging`) fits
    a per-channel circular beam (``-circular-beam`` with no ``-beam-size``),
    which can vary erratically between imaging intervals of the *same* fine
    chunk (e.g. 19.0/14.1/11.4 arcsec seen in practice). Tb bookkeeping
    requires each chunk's FORCED TABLE beam (``chunks[c]['bmsize']``) so the
    conversion matches the legacy per-chunk convention. This calls WSClean's
    ``-restore`` mode (model + residual -> restored image, no cleaning) with
    ``-beam-size``/``-circular-beam`` to rebuild the output at the table beam.

    :param image_fits: Path to the WSClean ``*-image.fits`` output to
        re-restore. Its sibling ``*-model.fits``/``*-residual.fits`` (same
        prefix, suffix swapped) must exist.
    :type image_fits: str
    :param bmsize: Forced circular beam FWHM in arcsec (the chunk's table beam).
    :type bmsize: float
    :returns: Path to the newly written ``*-tbeam-image.fits`` file on success,
        or ``None`` if the model/residual siblings are missing or the wsclean
        subprocess failed.
    :rtype: str or None
    """
    if not image_fits.endswith('-image.fits'):
        log_print('ERROR',
                  f"[fine_joint_imaging] cannot derive model/residual siblings for "
                  f"unexpected filename {image_fits}")
        return None

    prefix = image_fits[:-len('-image.fits')]
    model_fits = prefix + '-model.fits'
    residual_fits = prefix + '-residual.fits'
    output_fits = prefix + '-tbeam-image.fits'

    if not (os.path.exists(model_fits) and os.path.exists(residual_fits)):
        log_print('ERROR',
                  f"[fine_joint_imaging] missing model/residual sibling(s) for "
                  f"{os.path.basename(image_fits)} (model_exists="
                  f"{os.path.exists(model_fits)}, residual_exists="
                  f"{os.path.exists(residual_fits)}); cannot force table beam "
                  f"{bmsize} arcsec, falling back to the un-restored image for "
                  f"this item.")
        return None

    cmd = [WSCLEAN_BIN, '-beam-size', str(bmsize), '-circular-beam',
           '-restore', residual_fits, model_fits, output_fits]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception:
        log_print('ERROR',
                  f"[fine_joint_imaging] wsclean -restore subprocess raised for "
                  f"{os.path.basename(image_fits)}\n{traceback.format_exc()}")
        return None

    if proc.returncode != 0 or not os.path.exists(output_fits):
        log_print('ERROR',
                  f"[fine_joint_imaging] wsclean -restore failed (returncode="
                  f"{proc.returncode}) for {os.path.basename(image_fits)} at "
                  f"beam {bmsize} arcsec | cmd={' '.join(cmd)}\n"
                  f"stdout={proc.stdout}\nstderr={proc.stderr}")
        return None

    return output_fits


def _run_fine_joint_imaging(msfile, sidx, group_spw, group_spwstr, group_sp_index, group_bmsize,
                           chunks, workdir, imgoutdir, msname, ri_final, briggs_val, pols,
                           reftime_daily, viz_timerange, date_str, is_segmented, freq_setup,
                           tr_series_time=None, fits_tag='', data_column='CORRECTED_DATA',
                           solar_antenna_total=None):
    """Run ONE joint-deconvolution WSClean pass over a full coarse group's fine chunks.

    Images all fine sub-chunks of a coarse SPW group in a single WSClean
    invocation using ``-join-channels -channels-out N -channel-division-frequencies
    ... -fit-spectral-pol P``, with the FULL group's ``-spws`` selection so every
    output channel is deconvolved with the full-group uv coverage (rather than
    each fine chunk's sparse ~2-SPW subset, as the legacy per-chunk loop does).
    Each resulting per-channel image is then routed through the same
    registration/export path as the legacy per-chunk imaging
    (:func:`_register_and_export_interval_fits`), producing FITS files with the
    same naming convention (including each chunk's own ``spwstr`` tb-fits name)
    as the legacy path, so downstream merging/gating code is unaffected.

    :param msfile: Input measurement set path.
    :type msfile: str
    :param sidx: Coarse-group index/key (for logging only).
    :param group_spw: Full coarse-group SPW range string (e.g. ``'44~49'``).
    :type group_spw: str
    :param group_spwstr: Formatted group SPW string (see :func:`format_spw`); used
        only for logging/imname here (chunk outputs use each chunk's own spwstr).
    :type group_spwstr: str
    :param group_sp_index: Full group SPW id selection (comma list), passed to
        WSClean's ``-spws``, so the uv coverage is the FULL group's.
    :type group_sp_index: str
    :param group_bmsize: Beam size (arcsec) to use for the single joint WSClean
        call. WSClean fits one beam per output channel when given a single
        ``-beam-size``; using the group's beam size (rather than per-chunk) is
        an accepted simplification for this joint mode.
    :type group_bmsize: float
    :param chunks: Ordered list (lowest to highest frequency) of fine chunk
        dicts, each with keys ``'spw'`` (range string), ``'spwstr'`` (formatted
        string), ``'sp_index'`` (comma id list from
        :func:`_spw_indices_for_range`), and ``'bmsize'`` (per-chunk beam size,
        unused for the WSClean call itself but kept for parity/logging).
    :type chunks: list(dict)
    :param workdir: Working directory for WSClean outputs.
    :param imgoutdir: Output directory for synoptic FITS files.
    :param msname: MS basename used in WSClean image-name construction.
    :param ri_final: Interval dict from :func:`_compute_round_intervals`.
    :param briggs_val: Briggs robust weighting value.
    :param pols: Polarization selection string.
    :param reftime_daily: Daily reference time (for non-segmented restamping).
    :param viz_timerange: CASA timerange string used for the non-segmented path
        (unused when ``is_segmented`` is True; kept for signature parity with
        :func:`_run_final_imaging`).
    :param date_str: Date string (``YYYYMMDD``).
    :param is_segmented: Whether this is the segmented (multi-interval) path.
    :param freq_setup: Active :class:`FrequencySetup` instance.
    :param tr_series_time: Optional list of (start_Time, end_Time) tuples to
        filter imaging intervals.
    :param fits_tag: Optional infix spliced into output FITS filenames.
    :param data_column: MS data column to image.
    :param solar_antenna_total: Total solar antenna count for this epoch, or None.
    :returns: On success, ``dict`` mapping each chunk's SPW range string to its
        list of written synoptic FITS file paths (one entry per chunk, in the
        same order as ``chunks``). On failure (output-count mismatch), returns
        the module-level sentinel :data:`_FINE_JOINT_FALLBACK` so the caller can
        fall back to the legacy per-chunk imaging loop.
    :rtype: dict or object
    """
    gain = 0.2
    n_chunks = len(chunks)
    fit_spectral_pol = PIPELINE_CONFIG.get('fine_spectral_fit_spectral_pol', 2)

    try:
        boundaries_hz = _fine_joint_channel_division_frequencies(freq_setup, chunks)
    except AssertionError:
        log_print('ERROR',
                  f"[fine_joint_imaging] {traceback.format_exc()}")
        return _FINE_JOINT_FALLBACK

    antenna_ids = unflagged_solar_antenna_ids(
        msfile, group_sp_index, solar_antenna_total)
    n_ant_img = None if antenna_ids is None else len(antenna_ids)
    if n_ant_img is not None and solar_antenna_total is not None:
        log_print('INFO',
                  f"Group SPW {group_spwstr}: {n_ant_img}/{int(solar_antenna_total)} solar antennas "
                  f"have unflagged data available for joint fine-spectral imaging.")

    with pipeline_stage("fine_joint_imaging", spw=group_spw, spwstr=group_spwstr,
                        n_chunks=n_chunks, segmented=is_segmented):
        imname_strlist = ["eovsa", "major", f"{msname}", f"sp{group_spwstr}", 'finejoint']
        imname = '-'.join(imname_strlist)
        clean_obj = ww.WSClean(msfile)
        clean_obj.setup(size=1024, scale="2.5asec", pol=pols,
                        weight_briggs=briggs_val,
                        niter=20000, mgain=0.85, gain=gain,
                        data_column=data_column,
                        name=os.path.join(workdir, imname),
                        multiscale=True, multiscale_gain=0.3,
                        multiscale_scale_bias=0.6,
                        auto_mask=2, auto_threshold=1,
                        minuv_l=200,
                        intervals_out=ri_final['N1'],
                        no_negative=False, quiet=True,
                        # circular beam PER OUTPUT CHANNEL (no forced beam_size):
                        # forcing the single group-level beam on all channels skews
                        # the Tb conversion (Tb ~ 1/beam_area) of off-reference
                        # chunks by the beam-area ratio — e.g. +40% at the low edge
                        # of s05-10 — which spuriously trips the gate's peak
                        # backstop. With -circular-beam alone, wsclean fits a
                        # frequency-appropriate circular beam for each channel and
                        # the per-chunk FITS headers drive the correct conversion.
                        circular_beam=True,
                        spws=group_sp_index,
                        join_channels=True,
                        channels_out=n_chunks,
                        channel_division_frequencies=boundaries_hz,
                        fit_spectral_pol=fit_spectral_pol)
        clean_obj.run(dryrun=False)

        # Discover the actual output naming at runtime rather than assuming it
        # (wsclean conventionally uses '-t####' for intervals and '-####' for
        # channels, plus a per-interval '-MFS' set, but this is not guessed).
        all_fits = sorted(glob(os.path.join(workdir, imname + '-*image.fits')))
        # Also cover the no-'-t####'-infix case wsclean uses when intervals_out==1.
        all_fits += sorted(glob(os.path.join(workdir, imname + '-image.fits')))
        all_fits = sorted(set(all_fits))

        interval_re = re.compile(r'-t(\d+)-')
        channel_re = re.compile(r'-(\d+)-image\.fits$')
        mfs_re = re.compile(r'-MFS-image\.fits$')

        per_interval_channel = {}  # (interval_idx, channel_idx) -> filepath
        mfs_files = []
        for fpath in all_fits:
            base = os.path.basename(fpath)
            if mfs_re.search(base):
                mfs_files.append(fpath)
                continue
            interval_match = interval_re.search(base)
            interval_idx = int(interval_match.group(1)) if interval_match else 0
            channel_match = channel_re.search(base)
            if not channel_match:
                # Neither MFS nor a per-channel file; not part of the expected set.
                continue
            channel_idx = int(channel_match.group(1))
            per_interval_channel[(interval_idx, channel_idx)] = fpath

        n_intervals = int(ri_final['N1'])
        expected_count = n_intervals * n_chunks
        found_count = len(per_interval_channel)
        log_print('INFO',
                  f"[fine_joint_imaging] group {group_spwstr}: discovered {found_count} "
                  f"per-channel image(s) (+{len(mfs_files)} MFS) vs expected "
                  f"{expected_count} ({n_intervals} interval(s) x {n_chunks} channel(s)).")

        if found_count != expected_count:
            log_print('ERROR',
                      f"[fine_joint_imaging] output count mismatch for group {group_spwstr}: "
                      f"found {found_count}, expected {expected_count}. Falling back to the "
                      f"legacy per-chunk fine-imaging loop.")
            clean_junk(os.path.join(workdir, imname))
            return _FINE_JOINT_FALLBACK

        # wsclean orders channels by frequency; EOVSA SPW index increases with
        # frequency, so channel index c (0-based, ascending frequency) maps to
        # chunk c (chunks are ordered lowest to highest frequency by the caller).
        chunk_synfitsfiles = {chunk['spw']: [] for chunk in chunks}
        for channel_idx, chunk in enumerate(chunks):
            chunk_tag = f"fine:{chunk['spwstr']}"
            log_print('INFO',
                      f"[fine_joint_imaging] mapping chunk_tag={chunk_tag} <-> channel_idx={channel_idx}")
            fitsname = []
            for interval_idx in range(n_intervals):
                fpath = per_interval_channel.get((interval_idx, channel_idx))
                if fpath is None:
                    log_print('ERROR',
                              f"[fine_joint_imaging] missing image for chunk {chunk_tag}, "
                              f"interval {interval_idx}, channel {channel_idx}. Falling back to "
                              f"the legacy per-chunk fine-imaging loop.")
                    clean_junk(os.path.join(workdir, imname))
                    return _FINE_JOINT_FALLBACK
                log_print('INFO',
                          f"[fine_joint_imaging] chunk_tag={chunk_tag} channel_idx={channel_idx} "
                          f"interval_idx={interval_idx} file={os.path.basename(fpath)}")
                fitsname.append(fpath)
            fitsname = sorted(fitsname)

            if PIPELINE_CONFIG.get('fine_spectral_force_table_beam', True):
                chunk_bmsize = chunk['bmsize']
                restored_count = 0
                for fits_idx, fpath in enumerate(fitsname):
                    restored_path = _restore_image_at_table_beam(fpath, chunk_bmsize)
                    if restored_path is not None:
                        fitsname[fits_idx] = restored_path
                        restored_count += 1
                    # else: fall back to the original (un-restored) image for
                    # this interval; already logged by the helper.
                log_print('INFO',
                          f"[fine_joint_imaging] table-beam re-restore for chunk "
                          f"{chunk_tag}: forced beam={chunk_bmsize} arcsec, "
                          f"restored {restored_count}/{len(fitsname)} interval(s).")

            fitsname_helio = [f.replace('image.fits', 'image.helio.fits') for f in fitsname]
            fitsname_helio_ref_daily = [f.replace('image.fits', 'image.helio.ref_daily.fits') for f in fitsname]
            try:
                if is_segmented:
                    hf.imreg(vis=msfile, imagefile=fitsname, fitsfile=fitsname_helio,
                             timerange=ri_final['timeranges'], toTb=True)
                    for eoidx, (fits_helio, fits_helio_ref_daily) in enumerate(
                            zip(fitsname_helio, fitsname_helio_ref_daily)):
                        solar_diff_rot_heliofits(fits_helio, reftime_daily, fits_helio_ref_daily,
                                                 in_time=ri_final['time_intervals_major_avg'][eoidx])
                    fitsfilefinal = fitsname_helio_ref_daily
                else:
                    hf.imreg(vis=msfile, imagefile=fitsname, fitsfile=fitsname_helio,
                             timerange=[viz_timerange], toTb=True)
                    fitsfilefinal = fitsname_helio
            except Exception:
                log_print('ERROR', f"[fine_joint_imaging] helioprojective registration failed | "
                                   f"{_format_log_state(spw=chunk['spw'], spwstr=chunk['spwstr'], n_fits=len(fitsname))}\n"
                                   f"{traceback.format_exc()}")
                fitsfilefinal = []

            chunk_synfitsfiles[chunk['spw']] = _register_and_export_interval_fits(
                msfile, fitsfilefinal, ri_final, imgoutdir, chunk['spwstr'],
                reftime_daily, date_str, is_segmented,
                n_ant_img, solar_antenna_total, antenna_ids=antenna_ids,
                tr_series_time=tr_series_time, fits_tag=fits_tag)

        # Ignore/delete the joint run's MFS outputs -- the coarse product still
        # comes from the standard Step 4 run, not from this joint fine pass.
        for mfs_file in mfs_files:
            try:
                if os.path.exists(mfs_file):
                    os.remove(mfs_file)
            except Exception:
                log_print('WARNING',
                          f"[fine_joint_imaging] failed to remove MFS output {mfs_file}\n"
                          f"{traceback.format_exc()}")

        clean_junk(os.path.join(workdir, imname))

    log_print('INFO', f"Joint fine-spectral imaging for group SPW {group_spw}: completed")
    return chunk_synfitsfiles


def _stage_pipeline_input(msfile, msname, workdir, preserve_input=False):
    """Return a working MS, copying explicit inputs to pipeline-owned scratch."""
    input_stage_dir = workdir
    if preserve_input:
        input_stage_dir = os.path.join(workdir, '.pipeline_reused_input')
        os.makedirs(input_stage_dir, exist_ok=True)
    msfile_copy = os.path.join(input_stage_dir, f'{msname}.ms')
    if preserve_input and os.path.abspath(msfile) == os.path.abspath(msfile_copy):
        input_stage_dir = os.path.join(workdir, '.pipeline_reused_input_copy')
        os.makedirs(input_stage_dir, exist_ok=True)
        msfile_copy = os.path.join(input_stage_dir, f'{msname}.ms')
    if preserve_input and os.path.exists(msfile_copy):
        shutil.rmtree(msfile_copy, ignore_errors=True)
    if not os.path.exists(msfile_copy):
        if msfile.lower().endswith('.ms'):
            shutil.copytree(msfile, msfile_copy)
        elif msfile.lower().endswith('.ms.tar.gz'):
            subprocess.run(['tar', '-zxf', msfile, '-C', input_stage_dir], check=True)
        else:
            raise ValueError(f"Unsupported file format: {msfile}")
    return msfile_copy


def _path_is_within(path, directory):
    """Return whether ``path`` is on or below ``directory``."""
    path = os.path.abspath(path)
    directory = os.path.abspath(directory)
    return os.path.commonpath((path, directory)) == directory


def _archive_and_publish_outputvis(staged_outputvis, durable_outputvis):
    """Archive a scratch MS and atomically publish it to durable storage."""
    staged_outputvis = os.path.abspath(staged_outputvis)
    durable_outputvis = os.path.abspath(durable_outputvis)
    staged_archive = staged_outputvis + '.tar.gz'
    durable_archive = durable_outputvis + '.tar.gz'

    log_print('INFO', f"Archiving scratch output MS {staged_outputvis} to {staged_archive} ...")
    with tarfile.open(staged_archive, 'w:gz') as tar:
        tar.add(staged_outputvis, arcname=os.path.basename(durable_outputvis))
    if not os.path.exists(staged_archive) or os.path.getsize(staged_archive) <= 0:
        raise RuntimeError(f"Output MS archive is empty or missing: {staged_archive}")

    if staged_archive != durable_archive:
        os.makedirs(os.path.dirname(durable_archive) or '.', exist_ok=True)
        partial_archive = durable_archive + '.partial'
        try:
            if os.path.exists(partial_archive):
                os.remove(partial_archive)
            log_print('INFO', f"Publishing completed output archive to {durable_archive} ...")
            shutil.copy2(staged_archive, partial_archive)
            os.replace(partial_archive, durable_archive)
        except Exception:
            if os.path.exists(partial_archive):
                os.remove(partial_archive)
            raise
        os.remove(staged_archive)

    shutil.rmtree(staged_outputvis)
    log_print('INFO', f"Published durable output archive {durable_archive}")
    return durable_archive


def pipeline_run(vis, outputvis='', workdir=None, slfcaltbdir=None, imgoutdir=None,
                 figoutdir=None, clearcache=False, clearlargecache=False,
                 pols='XX', verbose=True, hanning=False, do_sbdcal=False,
                 overwrite=False, overwrite_caltb=True, mergeFITSonly=False,
                 niter_init=None, ncpu='auto', tr_series_imaging=None,
                 spws_imaging=None, fits_tag='', fine_spectral_imaging=False,
                 fine_spectral_only=False, custom_spws=None, force_feature_selfcal=False,
                 imaging_only=False, adaptive_s00=False, preserve_input=False):
    """
    Executes the EOVSA data processing pipeline for solar observation data.

    :param vis: Path to the input measurement set (MS) data.
    :type vis: str
    :param outputvis: Output path for the processed MS, defaults to an empty string.
    :type outputvis: str, optional
    :param workdir: Working directory for intermediate data, defaults to None which uses './'.
    :type workdir: str, optional
    :param slfcaltbdir: Directory for storing calibration tables, defaults to None.
    :type slfcaltbdir: str, optional
    :param imgoutdir: Output directory for image files, defaults to None.
    :type imgoutdir: str, optional
    :param figoutdir: Output directory for diagnostic figures, defaults to None.
    :type figoutdir: str, optional
    :param clearcache: If True, clears all cache files after processing, defaults to False.
    :type clearcache: bool, optional
    :param clearlargecache: If True, clears large cache files (model directories) after processing.
        When set, clearcache is forced to False. Defaults to False.
    :type clearlargecache: bool, optional
    :param pols: Polarization types to process, defaults to 'XX'.
    :type pols: str, optional
    :param verbose: Enables verbose output during processing, defaults to True.
    :type verbose: bool, optional
    :param hanning: If True, applies Hanning smoothing to the data, defaults to False.
    :type hanning: bool, optional
    :param do_sbdcal: Boolean flag to perform single-band delay calibration, defaults to False.
    :type do_sbdcal: bool
    :param overwrite: If True, overwrites existing files, defaults to False.
    :type overwrite: bool, optional
    :param overwrite_caltb: If True, overwrites existing calibration tables, defaults to True.
    :type overwrite_caltb: bool, optional
    :param mergeFITSonly: If True, skips all processing and only merges existing FITS files.
    :type mergeFITSonly: bool, optional
    :param niter_init: Override number of iterations for the init self-cal round.
        If None, uses the value from SELFCAL_ROUNDS config.
    :type niter_init: int, optional
    :param ncpu: Number of CPUs for parallel processing. 'auto' uses all available cores.
    :type ncpu: str or int, optional
    :param tr_series_imaging: Time ranges for imaging as list of time range strings.
        If provided, only processes intervals that overlap with these ranges.
    :type tr_series_imaging: list, optional
    :param spws_imaging: Spectral windows to process. If provided, overrides the default spwidx2proc.
    :type spws_imaging: list, optional
    :param custom_spws: Override the default FrequencySetup SPW grouping with
        explicit ranges such as ``['0~1', '2~4', '5~7']``. With fine imaging,
        the exact :data:`FINE_SPECTRAL_SPWS_52BAND` plan is instead interpreted
        as requested fine outputs and preserves the default seven parents.
    :type custom_spws: list or str, optional
    :param fine_spectral_imaging: If True, bootstrap each finer SPW chunk from
        its freshly imaged coarse parent's ``MODEL_DATA``, run one phase-only
        solve on an isolated child MS, and then run fine final imaging.
    :type fine_spectral_imaging: bool, optional
    :param fine_spectral_only: Deprecated parentless resume mode. The request
        is rejected because an archive MS has no trusted coarse-final
        ``MODEL_DATA`` checkpoint; use ``imaging_only`` with
        ``fine_spectral_imaging`` to regenerate each parent first.
    :type fine_spectral_only: bool, optional
    :param imaging_only: If True, skip preprocessing and self-calibration and
        run final (coarse + fine if fine_spectral_imaging) imaging from an
        existing selfcal MS product. Fine imaging requires the seven-parent
        provenance marker written by the current pipeline.
    :type imaging_only: bool, optional
    :param preserve_input: Stage ``vis`` as a pipeline-owned working copy so
        preprocessing, selfcal, archiving, and cache cleanup cannot mutate or
        remove the explicitly selected source artifact.
    :type preserve_input: bool, optional
    :param adaptive_s00: Opt in to the adaptive central-window s00 route.
        The default ``False`` preserves historical segmented full-track s00
        imaging. The opt-in is honored only for SPW ``(0, 1)`` when no
        ``tr_series_imaging`` filter is supplied.
    :type adaptive_s00: bool, optional
    :return: Dictionary mapping coarse indexes and fine SPW keys to output FITS file paths.
    :rtype: dict
    """
    if fine_spectral_only:
        fine_spectral_imaging = True
    custom_spws, requested_fine_spws = _resolve_fine_spectral_spw_mode(
        custom_spws, fine_spectral_imaging)
    if fine_spectral_only:
        raise RuntimeError(
            "fine_spectral_only is no longer safe because an archive MS has no trusted "
            "coarse-final MODEL_DATA checkpoint. Use imaging_only=True with "
            "fine_spectral_imaging=True to regenerate each parent first.")
    if requested_fine_spws is not None:
        log_print('INFO',
                  "Recognized the exact 52-band fine-output plan: preserving the default "
                  "seven parent groups for self-calibration and bootstrapping its 22 "
                  "strict child products.")

    with pipeline_stage(
        "pipeline_run",
        vis=vis,
        outputvis=outputvis,
        workdir=workdir,
        mergeFITSonly=mergeFITSonly,
        clearcache=clearcache,
        clearlargecache=clearlargecache,
        overwrite=overwrite,
        ncpu=ncpu,
        spws_imaging=spws_imaging,
        custom_spws=','.join(custom_spws) if custom_spws else None,
        requested_fine_spws=','.join(requested_fine_spws) if requested_fine_spws else None,
        fine_spectral_imaging=fine_spectral_imaging,
        fine_spectral_only=fine_spectral_only,
        imaging_only=imaging_only,
        adaptive_s00=adaptive_s00,
        preserve_input=preserve_input,
    ):
        log_print(
            'INFO',
            "s00 imaging route: {0}; adaptive_s00={1}. "
            "The historical segmented full-track route is the default; "
            "adaptive central-window imaging requires explicit opt-in."
            .format('adaptive opt-in' if adaptive_s00 else 'full-track default', adaptive_s00))
        if outputvis and os.path.exists(outputvis) and not overwrite and not fine_spectral_only:
            log_print('INFO', f"Output MS file {outputvis} already exists. Skipping processing.")
            return outputvis

        if not mergeFITSonly:
            _require_wsclean_available()

        # --- Handle clearcache / clearlargecache interaction ---
        if clearlargecache:
            clearcache = False

        # --- Resolve ncpu ---
        if ncpu == 'auto':
            import multiprocessing
            ncpu = multiprocessing.cpu_count()
        else:
            ncpu = int(ncpu)

        # --- Override niter_init in SELFCAL_ROUNDS if specified ---
        if niter_init is not None:
            for rd in SELFCAL_ROUNDS:
                if rd['name'] == 'init':
                    rd['niter'] = niter_init
                    break

        # --- Override spwidx2proc if spws_imaging is specified ---
        if spws_imaging is not None:
            spwidx2proc = [int(s) for s in spws_imaging]
        elif custom_spws is not None:
            spwidx2proc = list(range(len(custom_spws)))
        else:
            spwidx2proc = PIPELINE_CONFIG['spwidx2proc']

        bright_thresh_ = np.array(PIPELINE_CONFIG['bright_thresh'])
        tb_ratio_thresh_ = np.array(PIPELINE_CONFIG['tb_ratio_thresh'])
        briggs_ = PIPELINE_CONFIG['briggs']

        if workdir is None:
            workdir = './'
        if not workdir.endswith('/'):
            workdir += '/'
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)
        if slfcaltbdir is None:
            slfcaltbdir = workdir + '/'
        if imgoutdir is None:
            imgoutdir = workdir + '/'
        if figoutdir is None:
            figoutdir = workdir + '/'
        os.makedirs(figoutdir, exist_ok=True)

        msfile = os.path.normpath(vis)

        # Get the base name without any of the known extensions
        basename = os.path.basename(msfile)
        if basename.lower().endswith('.ms.tar.gz'):
            msname = basename[:-len('.ms.tar.gz')]
        elif basename.lower().endswith('.ms'):
            msname = basename[:-len('.ms')]
        else:
            raise ValueError(f"Unsupported file format: {basename}")

        msfile_copy = _stage_pipeline_input(
            msfile, msname, workdir, preserve_input=True)
        if preserve_input:
            log_print('INFO', f"Staged explicit input without modifying source: {msfile} -> {msfile_copy}")
        msfile = msfile_copy
        # Repair the verified Sep 1--9 no-Ant3 intervals for every input route,
        # including a reused daily MS that bypassed calibration. Other dates
        # and tracking intervals keep their existing flags.
        from suncasa.eovsa.antenna_flagging import flag_ant3_stow
        ant3_flag_report = flag_ant3_stow(
            msfile, provenance_path=os.path.join(imgoutdir, msname + '.ant3_flags.json'))
        log_print('INFO', 'Ant3 observing-state flags: ' + json.dumps(ant3_flag_report))
        if ant3_flag_report.get('cells_newly_flagged', 0):
            overwrite = True
            for stale_path in (msfile + '.flagversions', msfile + '.hanning'):
                if os.path.exists(stale_path):
                    _remove_existing_path_or_raise(stale_path, 'pre-stow-flag cache')
        if fine_spectral_only or imaging_only:
            # Resume-mode archives are built by per-coarse-group split + concat,
            # which leaves the main table GROUP-BLOCKED: TIME is non-monotonic
            # across the concatenated sub-MS blocks. wsclean's -intervals-out
            # mis-partitions such an MS — each group's data lands in only a
            # contiguous subset of the time intervals and the rest image as
            # blank (verified on a real scoped archive: 3 OK + 9 BLANK of 12
            # intervals unsorted vs 12/12 OK after time-sorting). Sort in place.
            tb.open(msfile)
            try:
                _time_col = tb.getcol('TIME')
            finally:
                tb.close()
            if not np.all(np.diff(_time_col) >= 0):
                msfile_sorted = os.path.join(workdir, f'{msname}.tsorted.ms')
                log_print('INFO',
                          f"Archive MS main table is not time-sorted (group-blocked concat); "
                          f"time-sorting {msfile} in place for interval-based imaging.")
                if os.path.exists(msfile_sorted):
                    shutil.rmtree(msfile_sorted, ignore_errors=True)
                ms.open(msfile, nomodify=False)
                try:
                    ms.timesort(msfile_sorted)
                finally:
                    ms.close()
                shutil.rmtree(msfile, ignore_errors=True)
                shutil.move(msfile_sorted, msfile)
            del _time_col
        msfile, msname = _prepare_yy_as_xx_ms(msfile, workdir, msname, pols, overwrite=overwrite)
        if str(pols).upper().replace(',', '') in ('YY', 'YY_AS_XX'):
            log_print('INFO', "Using YY data swapped into the XX slot for downstream diagnostic imaging")
            pols = 'XX'
        final_data_column = 'CORRECTED_DATA'
        if fine_spectral_only or imaging_only:
            final_data_column = 'CORRECTED_DATA' if _ms_has_column(msfile, 'CORRECTED_DATA') else 'DATA'
            log_print('INFO',
                      f"fine_spectral_only={fine_spectral_only}, imaging_only={imaging_only}: "
                      f"using {final_data_column} for final imaging from {msfile}")

        viz_timerange = ant_trange(msfile)
        (tstart, tend) = viz_timerange.split('~')
        tbg_msfile = Time(qa.quantity(tstart, 's')['value'], format='mjd').to_datetime()
        ted_msfile = Time(qa.quantity(tend, 's')['value'], format='mjd').to_datetime()
        tmid_msfile = tbg_msfile + (ted_msfile - tbg_msfile) / 2

        antenna = '0~12'
        solar_antenna_total = 13
        if tbg_msfile >= EOVSA15_UPGRADE_DATE.to_datetime():
            antenna = '0~14'
            solar_antenna_total = 15
        ## date_str is set to the day 1 if the time is between 08:00 UT on day 1 to 08:00 UT(+1) on day 2
        if tbg_msfile.time() < time(8, 0):
            date_local = tbg_msfile.date() - timedelta(days=1)
        else:
            date_local = tbg_msfile.date()
        date_str = date_local.strftime('%Y%m%d')
        reftime_daily = Time(datetime.combine(date_local, time(20, 0)))
        if requested_fine_spws is not None and Time(tbg_msfile).mjd <= SPW_EPOCH_SPLIT_MJD:
            raise ValueError(
                "The exact 24-product fine SPW plan is only valid for the 52-band EOVSA setup")
        freq_setup = FrequencySetup(Time(tbg_msfile), spws=custom_spws)
        # fine_spectral_only/imaging_only resume from an archive MS assembled via
        # per-coarse-group split()+concat, which CASA renumbers from 0. For a
        # SCOPED archive that renumbering is not the identity, so original SPW
        # indices (as computed throughout this module from FrequencySetup) must
        # be remapped to archive-row indices before being handed to WSClean's
        # -spws or any MS-row lookup. Normal runs (both flags False) leave
        # spw_remap as None, and every application site below no-ops on None.
        spw_remap = _archive_spw_index_remap(msfile, freq_setup) if (fine_spectral_only or imaging_only) else None
        spws_indices = freq_setup.spws_indices
        spw_config_indices = freq_setup.spw_config_indices
        tb_models = {}
        outfits_all = {}
        bright = {}
        bright_thresh = {}
        tb_ratio_thresh = {}
        segmented_imaging = {}
        briggs = {}
        bright_thresh_source = {}
        fits_mask = {}
        imaging_objs = {}
        synfits_manifest = {}
        s00_policy_by_key = {}
        fine_imaging_spws = {}
        fine_postprocess_items = []
        spws = freq_setup.spws
        defaultfreq = freq_setup.defaultfreq
        freq = defaultfreq
        nbands = freq_setup.nbands
        fine_output_spws = requested_fine_spws
        if (fine_spectral_imaging and fine_output_spws is None
                and custom_spws is None and nbands == 52):
            fine_output_spws = _normalize_spw_list(FINE_SPECTRAL_SPWS_52BAND)
        if (imaging_only and fine_spectral_imaging and nbands == 52
                and not _has_fine_bootstrap_parent_marker(msfile)):
            raise RuntimeError(
                f"[fine_bootstrap] {msfile} lacks the seven-parent checkpoint provenance "
                f"marker required for imaging-only fine products. Remint the archive with "
                f"the current seven-parent pipeline before retrying."
            )

        for sidx, sp_index in enumerate(spws_indices):
            config_idx = spw_config_indices[sidx]
            tb_models[sidx] = None
            outfits_all[sidx] = None
            bright[sidx] = False
            imaging_objs[sidx] = None
            reffreq, _ = freq_setup.get_reffreq_and_cdelt(spws[sidx])
            reffreq_ghz = float(reffreq.rstrip('GHz'))
            if custom_spws is None:
                bright_thresh[sidx] = float(bright_thresh_[config_idx])
                bright_thresh_source[sidx] = 'table-anchor'
            else:
                bright_thresh[sidx] = feature_brightness_threshold(reffreq_ghz)
                bright_thresh_source[sidx] = 'smooth-log-pchip'
            tb_ratio_thresh[sidx] = tb_ratio_thresh_[config_idx]
            # segmented_imaging[sidx] = True if sidx in [0, 1, 2, 3] else False
            ## testing with segmented imaging for all bands for now with the npz calibration, will revert to the above after validation
            segmented_imaging[sidx] = True if config_idx in [0, 1, 2, 3, 4, 5, 6] else False
            briggs[sidx] = briggs_[config_idx]

        fine_spectral_exclude_bounds = {
            _spw_range_bounds(excl) for excl in PIPELINE_CONFIG.get('fine_spectral_exclude', [])
        }
        if fine_spectral_imaging:
            for sidx, spw in enumerate(spws):
                if sidx not in spwidx2proc:
                    continue
                if _spw_range_bounds(spw) in fine_spectral_exclude_bounds:
                    log_print('INFO',
                              f"fine_spectral_exclude: skipping fine-spectral products for coarse group "
                              f"{spw} (sidx={sidx}); matched fine_spectral_exclude config.")
                    fine_imaging_spws[sidx] = []
                    continue
                fine_spws = _fine_spectral_children_for_parent(
                    spw, requested_spws=fine_output_spws)
                fine_imaging_spws[sidx] = fine_spws
            if mergeFITSonly:
                for sidx, fine_spws in fine_imaging_spws.items():
                    fine_postprocess_items.extend(
                        (f'fine:{format_spw(fine_spw)}', fine_spw, segmented_imaging[sidx])
                        for fine_spw in fine_spws)
            else:
                for fine_spws in fine_imaging_spws.values():
                    for fine_spw in fine_spws:
                        _remove_fine_spectral_products(
                            imgoutdir, date_str, format_spw(fine_spw), fits_tag=fits_tag)

        dsize, fdens = calc_diskmodel(tmid_msfile, nbands, freq)
        fdens = fdens / 2.0  # convert Stokes I to single-linear-pol flux density

        uvmin_l_str = _compute_uv_ranges(spws_indices, spws, freq, spwidx2proc)

        run_start_time = datetime.now()

        # --- Pre-processing: flagging, hanning smoothing ---
        if fine_spectral_only or imaging_only:
            log_print('INFO',
                      f"fine_spectral_only={fine_spectral_only}, imaging_only={imaging_only}: "
                      f"skipping preprocessing flag edits.")
        elif os.path.isdir(msfile + '.flagversions'):
            if verbose:
                log_print('INFO', f'Flagversion of {msfile} already exists. Skipped...')
        else:
            if verbose:
                log_print('INFO', f'Flagging any high amplitudes viz from flares or RFIs in {msfile}')
            with pipeline_stage("preprocess:flagging", msfile=msfile):
                flagmanager(msfile, mode='save', versionname='pipeline_init')
                flagdata(vis=msfile, mode="tfcrop", spw='', action='apply', display='',
                         timecutoff=3.0, freqcutoff=2.0, maxnpieces=2, flagbackup=False)
                flagmanager(msfile, mode='save', versionname='pipeline_remove_RFI-and-BURSTS')

        if hanning and not fine_spectral_only and not imaging_only:
            msfile_hanning = msfile + '.hanning'
            if not os.path.exists(msfile_hanning):
                if verbose:
                    log_print('INFO', f'Perform hanningsmooth for {msfile}...')
                with pipeline_stage("preprocess:hanning", msfile=msfile):
                    hanningsmooth(vis=msfile, datacolumn='data', outputvis=msfile_hanning)
            else:
                if verbose:
                    log_print('INFO', f'The hanningsmoothed {msfile} already exists. Skipped...')
            msfile = msfile_hanning

        delmod(msfile)

        wsclean_intervals = WSCleanTimeIntervals(msfile, combine_scans=True, pols=pols)
        wsclean_intervals.interval_length = 50
        wsclean_intervals.compute_intervals(nintervals_minor=3)
        tdur = wsclean_intervals.results()['tdur']

        caltbs_all = []
        slfcal_init_objs = []
        parent_checkpoints_current_run = set()
        imname_init_disk_strlist = ["eovsa", "disk", "init", f"{msname}"]

        # --- Parse tr_series_imaging time ranges ---
        tr_series_time = None
        if tr_series_imaging is not None:
            from eovsapy.util import Time as eoTime
            tr_series_time = []
            for tr in tr_series_imaging:
                if '~' in tr:
                    t0, t1 = tr.split('~')
                    tr_series_time.append((eoTime(t0), eoTime(t1)))
                else:
                    log_print('WARNING', f"Ignoring invalid time range '{tr}'. Expected format: 'start~end'.")

        if mergeFITSonly:
            log_print('INFO', "mergeFITSonly=True: skipping self-calibration and imaging, proceeding to merge.")
        else:
            # --- Main loop over spectral windows ---
            for sidx, sp_index in enumerate(spws_indices):
                spwstr = format_spw(spws[sidx])
                msfile_sp = f'{msname}.sp{spwstr}.slfcaled.ms'
                caltbs = []

                with pipeline_stage("spw", spw=spws[sidx], sidx=sidx, spwstr=spwstr, msfile=msfile):
                    if os.path.exists(msfile_sp):
                        if overwrite:
                            shutil.rmtree(msfile_sp, ignore_errors=True)
                        else:
                            log_print('INFO', f"MS file {msfile_sp} already exists. Skipping processing.")
                            slfcal_init_objs.append(None)
                            caltbs_all.append(caltbs)
                            continue
                    if sidx not in spwidx2proc:
                        slfcal_init_objs.append(None)
                        caltbs_all.append(caltbs)
                        continue

                    # imaging_only resumes from a (possibly scoped) archive MS whose
                    # SPW ids were renumbered by split()+concat; remap this group's
                    # original index string to the archive's row indices before it
                    # reaches any imaging/-spws consumer below. Brightness-check and
                    # disk-selfcal (unmapped uses of `sp_index` further down) are
                    # unreachable whenever spw_remap is not None, since those are
                    # gated by `if not imaging_only:` and spw_remap is only computed
                    # for fine_spectral_only/imaging_only (and this main loop is never
                    # reached when fine_spectral_only=True).
                    if spw_remap is not None:
                        sp_index = _remap_sp_index_str(sp_index, spw_remap)
                        if sp_index is None:
                            log_print('ERROR',
                                      f"[archive_spw_remap] skipping SPW {spws[sidx]} (sidx={sidx}): "
                                      f"SPW index missing from archive remap.")
                            slfcal_init_objs.append(None)
                            caltbs_all.append(caltbs)
                            continue

                    reffreq, cdelt4_real, bmsize = freq_setup.get_reffreq_and_cdelt(spws[sidx], return_bmsize=True)
                    log_print('INFO', f"Processing SPW {spws[sidx]} for msfile {msname} ...")

                    if imaging_only and not segmented_imaging[sidx]:
                        log_print('ERROR',
                                  f"imaging_only=True requires segmented imaging, but SPW {spws[sidx]} "
                                  f"(sidx={sidx}) is not in a segmented group; skipping this SPW. "
                                  f"The non-segmented final-imaging branch is stateful (imaging_objs/"
                                  f"model_ref_name_str) and is not supported for imaging_only resume.")
                        slfcal_init_objs.append(None)
                        caltbs_all.append(caltbs)
                        continue

                    if not imaging_only:
                        # --- Step 1: Brightness check ---
                        with pipeline_stage("brightness_check", spw=spws[sidx], sidx=sidx, spwstr=spwstr):
                            delmod(vis=msfile)
                            imname = '-'.join(imname_init_disk_strlist + [f"sp{spwstr}"])
                            clean_obj = ww.WSClean(msfile)
                            clean_obj.setup(size=1024, scale="2.5asec", pol=pols,
                                            weight_briggs=0.0, niter=500, mgain=0.85,
                                            data_column='DATA',
                                            name=os.path.join(workdir, imname),
                                            multiscale=True, multiscale_gain=0.3,
                                            multiscale_scale_bias=0.6,
                                            auto_mask=6, auto_threshold=3,
                                            intervals_out=1,
                                            no_negative=True, quiet=True,
                                            circular_beam=True, beam_size=bmsize,
                                            spws=sp_index)
                            wsclean_status = clean_obj.run(dryrun=False)
                            if wsclean_status != 0:
                                raise RuntimeError(
                                    f"WSClean brightness check failed with status "
                                    f"{wsclean_status} for SPW {spws[sidx]} using "
                                    f"{WSCLEAN_BIN!r}."
                                )
                            clean_junk(imname)

                        in_fits = os.path.join(workdir, f"{imname}-image.fits")
                        fits_candidates = sorted(glob(os.path.join(workdir, f"{imname}*image.fits")))
                        if not os.path.exists(in_fits):
                            if len(fits_candidates) == 1:
                                in_fits = fits_candidates[0]
                                log_print('WARNING',
                                          f"Exact WSClean FITS not found; using candidate {in_fits} for SPW {spws[sidx]}")
                            else:
                                log_print('ERROR',
                                          f"Expected WSClean FITS not found for SPW {spws[sidx]}: {in_fits}. "
                                          f"Candidates={fits_candidates}")
                                slfcal_init_objs.append(None)
                                caltbs_all.append(caltbs)
                                continue
                        diskstatss = []
                        for i, sp in enumerate(sp_index.split(',')):
                            reffreq, cdelt4_real, bmsize = freq_setup.get_reffreq_and_cdelt(f'{sp}~{sp}', return_bmsize=True)
                            sp = int(sp)
                            out_fits = '-'.join(imname_init_disk_strlist + [f'sp{sp:02d}_adddisk-model.fits'])
                            dsz = float(dsize[sp].rstrip('arcsec'))
                            fdn = fdens[sp]
                            diskstats = add_convolved_disk_to_fits(in_fits, out_fits, dsz, fdn, ignore_data=True,
                                                                   bmaj=bmsize / 3600., rfreq=reffreq,
                                                                   create_mask=(i == 0))
                            diskstatss.append(diskstats)

                        fits_mask[sidx] = os.path.join(workdir, f"{imname}-mask.fits")
                        tb_image = np.nanmean([ds[7] for ds in diskstatss])
                        if np.isnan(tb_image):
                            log_print('ERROR', f"TB image for SPW {spws[sidx]} is NaN. Skipping this SPW.")
                            slfcal_init_objs.append(None)
                            caltbs_all.append(caltbs)
                            continue
                        tb_model = np.nanmean([ds[8] for ds in diskstatss])
                        tb_models[sidx] = tb_model * 1e3
                        snr = np.nanmean([ds[10] for ds in diskstatss])
                        bright_ratio = tb_image / tb_model
                        bright[sidx] = bright_ratio * snr > bright_thresh[sidx]
                        log_print('INFO',
                                  f"SPW {spws[sidx]}: tb_image = {tb_image:.1f} kK, tb_model = {tb_model:.1f} kK, "
                                  f"Ratio = {bright_ratio * snr:.1f}, Thresh = {bright_thresh[sidx]:.1f} "
                                  f"({bright_thresh_source[sidx]}), SNR = {snr:.1f}")

                        if force_feature_selfcal and not bright[sidx]:
                            bright[sidx] = True
                            log_print('INFO', f"SPW {spws[sidx]}: force_feature_selfcal=True override. "
                                              f"Setting bright to True (test mode).")

                        # Key these band-specific SNR floors on the TRUE 7-band index
                        # spw_config_indices[sidx] (recomputed per-band in this loop), not the
                        # positional sidx, so they don't misfire under --custom-spws where
                        # sidx != band index. (config_idx from the earlier setup loop is stale here.)
                        if spw_config_indices[sidx] == 0 and snr >= 5:
                            bright[sidx] = True
                            log_print('INFO', f"SPW {spws[sidx]}: SNR is high enough ({snr:.1f}). Setting bright to True.")
                        if spw_config_indices[sidx] == 1 and snr >= 30:
                            bright[sidx] = True
                            log_print('INFO', f"SPW {spws[sidx]}: SNR is high enough ({snr:.1f}). Setting bright to True.")
                        ## the bright threshold is commented out for npz calibration testing, will reinstate after validation
                        # if bright[sidx]:
                        #     log_print('INFO', f"SPW {spws[sidx]} is bright. Proceeding with segmented imaging.")
                        #     segmented_imaging[sidx] = True
                        # if bright_ratio < tb_ratio_thresh[sidx]:
                        #     segmented_imaging[sidx] = False

                    # --- Compute time intervals for all rounds ---
                    # (needed by Step 4/4b regardless of imaging_only; cheap, no
                    # imaging/gaincal work performed here)
                    round_intervals = {}
                    for round_def in SELFCAL_ROUNDS:
                        mult = round_def['interval_multipliers'].get(spw_config_indices[sidx], 1)
                        round_intervals[round_def['name']] = _compute_round_intervals(wsclean_intervals, mult)
                    mult = FINAL_IMAGING_CONFIG['interval_multipliers'].get(spw_config_indices[sidx], 1)
                    round_intervals['final'] = _compute_round_intervals(wsclean_intervals, mult)

                    if not imaging_only:
                        # --- Step 2: Feature self-calibration (if bright) ---
                        feature_slfcal = bright[sidx]
                        datacolumn = "DATA"
                        if feature_slfcal:
                            slfcal_failed = False
                            slfcal_init_obj = None
                            for round_def in SELFCAL_ROUNDS:
                                slfcal_obj, caltbs, datacolumn = _run_slfcal_round(
                                    round_def, msfile, round_intervals[round_def['name']],
                                    spws[sidx], spwstr, workdir, antenna, caltbs, tdur,
                                    pols, bmsize, fits_mask[sidx], datacolumn,
                                    overwrite_caltb, uvmin_l_str[sidx], do_sbdcal=do_sbdcal,
                                    clearcache=clearcache)
                                if round_def['name'] == 'init':
                                    if slfcal_obj is None:
                                        slfcal_failed = True
                                        break
                                    slfcal_init_obj = slfcal_obj
                            if slfcal_failed:
                                slfcal_init_objs.append(None)
                                caltbs_all.append(caltbs)
                                continue
                            slfcal_init_objs.append(slfcal_init_obj)
                        else:
                            slfcal_init_objs.append(None)

                        # --- Step 3: Disk self-calibration ---
                        ri_init = round_intervals['init']
                        ri_final = round_intervals['final']
                        caltbs = _run_disk_selfcal(
                            msfile, sidx, spws[sidx], spwstr, sp_index, workdir, antenna,
                            caltbs, slfcal_init_objs[sidx], imname_init_disk_strlist,
                            freq_setup, dsize, fdens, ri_init, ri_final, tdur,
                            uvmin_l_str[sidx], overwrite_caltb, pols)
                        caltbs_all.append(caltbs)
                    else:
                        # imaging_only: no feature/disk self-calibration; the MS
                        # already carries the final selfcal solution baked into
                        # its DATA column (final_data_column). Still record an
                        # empty slfcal_init_objs/caltbs entry so downstream
                        # per-sidx bookkeeping (caltbs_all, slfcal_init_objs)
                        # stays aligned with spws_indices.
                        ri_final = round_intervals['final']
                        slfcal_init_objs.append(None)
                        caltbs_all.append(caltbs)
                        log_print('INFO',
                                  f"imaging_only=True: skipping brightness check, feature self-cal, and "
                                  f"disk self-cal for SPW {spws[sidx]}; proceeding directly to final imaging.")

                    adaptive_s00_active = _adaptive_s00_enabled(
                        adaptive_s00, spws[sidx], tr_series_time)
                    s00_policy = None
                    ri_final_coarse = ri_final
                    if _spw_range_bounds(spws[sidx]) == (0, 1):
                        if adaptive_s00_active:
                            log_print(
                                'INFO',
                                "[s00_route] adaptive opt-in active for SPW (0, 1); "
                                "using the central-window policy.")
                        elif adaptive_s00 and tr_series_time is not None:
                            log_print(
                                'INFO',
                                "[s00_route] full-track default for SPW (0, 1); "
                                "adaptive opt-in suppressed by tr_series_imaging.")
                        else:
                            log_print(
                                'INFO',
                                "[s00_route] full-track default for SPW (0, 1); "
                                "pass adaptive_s00=True for adaptive opt-in.")
                    if adaptive_s00_active:
                        removed_s00_products = _begin_s00_attempt(
                            s00_policy_by_key, synfits_manifest, sidx,
                            imgoutdir, date_str, spwstr, fits_tag=fits_tag)
                        if removed_s00_products:
                            log_print(
                                'INFO',
                                f"[s00_policy] removed {len(removed_s00_products)} stale "
                                f"same-route product(s) before the current attempt.")
                        init_obj = slfcal_init_objs[sidx]
                        source_model_fits = _slfcal_source_model_fits(init_obj)
                        s00_policy = _prepare_s00_final_policy(
                            msfile, msname, sp_index, workdir, imgoutdir,
                            wsclean_intervals, reftime_daily, source_model_fits,
                            fits_mask.get(sidx), briggs[sidx], bmsize, pols,
                            final_data_column, fits_tag=fits_tag)
                        if s00_policy is None:
                            log_print(
                                'ERROR',
                                f"[s00_policy] no conservative final-imaging plan for "
                                f"SPW {spws[sidx]}; skipping this product rather than "
                                f"falling back to unconstrained CLEAN.")
                            continue
                        ri_final_coarse = s00_policy['ri']
                        s00_policy_by_key[sidx] = s00_policy

                    # --- Step 4: Final imaging ---
                    synfitsfiles, imaging_objs = _run_final_imaging(
                        msfile, sidx, spws[sidx], spwstr, sp_index, workdir, imgoutdir,
                        msname, ri_final_coarse, briggs[sidx], bmsize, pols,
                        reftime_daily, viz_timerange, date_str,
                        segmented_imaging[sidx], imaging_objs, freq_setup,
                        tr_series_time=tr_series_time, fits_tag=fits_tag,
                        data_column=final_data_column,
                        solar_antenna_total=solar_antenna_total,
                        s00_policy=s00_policy)
                    synfits_manifest[sidx] = list(synfitsfiles)

                    if not segmented_imaging[sidx] and len(synfitsfiles) > 0:
                        outfits_all[sidx] = synfitsfiles[0]

                    # --- Step 4b: Optional finer spectral final imaging ---
                    if fine_spectral_imaging:
                        fine_spws = fine_imaging_spws.get(sidx, [])
                        if fine_spws and not synfitsfiles:
                            log_print('WARNING',
                                      f"[fine_bootstrap] coarse parent SPW {spws[sidx]} produced no "
                                      f"FITS; skipping fine products {fine_spws}.")
                            for fine_spw in fine_spws:
                                _remove_fine_spectral_products(
                                    imgoutdir, date_str, format_spw(fine_spw), fits_tag=fits_tag)
                        elif fine_spws:
                            log_print('INFO',
                                      f"Running parent-bootstrapped fine spectral imaging for "
                                      f"SPW {spws[sidx]}: {fine_spws}")
                            for fine_spw in fine_spws:
                                fine_spwstr = format_spw(fine_spw)
                                fine_key = f'fine:{fine_spwstr}'
                                synfits_manifest[fine_key] = []
                                _remove_fine_spectral_products(
                                    imgoutdir, date_str, fine_spwstr, fits_tag=fits_tag)
                                source_sp_index = _spw_indices_for_range(fine_spw)
                                if spw_remap is not None:
                                    source_sp_index = _remap_sp_index_str(source_sp_index, spw_remap)
                                    if source_sp_index is None:
                                        log_print('ERROR',
                                                  f"[archive_spw_remap] skipping fine chunk {fine_spw} for "
                                                  f"group SPW {spws[sidx]} (sidx={sidx}): SPW index missing "
                                                  f"from archive remap.")
                                        continue
                                _, _, fine_bmsize = freq_setup.get_reffreq_and_cdelt(
                                    fine_spw, return_bmsize=True)
                                fine_msfile = os.path.join(
                                    workdir, f'{msname}.sp{fine_spwstr}.fine-bootstrap.ms')
                                fine_result = _run_bootstrapped_fine_product(
                                    parent_msfile=msfile,
                                    source_sp_index=source_sp_index,
                                    fine_spw=fine_spw,
                                    fine_spwstr=fine_spwstr,
                                    fine_msfile=fine_msfile,
                                    workdir=workdir,
                                    antenna=antenna,
                                    parent_data_column=final_data_column,
                                    final_imaging_kwargs={
                                        'sidx': fine_key,
                                        'workdir': workdir,
                                        'imgoutdir': imgoutdir,
                                        'msname': msname,
                                        'ri_final': ri_final,
                                        'briggs_val': briggs[sidx],
                                        'bmsize': fine_bmsize,
                                        'pols': pols,
                                        'reftime_daily': reftime_daily,
                                        'viz_timerange': viz_timerange,
                                        'date_str': date_str,
                                        'is_segmented': segmented_imaging[sidx],
                                        'imaging_objs': imaging_objs,
                                        'freq_setup': freq_setup,
                                        'tr_series_time': tr_series_time,
                                        'fits_tag': fits_tag,
                                        'solar_antenna_total': solar_antenna_total,
                                    })
                                if fine_result is None:
                                    _remove_fine_spectral_products(
                                        imgoutdir, date_str, fine_spwstr, fits_tag=fits_tag)
                                    continue
                                fine_synfitsfiles, imaging_objs = fine_result
                                if PIPELINE_CONFIG.get('fine_spectral_snr_gate', True):
                                    fine_synfitsfiles = _apply_fine_spectral_quality_gate(
                                        fine_synfitsfiles, synfitsfiles, fine_key,
                                        cfg=PIPELINE_CONFIG)
                                synfits_manifest[fine_key] = list(fine_synfitsfiles)
                                if fine_synfitsfiles:
                                    fine_postprocess_items.append(
                                        (fine_key, fine_spw, segmented_imaging[sidx]))
                                else:
                                    _remove_fine_spectral_products(
                                        imgoutdir, date_str, fine_spwstr, fits_tag=fits_tag)

                    # --- Checkpoint this group's slfcaled visibilities NOW ---
                    # uvsub() in later groups' disk self-cal operates on ALL MS
                    # rows, re-subtracting stale MODEL_DATA from THIS group's
                    # already-imaged rows. Splitting at end-of-run therefore
                    # archived contaminated data for every group but the last
                    # (verified: resume-imaging the end-of-run archive collapsed
                    # the s31-43 coarse peak 9.3x vs the in-chain product).
                    # Split immediately after this group's final imaging so the
                    # archive holds exactly the state imaging consumed.
                    if outputvis and not fine_spectral_only and not imaging_only:
                        _require_ms_column(
                            msfile, 'CORRECTED_DATA',
                            f"Checkpoint split for SPW {spws[sidx]}")
                        if os.path.exists(msfile_sp):
                            shutil.rmtree(msfile_sp, ignore_errors=True)
                        log_print('INFO',
                                  f"Checkpointing SPW {spws[sidx]} (CORRECTED_DATA) -> {msfile_sp}")
                        split(vis=msfile, outputvis=msfile_sp, spw=spws[sidx], datacolumn='corrected')
                        if not os.path.isdir(msfile_sp):
                            log_print('WARNING', f"Per-group checkpoint split failed for SPW {spws[sidx]}.")
                        else:
                            parent_checkpoints_current_run.add(sidx)

                    elapsed_total = (datetime.now() - run_start_time).total_seconds() / 60
                    log_print('INFO', f"Pipeline for SPW {spws[sidx]}: completed in {elapsed_total:.1f} minutes")

        # --- Post-processing: merge FITS, add disk, move caltables ---
        post_tag = f'.{fits_tag}' if fits_tag else ''
        postprocess_items = [] if fine_spectral_only else [
            (sidx, spw, segmented_imaging.get(sidx, False))
            for sidx, spw in enumerate(spws)
        ]
        postprocess_items += fine_postprocess_items
        for out_key, spw, is_segmented in postprocess_items:
            spwstr = format_spw(spw)
            sp_st, sp_ed = _spw_range_bounds(spw)
            dszs = [float(dsize[sp].rstrip('arcsec')) for sp in range(sp_st, sp_ed + 1)]
            fdns = [fdens[sp] for sp in range(sp_st, sp_ed + 1)]
            reffreq, cdelt4_real, bmsize = freq_setup.get_reffreq_and_cdelt(spw, return_bmsize=True)
            outfits = os.path.join(imgoutdir,
                                   f'eovsa.synoptic_daily{post_tag}.{date_str}T200000Z.s{spwstr}.tb.fits')
            outfits_disk = outfits.replace('.tb.fits', '.tb.disk.fits')
            if is_segmented:
                synfitsfiles = _synoptic_merge_inputs(
                    out_key, synfits_manifest, imgoutdir, date_str, spwstr,
                    fits_tag=fits_tag, merge_existing=mergeFITSonly)
                snr_threshold = 3
                if len(synfitsfiles) > 0:
                    log_print('INFO', f"Merging synoptic images for SPW {spwstr} to {outfits} ...")
                    merge_kwargs = {}
                    if out_key in s00_policy_by_key:
                        # The adaptive s00 product is already one selected
                        # central window.  Do not let the generic multi-frame
                        # SNR selector reject that sole, policy-approved image.
                        merge_kwargs['deselect_index'] = np.zeros(
                            len(synfitsfiles), dtype=bool)
                    merge_FITSfiles(
                        synfitsfiles, outfits, overwrite=True,
                        snr_threshold=snr_threshold, reftime=reftime_daily,
                        **merge_kwargs)
                    _promote_imaging_antenna_manifest(synfitsfiles, outfits)
                    outfits_all[out_key] = outfits
                    synfitsfiles_disk = [l.replace('.tb.fits', '.tb.disk.fits') for l in synfitsfiles]
                    try:
                        add_convolved_disk_to_fits(synfitsfiles + [outfits], synfitsfiles_disk + [outfits_disk],
                                                   dszs, fdns, ignore_data=False, toTb=True,
                                                   rfreq=reffreq, bmaj=bmsize / 3600.)
                    except Exception:
                        log_print('ERROR', f"[postprocess:add_disk] failed | "
                                           f"{_format_log_state(spw=spw, spwstr=spwstr, n_synfits=len(synfitsfiles))}\n"
                                           f"{traceback.format_exc()}")
                    _compress_synoptic_fits_in_place(
                        synfitsfiles + [outfits] + synfitsfiles_disk + [outfits_disk])
                else:
                    log_print('WARNING', f"No synoptic images found for SPW {spwstr}. Skipping merge_FITSfiles.")
            else:
                if os.path.exists(outfits):
                    log_print('INFO', f"Add disk to synoptic image {outfits} ...")
                    outfits_all[out_key] = outfits
                    add_convolved_disk_to_fits([outfits], [outfits_disk], dszs, fdns,
                                               ignore_data=False, toTb=True,
                                               rfreq=reffreq, bmaj=bmsize / 3600.)
                    _compress_synoptic_fits_in_place([outfits, outfits_disk])
                else:
                    log_print('WARNING', f"No synoptic image found for SPW {spwstr}. Skipping disk addition.")

        # Move calibration tables to output directory
        log_print('INFO', f"Moving calibration tables to {slfcaltbdir} ...")
        caltbs_comb = [item for sublist in caltbs_all for item in sublist]
        for caltb in caltbs_comb:
            if os.path.exists(caltb):
                os.makedirs(slfcaltbdir, exist_ok=True)
                targetfile = os.path.join(slfcaltbdir,
                                          os.path.basename(caltb).replace("caltb_", f"caltb_{date_str}_"))
                if os.path.exists(targetfile):
                    shutil.rmtree(targetfile, ignore_errors=True)
                shutil.move(caltb, targetfile)
        if outputvis and os.path.isdir(msfile + '.flagversions'):
            targetdir = os.path.dirname(outputvis) or '.'
            targetfile = os.path.join(targetdir, os.path.basename(msfile) + '.flagversions')
            if os.path.abspath(msfile + '.flagversions') != os.path.abspath(targetfile):
                _remove_existing_path_or_raise(targetfile, 'flagversions directory')
                shutil.move(msfile + '.flagversions', targetfile)

        # --- Cache cleanup ---
        if clearcache:
            log_print('INFO', "Clearing all cache files in workdir ...")
            for pattern in ['*.dirty.fits', '*.psf.fits', '*.residual.fits', '*.model.fits',
                             'modelrot_*', '*.slfcaled.ms']:
                for f in glob(os.path.join(workdir, pattern)):
                    if os.path.isdir(f):
                        shutil.rmtree(f, ignore_errors=True)
                    else:
                        os.remove(f)
        elif clearlargecache:
            log_print('INFO', "Clearing large cache files (model directories) in workdir ...")
            for pattern in ['modelrot_*', '*.slfcaled.ms']:
                for f in glob(os.path.join(workdir, pattern)):
                    if os.path.isdir(f):
                        shutil.rmtree(f, ignore_errors=True)
                    else:
                        os.remove(f)

        # --- Assemble selfcal'd outputvis from per-spw splits via concat ---
        if outputvis and not fine_spectral_only and not imaging_only and not os.path.exists(outputvis):
            _require_ms_column(
                msfile, 'CORRECTED_DATA',
                "End-of-run self-calibrated output assembly")
            assembly_outputvis = outputvis
            if not _path_is_within(outputvis, workdir):
                assembly_dir = os.path.join(workdir, '.outputvis_staging')
                os.makedirs(assembly_dir, exist_ok=True)
                assembly_outputvis = os.path.join(
                    assembly_dir,
                    os.path.basename(outputvis),
                )
                if os.path.exists(assembly_outputvis):
                    _remove_existing_path_or_raise(
                        assembly_outputvis,
                        'stale scratch outputvis',
                    )
                if os.path.exists(assembly_outputvis + '.tar.gz'):
                    os.remove(assembly_outputvis + '.tar.gz')
            log_print(
                'INFO',
                f"Building selfcal'd outputvis on processing storage: {assembly_outputvis}",
            )
            os.makedirs(os.path.dirname(assembly_outputvis) or '.', exist_ok=True)
            slfcaled_spw_ms_list = []
            used_end_of_run_checkpoint_fallback = False
            for sidx, sp_index in enumerate(spws_indices):
                spwstr = format_spw(spws[sidx])
                msfile_sp = os.path.join(workdir, f'{msname}.sp{spwstr}.slfcaled.ms')
                if not os.path.isdir(msfile_sp):
                    # Fallback only: groups are checkpointed in-loop right after
                    # their final imaging (see the per-group split above), which
                    # captures the pre-contamination CORRECTED_DATA state. A
                    # split taken HERE (end of run) may carry later groups'
                    # uvsub cross-talk — log loudly.
                    log_print('WARNING',
                              f"No in-loop checkpoint for SPW {spws[sidx]}; splitting at end-of-run "
                              f"(state may include later groups' uvsub cross-talk) -> {msfile_sp}")
                    used_end_of_run_checkpoint_fallback = True
                    split(vis=msfile, outputvis=msfile_sp, spw=spws[sidx], datacolumn='corrected')
                if os.path.isdir(msfile_sp):
                    slfcaled_spw_ms_list.append(msfile_sp)
                else:
                    log_print('WARNING', f"Split failed for SPW {spws[sidx]}; skipping in outputvis.")
            if slfcaled_spw_ms_list:
                log_print('INFO',
                          f"Concatenating {len(slfcaled_spw_ms_list)} per-spw MS files into {assembly_outputvis} ...")
                # timesort=True: without it the concat output is group-blocked
                # (TIME non-monotonic), which breaks wsclean -intervals-out when
                # the archive is later consumed by --imaging-only/--fine-spectral-only.
                concat(vis=slfcaled_spw_ms_list, concatvis=assembly_outputvis, freqtol='', dirtol='', timesort=True)
                log_print('INFO', f"Selfcal'd scratch outputvis saved to {assembly_outputvis}")
                if (_is_default_52_parent_plan(spws)
                        and len(slfcaled_spw_ms_list) == len(spws)
                        and not used_end_of_run_checkpoint_fallback
                        and len(parent_checkpoints_current_run) == len(spws)):
                    try:
                        _write_fine_bootstrap_parent_marker(assembly_outputvis)
                    except OSError:
                        log_print('WARNING',
                                  f"Failed to write fine-bootstrap parent provenance marker in "
                                  f"{assembly_outputvis}; imaging-only fine reruns will be disabled.\n"
                                  f"{traceback.format_exc()}")
                _archive_and_publish_outputvis(assembly_outputvis, outputvis)
            else:
                log_print('WARNING', "No per-spw slfcaled MS files produced; outputvis not created.")

        # --- Tar the source UDB*.ms working copy and remove it ---
        if fine_spectral_only or imaging_only:
            log_print('INFO',
                      f"fine_spectral_only={fine_spectral_only}, imaging_only={imaging_only}: "
                      f"retaining working MS {msfile}.")
        elif os.path.isdir(msfile):
            tar_path = msfile + '.tar.gz'
            log_print('INFO', f"Archiving {msfile} to {tar_path} ...")
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(msfile, arcname=os.path.basename(msfile))
            if os.path.exists(tar_path) and os.path.getsize(tar_path) > 0:
                shutil.rmtree(msfile)
                log_print('INFO', f"Removed {msfile} after successful archiving to {tar_path}")
            else:
                log_print('WARNING', f"Tar archive {tar_path} appears empty or missing. Keeping {msfile}.")

        log_step('INFO', 'pipeline_run', 'completed',
                 processed_spws=sorted(str(key) for key in outfits_all.keys()))
        return outfits_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="EOVSA synoptic imaging pipeline (WSClean). "
                    "Addresses smearing in all-day synthesis images: "
                    "https://www.ovsa.njit.edu/wiki/index.php/All-Day_Synthesis_Issues")
    parser.add_argument('vis', type=str, help='Path to the input measurement set (MS) data.')
    parser.add_argument('--outputvis', type=str, default='', help='Output path for the processed MS.')
    parser.add_argument('--workdir', type=str, help='Working directory for intermediate data.')
    parser.add_argument('--slfcaltbdir', type=str, help='Directory for storing calibration tables.')
    parser.add_argument('--imgoutdir', type=str, help='Output directory for image files.')
    parser.add_argument('--figoutdir', type=str, help='Output directory for figures.')
    parser.add_argument('--clearcache', action='store_true', help='Clears all cache files after processing.')
    parser.add_argument('--clearlargecache', action='store_true',
                        help='Clears the large cache files after processing. If True, clearcache will be set to False.')
    parser.add_argument('--pols', type=str, default='XX', help='Polarization types to process.')
    parser.add_argument('--mergeFITSonly', action='store_true', help='Skips processing and only merges FITS files.')
    # parser.add_argument('--verbose', action='store_true', help='Enables verbose output during processing.')
    # parser.add_argument('--do_diskslfcal', action='store_true', help='Performs disk self-calibration.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrites existing files.')
    parser.add_argument('--niter_init', type=int, default=200, help='Initial number of iterations for imaging.')
    parser.add_argument('--ncpu', type=str, default='auto',
                        help="Specifies the number of CPUs for parallel processing.")
    parser.add_argument('--tr_series_imaging', nargs='*', help='Time ranges for imaging, expects a list of tuples.')
    parser.add_argument('--spws_imaging', nargs='*', help='Spectral windows selected for imaging.')
    parser.add_argument('--custom-spws', nargs='+',
                        help='Override FrequencySetup SPW groupings. With --fine-spectral-imaging, '
                             'the exact 24-product plan preserves the seven default parents.')
    parser.add_argument('--fits_tag', type=str, default='',
                        help='Optional tag inserted into synoptic FITS filenames.')
    parser.add_argument('--fine-spectral-imaging', action='store_true',
                        help='Bootstrap fine SPW chunks from each default parent, run one phase-only '
                             'solve, then run fine final imaging.')
    parser.add_argument('--fine-spectral-only', action='store_true',
                        help='Deprecated parentless resume mode; the request is rejected. Use '
                             '--imaging-only --fine-spectral-imaging to regenerate parents first.')
    parser.add_argument('--imaging-only', action='store_true',
                        help='Skip preprocessing and self-calibration and rerun final '
                             '(coarse + fine if --fine-spectral-imaging) imaging from an '
                             'existing selfcal MS product. Fine resume requires a current '
                             'seven-parent archive marker.')
    parser.add_argument('--force-feature-selfcal', action='store_true',
                        help='TEST ONLY: force feature self-calibration for all processed SPW groups, '
                             'bypassing the brightness gate. Default off.')
    parser.add_argument('--adaptive-s00', action='store_true',
                        help='Opt in to adaptive central-window imaging for s00-01 only. '
                             'Default uses the historical segmented full-track route; '
                             'the opt-in is suppressed when --tr_series_imaging is supplied.')
    parser.add_argument('--hanning', action='store_true', help='Applies Hanning smoothing to the data.')
    parser.add_argument('--do_sbdcal', action='store_true', help='Perform single-band delay calibration.')
    parser.add_argument('--debug_mode', action='store_true', help='Enables debug mode with finer control over parameters.')

    args = parser.parse_args()

    if args.debug_mode:
        log_print('INFO', "Debug mode enabled. Running debug pipeline script instead.")
        log_print('INFO', "Use debug_pipeline_wsclean.py for standalone debug/test runs.")

    pipeline_run(
        vis=args.vis,
        outputvis=args.outputvis,
        workdir=args.workdir,
        slfcaltbdir=args.slfcaltbdir,
        imgoutdir=args.imgoutdir,
        figoutdir=args.figoutdir,
        clearcache=args.clearcache,
        clearlargecache=args.clearlargecache,
        pols=args.pols,
        overwrite=args.overwrite,
        mergeFITSonly=args.mergeFITSonly,
        niter_init=args.niter_init,
        ncpu=args.ncpu,
        tr_series_imaging=args.tr_series_imaging,
        spws_imaging=args.spws_imaging,
        fits_tag=args.fits_tag,
        custom_spws=args.custom_spws,
        fine_spectral_imaging=args.fine_spectral_imaging,
        fine_spectral_only=args.fine_spectral_only,
        imaging_only=args.imaging_only,
        adaptive_s00=args.adaptive_s00,
        force_feature_selfcal=args.force_feature_selfcal,
        hanning=args.hanning,
        do_sbdcal=args.do_sbdcal,
    )
