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
import inspect
import logging
import os
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
from scipy import ndimage
from sunpy import map as smap
from typing import NamedTuple, List, Union
from scipy.signal import fftconvolve
import numbers
from suncasa.casa_compat import import_casatasks
from suncasa.utils import helioimage2fits as hf
from suncasa.utils import mstools as mstl
from suncasa.eovsa import wrap_wsclean as ww
from suncasa.io import ndfits
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface, sun
from scipy import constants
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


def write_compressed_fits(out_fits, data, header, overwrite=True):
    status = ndfits.write(
        out_fits,
        np.asarray(data).copy(),
        header.copy(),
        overwrite=overwrite,
        compression_type='RICE_1',
        quantize_level=4.0,
    )
    if status != 1:
        raise ValueError(f'Failed to write compressed FITS: {out_fits}')
    return out_fits


def write_compressed_fits_from_file(in_fits, out_fits, overwrite=True):
    with fits.open(in_fits, memmap=False) as hdul:
        image_hdu = None
        for hdu in hdul:
            if hdu.data is not None and hdu.header.get('NAXIS', 0) > 0:
                image_hdu = hdu
                break
        if image_hdu is None:
            raise ValueError(f'No image HDU found in {in_fits}')
        data = np.asarray(image_hdu.data).copy()
        header = image_hdu.header.copy()
    return write_compressed_fits(out_fits, data, header, overwrite=overwrite)


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


WSCLEAN_BIN = _resolve_wsclean_bin()
logging.info("Using wsclean executable: %s", WSCLEAN_BIN)

# ============================================================
# Pipeline Configuration Constants
# ============================================================

PIPELINE_CONFIG = {
    'bright_thresh': [350, 900, 2000, 2800, 800, 150, 100],
    'tb_ratio_thresh': [1.5, 5.0, 5.0, 3.0, 1.5, 1.5, 1.5],
    'briggs': [-1.0, -1.0, -1.0, -1.0, -1.0, -0.5, -0.5],
    'spwidx2proc': [0, 1, 2, 3, 4, 5, 6],
}

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
                        showplt=False):
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

    :returns:
      - `select_indices` (bool array): True for images kept (not noisy).
      - `rms_values` (float array): RMS noise for each image.
      - `snr_values` (float array): SNR for each image.
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    # Prepare masks
    images = np.squeeze(data_stack)
    ny, nx = images.shape[1], images.shape[2]
    mask_ring, mask_disk = get_solar_radius_mask(rsun_pix, crpix1, crpix2, ny, nx, rsun_ratio=rsun_ratio)

    # Compute RMS and SNR for all frames (vectorized)
    ring_pixels = images[:, mask_ring]   # (n_images, n_ring_pixels)
    disk_pixels = images[:, mask_disk]   # (n_images, n_disk_pixels)
    ring_rms_values = np.sqrt(np.nanmean(ring_pixels ** 2, axis=1))
    disk_rms_values = np.sqrt(np.nanmean(disk_pixels ** 2, axis=1))
    signals = np.nanpercentile(disk_pixels, 99.999, axis=1)
    snr_values = signals / ring_rms_values

    # --- Automatic RMS threshold (knee detection) ---
    sorted_rms = np.sort(ring_rms_values)
    N = len(sorted_rms)

    pts = np.vstack((np.arange(N), sorted_rms)).T
    p0, pN = pts[0], pts[-1]
    vec = pN - p0
    length = np.linalg.norm(vec)

    rel = pts - p0
    proj = np.dot(rel, vec / length)
    proj_vec = np.outer(proj, vec / length)
    dists = np.linalg.norm(rel - proj_vec, axis=1)

    knee = np.argmax(dists)
    fallback = np.nanmedian(ring_rms_values) + np.nanstd(ring_rms_values)

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

    # Flag noisy: low‐SNR or high‐RMS, but ignore very‐high‐SNR outliers
    noisy = ((snr_values < snr_threshold) |
             (ring_rms_values > rms_threshold)) & \
            (snr_values <= 3 * snr_threshold)

    # Ensure we keep at least two frames
    keep = ~noisy
    if keep.sum() == 0:
        top2 = np.argsort(snr_values)[-2:]
        keep[top2] = True

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
        write_compressed_fits(outf, out_data, hdr, overwrite=True)

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

    def __init__(self, tim=None):
        if tim is None:
            tim = Time.now()
        self.tim = tim
        self.spw2band = np.array([0, 1] + list(range(4, 52)))
        self.bandwidth = 0.325  # 325 MHz
        self.defaultfreq = 1.1 + self.bandwidth * (self.spw2band + 0.5)

        if self.tim.mjd > 58536:
            self.eofreq = self.defaultfreq
            # self.spws = ['0~1', '2~4', '5~10', '11~20', '21~30', '31~40', '41~49']
            self.spws = ['0~1', '2~4', '5~10', '11~20', '21~30', '31~43', '44~49']
            self.nbands = len(self.spw2band)
        else:
            self.bandwidth = 0.5  # 500 MHz
            self.nbands = 34
            self.eofreq = 1.419 + np.arange(self.nbands) * self.bandwidth
            self.spws = ['1~3', '4~9', '10~16', '17~24', '25~30']

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
                    overwrite=True, snr_threshold=None, rms_threshold=None, showplt=False):
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

    Examples
    --------
    Merge three FITS files without exposure time weighting and with on-disk residual suppression:

    >>> merge_FITSfiles(['sun_01.fits', 'sun_02.fits', 'sun_03.fits'], 'sun_merged.fits',
    ...                 overwrite=True)
    """
    from astropy.io import fits
    from astropy.time import Time
    from datetime import datetime, timedelta

    exptimes = []
    data = []
    for fidx, file in enumerate(fitsfilesin):
        with fits.open(file) as hdulist:
            for idx, hdu in enumerate(hdulist):
                if 'CDELT1' in hdu.header:
                    header = hdu.header
                    data.append(np.squeeze(hdu.data))
                    exptimes.append(header['EXPTIME'])
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
    date_obs = Time(header['date-obs']).datetime.replace(hour=20, minute=0, second=0) - timedelta(
        seconds=exptime / 2.0)
    newheader.update({'EXPTIME': exptime, 'DATE-OBS': date_obs.strftime('%Y-%m-%dT%H:%M:%S')})
    newheader['HISTORY'] = 'Merged from multiple images'
    write_compressed_fits(outfits, date_merged, newheader, overwrite=overwrite)
    log_print('INFO',
              f'{np.count_nonzero(deselect_index)} out of {len(fitsfilesin)} images (index{np.where(deselect_index)}) are not selected for merging due to low SNR.')
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
                 reftime_daily=None):
        self.msfile = msfile
        self.time_intervals = time_intervals
        self.time_intervals_major_avg = time_intervals_major_avg
        self.time_intervals_minor_avg = time_intervals_minor_avg
        self.spw = str(spw)
        if '~' in spw:
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


def _prepare_single_pol_ms(msfile, workdir, msname, pols, overwrite=False):
    """Return a single-correlation working MS for YY diagnostic runs."""
    pol = str(pols).upper().replace(',', '')
    yy_as_xx = pol == 'YY_AS_XX'
    if pol not in ('YY', 'YY_AS_XX'):
        return msfile, msname

    pol_msname = f'{msname}.YYasXX' if yy_as_xx else f'{msname}.{pol}'
    pol_msfile = os.path.join(workdir, f'{pol_msname}.ms')
    if os.path.isdir(pol_msfile):
        if overwrite:
            shutil.rmtree(pol_msfile, ignore_errors=True)
        else:
            log_print('INFO', f"Using existing {pol}-only working MS {pol_msfile}")
            return pol_msfile, pol_msname

    log_print('INFO', f"Creating YY-only working MS {pol_msfile} from {msfile}")
    split(vis=msfile, outputvis=pol_msfile, correlation='YY', datacolumn='data')
    if not os.path.isdir(pol_msfile):
        raise RuntimeError(f"Failed to create YY-only working MS: {pol_msfile}")
    if yy_as_xx:
        poltb = os.path.join(pol_msfile, 'POLARIZATION')
        tb.open(poltb, nomodify=False)
        relabeled_rows = 0
        try:
            for row in range(tb.nrows()):
                corr_type = tb.getcell('CORR_TYPE', row)
                corr_product = tb.getcell('CORR_PRODUCT', row)
                if corr_type.shape[0] == 0:
                    continue
                if corr_type.shape[0] != 1:
                    raise RuntimeError(
                        f"Expected one correlation in {poltb} row {row}; got CORR_TYPE shape {corr_type.shape}")
                corr_type[...] = 9  # CASA Stokes enum: XX
                corr_product[...] = 0  # receptor X,X
                tb.putcell('CORR_TYPE', row, corr_type)
                tb.putcell('CORR_PRODUCT', row, corr_product)
                relabeled_rows += 1
        finally:
            tb.close()
        if relabeled_rows == 0:
            raise RuntimeError(f"No populated polarization rows found in {poltb}")
        log_print('INFO', f"Relabeled YY-only working MS as XX for downstream diagnostic path: {pol_msfile} ({relabeled_rows} row(s))")
    return pol_msfile, pol_msname


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
            subprocess.run(cmd, shell=True, check=False)

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
                subprocess.run(cmd, shell=True, check=False)
        log_print('INFO', f'Subtracting disk model from the data for SPW {spw}')
        uvsub(vis=msfile)
        elapsed_sub = (datetime.now() - run_start_sub).total_seconds() / 60
        log_print('INFO', f"Disk subtraction for SPW {spwstr}: completed in {elapsed_sub:.1f} minutes")

    return caltbs


def _run_final_imaging(msfile, sidx, spw, spwstr, sp_index, workdir, imgoutdir,
                       msname, ri_final, briggs_val, bmsize, pols,
                       reftime_daily, viz_timerange, date_str,
                       is_segmented, imaging_objs, freq_setup,
                       tr_series_time=None, fits_tag=''):
    """Run final imaging (segmented or non-segmented) and return output FITS paths.

    :param tr_series_time: List of (start_Time, end_Time) tuples to filter imaging intervals.
        If provided, only processes intervals overlapping with these ranges.
    :param fits_tag: Optional infix (e.g. ``'test'``) spliced after ``eovsa.synoptic``
        / ``eovsa.synoptic_daily`` in the output FITS filenames so alternate-source
        runs (e.g. calwidget_v2 NPZ calibration) do not collide with production
        artefacts.
    """
    gain = 0.2
    reffreq, cdelt4_real, _ = freq_setup.get_reffreq_and_cdelt(spw, return_bmsize=True)

    with pipeline_stage("final_imaging", spw=spw, spwstr=spwstr, segmented=is_segmented):
        if is_segmented:
            imname_strlist = ["eovsa", "major", f"{msname}", f"sp{spwstr}", 'final']
            imname = '-'.join(imname_strlist)
            clean_obj = ww.WSClean(msfile)
            clean_obj.setup(size=1024, scale="2.5asec", pol=pols,
                            weight_briggs=briggs_val,
                            niter=20000, mgain=0.85, gain=gain,
                            data_column='CORRECTED_DATA',
                            name=os.path.join(workdir, imname),
                            multiscale=True, multiscale_gain=0.3,
                            multiscale_scale_bias=0.6,
                            auto_mask=2, auto_threshold=1,
                            minuv_l=200,
                            intervals_out=ri_final['N1'],
                            no_negative=False, quiet=True,
                            circular_beam=True, beam_size=bmsize,
                            spws=sp_index)
            clean_obj.run(dryrun=False)

            fitsname = sorted(glob(os.path.join(workdir, imname + '-t*-image.fits')))
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
                                    data_column="CORRECTED_DATA", pols=pols,
                                    no_negative=False, circular_beam=False,
                                    theoretic_beam=True,
                                    reftime_daily=reftime_daily)
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
            subprocess.run(cmd, shell=True, check=False)
            uvsub(vis=msfile, reverse=True)

            imname_strlist = ["eovsa", "major", f"{msname}", f"sp{spwstr}", 'final']
            imname = '-'.join(imname_strlist)
            clean_obj.setup(size=1024, scale="2.5asec", pol=pols,
                            weight_briggs=briggs_val,
                            niter=20000, mgain=0.85, gain=gain,
                            data_column='CORRECTED_DATA',
                            name=os.path.join(workdir, imname),
                            multiscale=True, multiscale_gain=0.3,
                            multiscale_scale_bias=0.6,
                            auto_mask=2, auto_threshold=1,
                            minuv_l=200,
                            intervals_out=1,
                            no_negative=False, quiet=True,
                            theoretic_beam=True, circular_beam=False,
                            spws=sp_index)
            clean_obj.run(dryrun=False)
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
        log_print('INFO', f"Writing compressed FITS {eofile} to {synfitsfile} ...")
        write_compressed_fits_from_file(eofile, synfitsfile, overwrite=True)

    return synfitsfiles, imaging_objs


def pipeline_run(vis, outputvis='', workdir=None, slfcaltbdir=None, imgoutdir=None,
                 figoutdir=None, clearcache=False, clearlargecache=False,
                 pols='XX', verbose=True, hanning=False, do_sbdcal=False,
                 overwrite=False, overwrite_caltb=True, mergeFITSonly=False,
                 niter_init=None, ncpu='auto', tr_series_imaging=None,
                 spws_imaging=None, fits_tag=''):
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
    :return: Dictionary mapping spectral window index to output FITS file paths.
    :rtype: dict
    """
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
    ):
        if os.path.exists(outputvis) and not overwrite:
            log_print('INFO', f"Output MS file {outputvis} already exists. Skipping processing.")
            return outputvis

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

        msfile_copy = os.path.join(workdir, f'{msname}.ms')
        if not os.path.exists(msfile_copy):
            if msfile.lower().endswith('.ms'):
                shutil.copytree(msfile, msfile_copy)
            elif msfile.lower().endswith('.ms.tar.gz'):
                subprocess.run(['tar', '-zxf', msfile, '-C', workdir], check=True)
            else:
                raise ValueError(f"Unsupported file format: {msfile}")
        msfile = msfile_copy
        msfile, msname = _prepare_single_pol_ms(msfile, workdir, msname, pols, overwrite=overwrite)
        if str(pols).upper().replace(',', '') == 'YY_AS_XX':
            log_print('INFO', "Using YY data relabeled as XX for downstream diagnostic imaging")
            pols = 'XX'

        viz_timerange = ant_trange(msfile)
        (tstart, tend) = viz_timerange.split('~')
        tbg_msfile = Time(qa.quantity(tstart, 's')['value'], format='mjd').to_datetime()
        ted_msfile = Time(qa.quantity(tend, 's')['value'], format='mjd').to_datetime()
        tmid_msfile = tbg_msfile + (ted_msfile - tbg_msfile) / 2

        antenna = '0~12'
        if tbg_msfile >= EOVSA15_UPGRADE_DATE.to_datetime():
            antenna = '0~14'
        ## date_str is set to the day 1 if the time is between 08:00 UT on day 1 to 08:00 UT(+1) on day 2
        if tbg_msfile.time() < time(8, 0):
            date_local = tbg_msfile.date() - timedelta(days=1)
        else:
            date_local = tbg_msfile.date()
        date_str = date_local.strftime('%Y%m%d')
        reftime_daily = Time(datetime.combine(date_local, time(20, 0)))
        freq_setup = FrequencySetup(Time(tbg_msfile))
        spws_indices = freq_setup.spws_indices
        tb_models = {}
        outfits_all = {}
        bright = {}
        bright_thresh = {}
        tb_ratio_thresh = {}
        segmented_imaging = {}
        briggs = {}
        fits_mask = {}
        imaging_objs = {}
        spws = freq_setup.spws
        defaultfreq = freq_setup.defaultfreq
        freq = defaultfreq
        nbands = freq_setup.nbands

        for sidx, sp_index in enumerate(spws_indices):
            tb_models[sidx] = None
            outfits_all[sidx] = None
            bright[sidx] = False
            imaging_objs[sidx] = None
            bright_thresh[sidx] = bright_thresh_[sidx]
            tb_ratio_thresh[sidx] = tb_ratio_thresh_[sidx]
            # segmented_imaging[sidx] = True if sidx in [0, 1, 2, 3] else False
            ## testing with segmented imaging for all bands for now with the npz calibration, will revert to the above after validation
            segmented_imaging[sidx] = True if sidx in [0, 1, 2, 3, 4, 5, 6] else False
            briggs[sidx] = briggs_[sidx]

        dsize, fdens = calc_diskmodel(tmid_msfile, nbands, freq)
        fdens = fdens / 2.0  # convert Stokes I to single-linear-pol flux density

        uvmin_l_str = _compute_uv_ranges(spws_indices, spws, freq, spwidx2proc)

        run_start_time = datetime.now()

        # --- Pre-processing: flagging, hanning smoothing ---
        if os.path.isdir(msfile + '.flagversions'):
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

        if hanning:
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

                    reffreq, cdelt4_real, bmsize = freq_setup.get_reffreq_and_cdelt(spws[sidx], return_bmsize=True)
                    log_print('INFO', f"Processing SPW {spws[sidx]} for msfile {msname} ...")

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
                        clean_obj.run(dryrun=False)
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
                              f"Ratio = {bright_ratio * snr:.1f}, Thresh = {bright_thresh[sidx]}, SNR = {snr:.1f}")

                    if sidx == 0 and snr >= 5:
                        bright[sidx] = True
                        log_print('INFO', f"SPW {spws[sidx]}: SNR is high enough ({snr:.1f}). Setting bright to True.")
                    if sidx == 1 and snr >= 30:
                        bright[sidx] = True
                        log_print('INFO', f"SPW {spws[sidx]}: SNR is high enough ({snr:.1f}). Setting bright to True.")
                    ## the bright threshold is commented out for npz calibration testing, will reinstate after validation
                    # if bright[sidx]:
                    #     log_print('INFO', f"SPW {spws[sidx]} is bright. Proceeding with segmented imaging.")
                    #     segmented_imaging[sidx] = True
                    # if bright_ratio < tb_ratio_thresh[sidx]:
                    #     segmented_imaging[sidx] = False

                    # --- Compute time intervals for all rounds ---
                    round_intervals = {}
                    for round_def in SELFCAL_ROUNDS:
                        mult = round_def['interval_multipliers'].get(sidx, 1)
                        round_intervals[round_def['name']] = _compute_round_intervals(wsclean_intervals, mult)
                    mult = FINAL_IMAGING_CONFIG['interval_multipliers'].get(sidx, 1)
                    round_intervals['final'] = _compute_round_intervals(wsclean_intervals, mult)

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

                    # --- Step 4: Final imaging ---
                    synfitsfiles, imaging_objs = _run_final_imaging(
                        msfile, sidx, spws[sidx], spwstr, sp_index, workdir, imgoutdir,
                        msname, ri_final, briggs[sidx], bmsize, pols,
                        reftime_daily, viz_timerange, date_str,
                        segmented_imaging[sidx], imaging_objs, freq_setup,
                        tr_series_time=tr_series_time, fits_tag=fits_tag)

                    if not segmented_imaging[sidx] and len(synfitsfiles) > 0:
                        outfits_all[sidx] = synfitsfiles[0]

                    elapsed_total = (datetime.now() - run_start_time).total_seconds() / 60
                    log_print('INFO', f"Pipeline for SPW {spws[sidx]}: completed in {elapsed_total:.1f} minutes")

        # --- Post-processing: merge FITS, add disk, move caltables ---
        post_tag = f'.{fits_tag}' if fits_tag else ''
        for sidx, spw in enumerate(spws):
            spwstr = format_spw(spw)
            sp_st, sp_ed = spws[sidx].split('~')
            dszs = [float(dsize[int(sp)].rstrip('arcsec')) for sp in range(int(sp_st), int(sp_ed) + 1)]
            fdns = [fdens[int(sp)] for sp in range(int(sp_st), int(sp_ed) + 1)]
            reffreq, cdelt4_real, bmsize = freq_setup.get_reffreq_and_cdelt(spws[sidx], return_bmsize=True)
            outfits = os.path.join(imgoutdir,
                                   f'eovsa.synoptic_daily{post_tag}.{date_str}T200000Z.s{spwstr}.tb.fits')
            outfits_disk = outfits.replace('.tb.fits', '.tb.disk.fits')
            if segmented_imaging.get(sidx, False):
                synfitsfiles = sorted(glob(os.path.join(imgoutdir,
                                                        f"eovsa.synoptic{post_tag}.{date_str[:-1]}?T??????Z.s{spwstr}.tb.fits")))
                snr_threshold = 3
                if len(synfitsfiles) > 0:
                    log_print('INFO', f"Merging synoptic images for SPW {spwstr} to {outfits} ...")
                    merge_FITSfiles(synfitsfiles, outfits, overwrite=True, snr_threshold=snr_threshold)
                    outfits_all[sidx] = outfits
                    synfitsfiles_disk = [l.replace('.tb.fits', '.tb.disk.fits') for l in synfitsfiles]
                    try:
                        add_convolved_disk_to_fits(synfitsfiles + [outfits], synfitsfiles_disk + [outfits_disk],
                                                   dszs, fdns, ignore_data=False, toTb=True,
                                                   rfreq=reffreq, bmaj=bmsize / 3600.)
                    except Exception:
                        log_print('ERROR', f"[postprocess:add_disk] failed | "
                                           f"{_format_log_state(spw=spw, spwstr=spwstr, n_synfits=len(synfitsfiles))}\n"
                                           f"{traceback.format_exc()}")
                else:
                    log_print('WARNING', f"No synoptic images found for SPW {spwstr}. Skipping merge_FITSfiles.")
            else:
                if os.path.exists(outfits):
                    log_print('INFO', f"Add disk to synoptic image {outfits} ...")
                    outfits_all[sidx] = outfits
                    add_convolved_disk_to_fits([outfits], [outfits_disk], dszs, fdns,
                                               ignore_data=False, toTb=True,
                                               rfreq=reffreq, bmaj=bmsize / 3600.)
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
                if os.path.exists(targetfile):
                    shutil.rmtree(targetfile, ignore_errors=True)
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
        if outputvis and not os.path.exists(outputvis):
            log_print('INFO', f"Building selfcal'd outputvis {outputvis} from per-spw splits ...")
            os.makedirs(os.path.dirname(outputvis) or '.', exist_ok=True)
            slfcaled_spw_ms_list = []
            for sidx, sp_index in enumerate(spws_indices):
                spwstr = format_spw(spws[sidx])
                msfile_sp = os.path.join(workdir, f'{msname}.sp{spwstr}.slfcaled.ms')
                if os.path.exists(msfile_sp):
                    shutil.rmtree(msfile_sp, ignore_errors=True)
                log_print('INFO', f"Splitting SPW {spws[sidx]} (CORRECTED_DATA) -> {msfile_sp}")
                split(vis=msfile, outputvis=msfile_sp, spw=spws[sidx], datacolumn='corrected')
                if os.path.isdir(msfile_sp):
                    slfcaled_spw_ms_list.append(msfile_sp)
                else:
                    log_print('WARNING', f"Split failed for SPW {spws[sidx]}; skipping in outputvis.")
            if slfcaled_spw_ms_list:
                log_print('INFO',
                          f"Concatenating {len(slfcaled_spw_ms_list)} per-spw MS files into {outputvis} ...")
                concat(vis=slfcaled_spw_ms_list, concatvis=outputvis, freqtol='', dirtol='')
                log_print('INFO', f"Selfcal'd outputvis saved to {outputvis}")
                # Tar the outputvis and remove the MS directory
                outputvis_tar = outputvis + '.tar.gz'
                log_print('INFO', f"Archiving {outputvis} to {outputvis_tar} ...")
                with tarfile.open(outputvis_tar, 'w:gz') as tar:
                    tar.add(outputvis, arcname=os.path.basename(outputvis))
                if os.path.exists(outputvis_tar) and os.path.getsize(outputvis_tar) > 0:
                    shutil.rmtree(outputvis)
                    log_print('INFO', f"Removed {outputvis} after successful archiving to {outputvis_tar}")
                else:
                    log_print('WARNING', f"Tar archive {outputvis_tar} appears empty or missing. Keeping {outputvis}.")
            else:
                log_print('WARNING', "No per-spw slfcaled MS files produced; outputvis not created.")

        # --- Tar the source UDB*.ms working copy and remove it ---
        if os.path.isdir(msfile):
            tar_path = msfile + '.tar.gz'
            log_print('INFO', f"Archiving {msfile} to {tar_path} ...")
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(msfile, arcname=os.path.basename(msfile))
            if os.path.exists(tar_path) and os.path.getsize(tar_path) > 0:
                shutil.rmtree(msfile)
                log_print('INFO', f"Removed {msfile} after successful archiving to {tar_path}")
            else:
                log_print('WARNING', f"Tar archive {tar_path} appears empty or missing. Keeping {msfile}.")

        log_step('INFO', 'pipeline_run', 'completed', processed_spws=sorted(outfits_all.keys()))
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
    parser.add_argument('--fits_tag', type=str, default='',
                        help='Optional tag inserted into synoptic FITS filenames.')
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
        hanning=args.hanning,
        do_sbdcal=args.do_sbdcal,
    )
