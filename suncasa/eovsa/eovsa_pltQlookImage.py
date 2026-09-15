import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sunpy import map as smap
import sunpy
from suncasa.utils import plot_mapX as pmX
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colorbar as colorbar
import matplotlib.patches as patches
from datetime import timedelta
from datetime import datetime
from eovsapy.spw_config import SPWS_34BAND, SPWS_52BAND, SPWS_52BAND_ALT, SPW_EPOCH_SPLIT_DATE
from glob import glob
import numpy as np
from astropy.time import Time
import urllib.request
import urllib.error
import socket

socket.setdefaulttimeout(180)
HELIOVIEWER_TIMEOUT_S = 5
HELIOVIEWER_TIMEOUT_LIMIT = 3
QUERY_TIMEOUT_S = 5

imgfitsdir = '/data1/eovsa/fits/synoptic/'
imgfitstmpdir = os.path.join(os.environ.get('EOVSA_WORKDIR', '/data1/workdir'), 'fitstmp')
pltfigdir = '/common/webplots/SynopticImg/eovsamedia/eovsa-browser/'

PRODUCT_VERSIONS = (
    'v1.0',
    'v2.0',
    'v2.0_alt',
    'v2.1',
    'v2.1_alt',
    'legacy_v2.0',
)


def normalize_product_version(version):
    """Validate and return the canonical product-version tag.

    :param version: Product-version selector.
    :type version: str
    :returns: The unchanged canonical selector.
    :rtype: str
    :raises ValueError: If ``version`` is a retired or unknown selector.
    """
    if version not in PRODUCT_VERSIONS:
        raise ValueError(
            'Product version {0} is not supported. Valid versions are {1}.'.format(
                version, ', '.join(PRODUCT_VERSIONS)
            )
        )
    return version


def synoptic_product_path(dateobj, filename, version=None):
    datestrdir = dateobj.strftime("%Y/%m/%d")
    candidates = []
    if version:
        canonical = normalize_product_version(version)
        candidates.append(os.path.join(imgfitsdir, datestrdir, canonical, filename))
    candidates.append(os.path.join(imgfitsdir, datestrdir, filename))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def synoptic_preview_dir(dateobj, version=None, create=False):
    datestrdir = dateobj.strftime("%Y/%m/%d")
    outdir = os.path.join(pltfigdir, datestrdir)
    if version:
        outdir = os.path.join(outdir, normalize_product_version(version))
    if create:
        os.makedirs(outdir, exist_ok=True)
    return outdir


def synoptic_pipeline_status_path(dateobj, version='v2.0'):
    """Return the per-date synoptic pipeline status path.

    :param dateobj: Pipeline date used to construct the status filename.
    :type dateobj: datetime.datetime
    :param version: Synoptic product version.
    :type version: str
    :returns: Absolute status JSON path.
    :rtype: str
    """
    version = normalize_product_version(version)
    datestr = dateobj.strftime('%Y%m%d')
    datestrdir = dateobj.strftime('%Y/%m/%d')
    canonical_path = os.path.join(
        imgfitsdir,
        datestrdir,
        'eovsa.synoptic_pipeline_status.{}.{}.json'.format(datestr, version),
    )
    return canonical_path


def eovsa_preview_review_label(dateobj, version='v2.0'):
    """Return the visible review tag for a date/version, if needed.

    :param dateobj: Pipeline date whose status metadata should be read.
    :type dateobj: datetime.datetime
    :param version: Synoptic product version.
    :type version: str
    :returns: Review label, or an empty string when no review is requested.
    :rtype: str
    """
    statusfile = synoptic_pipeline_status_path(dateobj, version=version)
    if not os.path.exists(statusfile):
        return ''
    try:
        with open(statusfile, 'r') as infile:
            status = json.load(infile)
    except (OSError, TypeError, ValueError):
        return ''
    if not isinstance(status, dict):
        return ''

    pipeline_state = str(status.get('state') or '').strip()
    qa_state = str(status.get('s00_qa_state') or '').strip()
    qa_reason = str(status.get('s00_qa_reason') or '').strip()
    qa_reason_code = qa_reason.split(':', 1)[0].strip()
    requires_review = status.get('s00_qa_requires_review') is True
    requires_review = requires_review or status.get('s00_qa_warning') is True
    requires_review = requires_review or pipeline_state.lower() in (
        'imaging_review_required',
        'review_required',
    )
    requires_review = requires_review or qa_state.upper().startswith(
        ('FAIL_', 'REVIEW_')
    )
    requires_review = requires_review or qa_reason_code.upper().startswith(
        ('FAIL_', 'REVIEW_')
    )
    if not requires_review:
        return ''

    review_code = qa_state or qa_reason_code or pipeline_state or 'REQUIRED'
    return 'REVIEW: {}'.format(review_code.upper().replace(' ', '_'))


def fits_tag_infix(fits_tag):
    if not fits_tag:
        return ''
    return '.{}'.format(str(fits_tag).lstrip('.'))


def eovsa_preview_filename(size, band_number, fits_tag=''):
    # fits_tag selects the input FITS product; browser previews are canonical per version folder.
    return '{}_eovsa_bd{:02d}.jpg'.format(size, band_number)


def eovsa_preview_label(eomap):
    label = 'EOVSA {:.1f} GHz  {}'.format(
        eomap.meta['CRVAL3'] / 1e9,
        eomap.date.strftime('%d-%b-%Y 20:00 UT'))
    nant_img = eomap.meta.get('NANTIMG')
    nant_total = eomap.meta.get('NANTTOT')
    if nant_img is None or nant_total is None:
        return label
    try:
        return '{}   Ants used: {}/{}'.format(label, int(nant_img), int(nant_total))
    except (TypeError, ValueError):
        return label


def eovsa_preview_warning_label(eomap):
    cal_date = eomap.meta.get('CALDATE')
    cal_mode = str(eomap.meta.get('CALMODE', '')).strip().upper()
    cal_warn = eomap.meta.get('CALWARN')
    if isinstance(cal_warn, str):
        cal_warn = cal_warn.strip().upper() in ('T', 'TRUE', '1', 'YES')
    if not cal_date or not (cal_warn or cal_mode == 'FALLBACK'):
        return ''
    return 'Provisional cal: {}'.format(str(cal_date).strip())


def synoptic_daily_product_filename(dateobj, spwstr, version='v2.0', fits_tag=''):
    datestr = dateobj.strftime('%Y%m%d')
    tag = fits_tag_infix(fits_tag)
    version = normalize_product_version(version)
    if version == 'v1.0':
        return 'eovsa_{}.spw{}.tb.disk.fits'.format(datestr, spwstr)
    if version == 'legacy_v2.0':
        return f'eovsa.synoptic_daily{tag}.{datestr}T200000_UTC.s{spwstr}.tb.fits'
    return f'eovsa.synoptic_daily{tag}.{datestr}T200000Z.s{spwstr}.tb.disk.fits'


def eovsa_fineband_preview_filename(size, spwstr, fits_tag=''):
    # Parallel to eovsa_preview_filename's '..._eovsa_bdNN.jpg' scheme, but keyed
    # by the literal spw tag string (e.g. '05-06') instead of a bd index, so it
    # can never collide with the fixed bd01..bd07 names used by the standard
    # 7-band products.
    return '{}_eovsa_s{}.jpg'.format(size, spwstr)


# Standard 7-band anchor frequencies (GHz), used by interp_band_scale() as a
# fallback when the actual per-date standard-band FITS products are not
# available to read CRVAL3 from. Derived from the SPWS_52BAND grouping
# ['0~1', '2~4', '5~10', '11~20', '21~30', '31~43', '44~49'] for a
# representative post-SPW_EPOCH_SPLIT_DATE day.
STANDARD_BAND_ANCHOR_GHZ_FALLBACK = [1.4183, 2.8736, 4.3317, 6.9297, 10.1797, 13.9172, 17.0047]

# Standard-band spw tags (zero-padded, dash-joined) matching SPWS_52BAND, in
# the same order as STANDARD_BAND_ANCHOR_GHZ_FALLBACK / the vmaxs/vmins lists
# in main(). Used to (a) look up per-date anchor frequencies from the actual
# standard-band FITS products, and (b) identify which spw tags found on disk
# are "standard" (already plotted by the unchanged bd01..bd07 loop) versus
# "extra"/fine-spectral (to be discovered and plotted separately).
STANDARD_BAND_SPW_TAGS = ['00-01', '02-04', '05-10', '11-20', '21-30', '31-43', '44-49']


def _read_crval3_ghz(fits_path):
    """Read CRVAL3 (Hz) from whichever HDU carries the WCS header and return it in GHz.

    Handles both single-HDU FITS and the 2-HDU compressed FITS layout (where
    the WCS keywords live on an image/table extension rather than the
    primary HDU), by scanning HDUs in order and using the first one that
    carries CRVAL3.

    :param fits_path: Path to a FITS file.
    :type fits_path: str
    :return: Reference frequency in GHz.
    :rtype: float
    :raises KeyError: If no HDU in the file carries a CRVAL3 keyword.
    """
    from astropy.io import fits as _fits
    with _fits.open(fits_path) as hdul:
        for hdu in hdul:
            if 'CRVAL3' in hdu.header:
                return hdu.header['CRVAL3'] / 1e9
    raise KeyError('CRVAL3 not found in any HDU of {}'.format(fits_path))


def _standard_band_anchor_ghz(dateobj=None, version='v2.0', fits_tag=''):
    """Return the 7 standard-band anchor frequencies (GHz) for interp_band_scale().

    Attempts to read CRVAL3 from the actual standard-band (SPWS_52BAND)
    ``.tb.disk.fits`` products for the given date, in the same order as
    STANDARD_BAND_SPW_TAGS / the vmaxs and vmins lists in main(). If
    ``dateobj`` is None, or any product is missing/unreadable, falls back to
    STANDARD_BAND_ANCHOR_GHZ_FALLBACK.

    :param dateobj: Date to look up standard-band products for.
    :type dateobj: datetime, optional
    :param version: Product version subdirectory to search.
    :type version: str, optional
    :param fits_tag: Optional fits_tag used to build the standard-band filenames.
    :type fits_tag: str, optional
    :return: 7 anchor frequencies in GHz.
    :rtype: list[float]
    """
    if dateobj is None:
        return list(STANDARD_BAND_ANCHOR_GHZ_FALLBACK)
    anchors = []
    for spwstr in STANDARD_BAND_SPW_TAGS:
        eofile = synoptic_product_path(
            dateobj,
            synoptic_daily_product_filename(dateobj, spwstr, version=version, fits_tag=fits_tag),
            version=version)
        if not os.path.exists(eofile):
            return list(STANDARD_BAND_ANCHOR_GHZ_FALLBACK)
        try:
            anchors.append(_read_crval3_ghz(eofile))
        except Exception:
            return list(STANDARD_BAND_ANCHOR_GHZ_FALLBACK)
    return anchors


def interp_band_scale(freq_ghz, anchor_ghz=None, vmaxs=None, vmins=None):
    """Interpolate production vmin/vmax color scaling for an arbitrary frequency.

    Performs log10(GHz) vs. log10(|value|) interpolation of the 7 standard
    (SPWS_52BAND) production vmax/vmin anchor values, so that non-standard
    band groupings (fine-spectral chunks, custom merges, etc.) get a
    reasonable, continuous color scale consistent with the production
    7-band quicklook images. Values outside the anchor frequency range are
    clamped to the nearest anchor's scale (no extrapolation). At an exact
    anchor frequency, the corresponding anchor vmin/vmax is returned exactly.

    :param freq_ghz: Frequency, in GHz, to interpolate the scale for.
    :type freq_ghz: float
    :param anchor_ghz: 7 anchor frequencies (GHz), in the same order as
        vmaxs/vmins. Defaults to STANDARD_BAND_ANCHOR_GHZ_FALLBACK.
    :type anchor_ghz: list[float], optional
    :param vmaxs: 7 standard-band production vmax anchors. Defaults to the
        vmaxs used in main() for the standard 7 bands.
    :type vmaxs: list[float], optional
    :param vmins: 7 standard-band production vmin anchors (negative).
        Defaults to the vmins used in main() for the standard 7 bands.
    :type vmins: list[float], optional
    :return: (vmin, vmax) interpolated (or clamped) for freq_ghz.
    :rtype: tuple[float, float]
    """
    if anchor_ghz is None:
        anchor_ghz = STANDARD_BAND_ANCHOR_GHZ_FALLBACK
    if vmaxs is None:
        vmaxs = [70.0e4, 30e4, 18e4, 13e4, 8e4, 6e4, 6e4]
    if vmins is None:
        vmins = [-18.0e3, -8e3, -4.8e3, -3.4e3, -2.1e3, -1.6e3, -1.6e3]

    anchor_ghz = np.asarray(anchor_ghz, dtype=float)
    vmaxs = np.asarray(vmaxs, dtype=float)
    vmins = np.asarray(vmins, dtype=float)

    order = np.argsort(anchor_ghz)
    a_ghz = anchor_ghz[order]
    a_vmax = vmaxs[order]
    a_vmin = vmins[order]

    # Exact hit (or clamp target) on an anchor frequency: return that anchor's
    # vmin/vmax verbatim rather than round-tripping through log10/10**, which
    # can introduce float64 noise at the ~1e-11 relative level and would
    # otherwise violate the "exact at anchors" contract.
    clamped_ghz = min(max(freq_ghz, a_ghz.min()), a_ghz.max())
    exact_idx = np.flatnonzero(a_ghz == clamped_ghz)
    if exact_idx.size:
        i = exact_idx[0]
        return float(a_vmin[i]), float(a_vmax[i])

    log_a = np.log10(a_ghz)
    log_vmax = np.log10(a_vmax)
    log_vmin_mag = np.log10(np.abs(a_vmin))

    lg_c = np.log10(clamped_ghz)

    vmax_i = float(10 ** np.interp(lg_c, log_a, log_vmax))
    vmin_i = float(-(10 ** np.interp(lg_c, log_a, log_vmin_mag)))
    return vmin_i, vmax_i


def clearImage():
    for (dirpath, dirnames, filenames) in os.walk(pltfigdir):
        for filename in filenames:
            for k in ['0094', '0193', '0335', '4500', '0171', '0304', '0131', '1700', '0211', '1600', '_HMIcont',
                      '_HMImag']:
                # for k in ['_Halph_fr']:
                if k in filename:
                    print(os.path.join(dirpath, filename))
                    os.system('rm -rf ' + os.path.join(dirpath, filename))


def pltEmptyImage2(dpis_dict={'t': 32.0}):
    imgoutdir = './nodata/'

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    rect_bkg = patches.Rectangle((-1227, -1227), 1227 * 2, 1227 * 2, linewidth=0, edgecolor='none', facecolor='k',
                                 alpha=0.9)
    rect_bar = patches.Rectangle((-1227, -300), 1227 * 2, 300 * 2, linewidth=0, edgecolor='none', facecolor='w',
                                 alpha=0.5)

    ax.add_patch(rect_bkg)
    ax.add_patch(rect_bar)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.text(0.5, 0.5, 'No Data',
            transform=ax.transAxes, color='w', ha='center', va='center', fontsize=120)
    ax.set_xlim(-1227, 1227)
    ax.set_ylim(-1227, 1227)

    for l, dpi in dpis_dict.items():
        figname = 'nodata.jpg'
        fig.savefig(figname, dpi=int(dpi), pil_kwargs={"quality":85})
    return


def pltEmptyImage(datestr, spws, vmaxs, vmins, dpis_dict={'t': 32.0}):
    plt.ioff()
    dateobj = datetime.strptime(datestr, "%Y-%m-%d")
    imgoutdir = './nodata/'

    cmap = plt.get_cmap('sdoaia304')

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    rect = patches.Rectangle((-1227, -300), 1227 * 2, 300 * 2, linewidth=0, edgecolor='none', facecolor='k', alpha=0.5)

    for s, sp in enumerate(spws):
        ax.cla()
        spwstr = '-'.join(['{:02d}'.format(int(sp_)) for sp_ in sp.split('~')])
        eofile = synoptic_product_path(
            dateobj,
            synoptic_daily_product_filename(dateobj, spwstr, version='v1.0'),
            version='v1.0')
        if not os.path.exists(eofile): continue
        if not os.path.exists(imgoutdir): os.makedirs(imgoutdir)
        eomap = smap.Map(eofile)
        norm = colors.Normalize(vmin=vmins[s], vmax=vmaxs[s])
        eomap_ = pmX.Sunmap(eomap)
        eomap_.imshow(axes=ax, cmap=cmap, norm=norm, alpha=0.75)
        eomap_.draw_limb(axes=ax, lw=0.5, alpha=0.5)
        eomap_.draw_grid(axes=ax, grid_spacing=10. * u.deg, lw=0.5)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5, 0.5, 'No Data',
                transform=ax.transAxes, color='w', ha='center', va='center', fontsize=120)
        ax.add_patch(rect)
        ax.set_xlim(-1227, 1227)
        ax.set_ylim(-1227, 1227)

        for l, dpi in dpis_dict.items():
            figname = os.path.join(imgoutdir, '{}_eovsa_bd{:02d}.jpg'.format(l, s + 1))
            fig.savefig(figname, dpi=int(dpi), pil_kwargs={"quality":85})
    return


def _render_eovsa_band_frame(eofile, ax, cmap, vmin, vmax, dpis_dict, imgoutdir, filename_fn, fig,
                             review_label=''):
    """Render one EOVSA band FITS product into the shared preview axes and save it.

    Extracted, behavior-preserving, from the per-band body of
    ``pltEovsaQlookImage_v3`` so both the standard 7-band loop and the
    discovery-mode (non-standard band) loop share identical rendering code
    (cmap, AsinhStretch, ImageNormalize, limb/grid overlays, labels, and
    save/print behavior).

    :param eofile: Path to the ``.tb.disk.fits`` product to render.
    :type eofile: str
    :param ax: Matplotlib axes to draw into (cleared by the caller beforehand).
    :type ax: matplotlib.axes.Axes
    :param cmap: Colormap to use (sdoaia304 with bad-color set by the caller).
    :type cmap: matplotlib.colors.Colormap
    :param vmin: Lower bound for ImageNormalize.
    :type vmin: float
    :param vmax: Upper bound for ImageNormalize.
    :type vmax: float
    :param dpis_dict: Mapping of size-label ('t'/'l'/'f') to DPI value.
    :type dpis_dict: dict
    :param imgoutdir: Output directory for the rendered JPEGs.
    :type imgoutdir: str
    :param filename_fn: Callable(size_label) -> output filename for that size.
    :type filename_fn: callable
    :param fig: Figure owning ``ax``, used to save each size variant.
    :type fig: matplotlib.figure.Figure
    :return: None
    :rtype: None
    """
    from astropy.visualization.stretch import AsinhStretch
    from astropy.visualization import ImageNormalize
    eomap = smap.Map(eofile)
    stretch = AsinhStretch(a=0.15)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch)
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    eomap_ = pmX.Sunmap(eomap)
    eomap_.imshow(axes=ax, cmap=cmap, norm=norm)
    eomap_.draw_limb(axes=ax, lw=0.5, alpha=0.5)
    eomap_.draw_grid(axes=ax, grid_spacing=10. * u.deg, lw=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    warning_label = eovsa_preview_warning_label(eomap)
    if review_label:
        ax.text(0.02, 0.98, review_label,
                transform=ax.transAxes, color='#ff6b6b', ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=2.0))
    if warning_label:
        ax.text(0.02, 0.88 if review_label else 0.98, warning_label,
                transform=ax.transAxes, color='#ffd166', ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', pad=2.0))
    ax.text(0.02, 0.02,
            eovsa_preview_label(eomap),
            transform=ax.transAxes, color='w', ha='left', va='bottom', fontsize=9)
    ax.text(0.98, 0.02, 'Max Tb {:.0f} K'.format(np.nanmax(eomap.data)),
            transform=ax.transAxes, color='w', ha='right', va='bottom', fontsize=9)
    ax.set_xlim(-1227, 1227)
    ax.set_ylim(-1227, 1227)

    print(f'Processing EOVSA images {eofile}')
    for l, dpi in dpis_dict.items():
        figname = os.path.join(imgoutdir, filename_fn(l))
        fig.savefig(figname, dpi=int(dpi), pil_kwargs={"quality": 85})
        print('EOVSA image saved to {}'.format(figname))


def _discover_fineband_spw_tags(dateobj, version, fits_tag, standard_spwstrs):
    """Glob the date/version product dir for extra (non-standard) daily band tags.

    Looks for ``eovsa.synoptic_daily{tag}.{date}T200000Z.s*.tb.disk.fits``
    products in the same product directory used for the standard bands, and
    returns the spw tag strings (e.g. '05-06', '31-33') for any files whose
    tag is not already one of ``standard_spwstrs``. This is purely additive:
    on dates/versions where only the standard products exist, this returns
    an empty list and the discovery-mode loop has no effect whatsoever.

    :param dateobj: Date being processed.
    :type dateobj: datetime
    :param version: Product version subdirectory to search.
    :type version: str
    :param fits_tag: Optional fits_tag used to build the daily filenames.
    :type fits_tag: str
    :param standard_spwstrs: The zero-padded, dash-joined spw tags already
        handled by the standard-band loop (to be excluded).
    :type standard_spwstrs: set[str]
    :return: Sorted list of extra spw tag strings found on disk.
    :rtype: list[str]
    """
    import re
    datestr = dateobj.strftime('%Y%m%d')
    tag = fits_tag_infix(fits_tag)
    datestrdir = dateobj.strftime("%Y/%m/%d")
    candidates_dirs = []
    if version:
        canonical = normalize_product_version(version)
        candidates_dirs.append(os.path.join(imgfitsdir, datestrdir, canonical))
    candidates_dirs.append(os.path.join(imgfitsdir, datestrdir))

    pattern = f'eovsa.synoptic_daily{tag}.{datestr}T200000Z.s*.tb.disk.fits'
    tag_re = re.compile(r'\.s(\d{2}-\d{2})\.tb\.disk\.fits$')

    found = {}
    for d in candidates_dirs:
        for fpath in glob(os.path.join(d, pattern)):
            m = tag_re.search(os.path.basename(fpath))
            if not m:
                continue
            spwstr = m.group(1)
            if spwstr in standard_spwstrs:
                continue
            found.setdefault(spwstr, fpath)
    return sorted(found.keys())


def pltEovsaQlookImage_v3(datestr, spws, vmaxs, vmins, dpis_dict, fig=None, ax=None, overwrite=False, verbose=False,
                           version='v2.0', fits_tag='', include_fine_bands=True):
    from astropy.visualization.stretch import AsinhStretch
    from astropy.visualization import ImageNormalize
    plt.ioff()
    dateobj = datetime.strptime(datestr, "%Y-%m-%d")
    imgoutdir = synoptic_preview_dir(dateobj, version=version)
    review_label = eovsa_preview_review_label(dateobj, version=version)

    cmap = plt.get_cmap('sdoaia304')
    cmap.set_bad(color='k')

    if fig is None or ax is None:
        mkfig = True
    else:
        mkfig = False

    if mkfig:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    if verbose: print('Processing EOVSA images for date {}'.format(dateobj.strftime('%Y-%m-%d')))
    standard_spwstrs = set()
    for s, sp in enumerate(spws):
        spwstr = '-'.join(['{:02d}'.format(int(sp_)) for sp_ in sp.split('~')])
        standard_spwstrs.add(spwstr)
        fexists = []
        for l, dpi in dpis_dict.items():
            figname = os.path.join(imgoutdir, eovsa_preview_filename(l, s + 1, fits_tag=fits_tag))
            fexists.append(os.path.exists(figname))

        if overwrite or review_label or (False in fexists):
            ax.cla()
            eofile = synoptic_product_path(
                dateobj,
                synoptic_daily_product_filename(dateobj, spwstr, version=version, fits_tag=fits_tag),
                version=version)
            if not os.path.exists(eofile):
                print('Fail to plot {} as it does not exist'.format(eofile))
                continue
            synoptic_preview_dir(dateobj, version=version, create=True)
            try:
                _render_eovsa_band_frame(
                    eofile, ax, cmap, vmins[s], vmaxs[s], dpis_dict, imgoutdir,
                    lambda l, s=s: eovsa_preview_filename(l, s + 1, fits_tag=fits_tag), fig,
                    review_label=review_label)
            except Exception as err:
                print('Fail to plot {}'.format(eofile))
                print(err)

    # --- Discovery mode: additional (non-standard) band-segmentation products ---
    # This is purely additive and runs after the standard-band loop above, whose
    # code path and output filenames are completely unchanged. On dates/versions
    # where only the standard 7 bands exist, _discover_fineband_spw_tags() finds
    # nothing and this block has no effect (no extra files, no extra output).
    if include_fine_bands:
        try:
            extra_spwstrs = _discover_fineband_spw_tags(dateobj, version, fits_tag, standard_spwstrs)
        except Exception as err:
            print('Fail to discover non-standard band products for {}'.format(datestr))
            print(err)
            extra_spwstrs = []

        if extra_spwstrs:
            anchor_ghz = _standard_band_anchor_ghz(dateobj, version=version, fits_tag=fits_tag)

        for spwstr in extra_spwstrs:
            fexists = []
            for l, dpi in dpis_dict.items():
                figname = os.path.join(imgoutdir, eovsa_fineband_preview_filename(l, spwstr, fits_tag=fits_tag))
                fexists.append(os.path.exists(figname))

            if not (overwrite or review_label or (False in fexists)):
                continue

            ax.cla()
            eofile = synoptic_product_path(
                dateobj,
                synoptic_daily_product_filename(dateobj, spwstr, version=version, fits_tag=fits_tag),
                version=version)
            if not os.path.exists(eofile):
                print('Fail to plot {} as it does not exist'.format(eofile))
                continue
            synoptic_preview_dir(dateobj, version=version, create=True)
            try:
                freq_ghz = _read_crval3_ghz(eofile)
                vmin_i, vmax_i = interp_band_scale(freq_ghz, anchor_ghz=anchor_ghz)
                _render_eovsa_band_frame(
                    eofile, ax, cmap, vmin_i, vmax_i, dpis_dict, imgoutdir,
                    lambda l, spwstr=spwstr: eovsa_fineband_preview_filename(l, spwstr, fits_tag=fits_tag), fig,
                    review_label=review_label)
            except Exception as err:
                print('Fail to plot {}'.format(eofile))
                print(err)

    if mkfig:
        pass
    else:
        plt.close(fig)
    return


def pltEovsaQlookImage(datestr, spws, vmaxs, vmins, dpis_dict, fig=None, ax=None, overwrite=False, verbose=False,
                        version='v1.0', fits_tag=''):
    from astropy.visualization.stretch import AsinhStretch
    from astropy.visualization import ImageNormalize
    plt.ioff()
    dateobj = datetime.strptime(datestr, "%Y-%m-%d")
    imgoutdir = synoptic_preview_dir(dateobj, version=version)
    review_label = eovsa_preview_review_label(dateobj, version=version)

    cmap = plt.get_cmap('sdoaia304')
    cmap.set_bad(color='k')

    if fig is None or ax is None:
        mkfig = True
    else:
        mkfig = False

    if mkfig:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    if verbose: print('Processing EOVSA images for date {}'.format(dateobj.strftime('%Y-%m-%d')))
    for s, sp in enumerate(spws):
        fexists = []
        for l, dpi in dpis_dict.items():
            figname = os.path.join(imgoutdir, eovsa_preview_filename(l, s + 1, fits_tag=fits_tag))
            fexists.append(os.path.exists(figname))

        if overwrite or review_label or (False in fexists):
            ax.cla()
            spwstr = '-'.join(['{:02d}'.format(int(sp_)) for sp_ in sp.split('~')])
            eofile = synoptic_product_path(
                dateobj,
                synoptic_daily_product_filename(dateobj, spwstr, version=version, fits_tag=fits_tag),
                version=version)
            if not os.path.exists(eofile):
                print('Fail to plot {} as it does not exist'.format(eofile))
                continue
            synoptic_preview_dir(dateobj, version=version, create=True)
            try:
                eomap = smap.Map(eofile)
                stretch = AsinhStretch(a=0.15)
                norm = ImageNormalize(vmin=vmins[s], vmax=vmaxs[s], stretch=stretch)
                # norm = colors.Normalize(vmin=vmins[s], vmax=vmaxs[s])
                eomap_ = pmX.Sunmap(eomap)
                eomap_.imshow(axes=ax, cmap=cmap, norm=norm)
                eomap_.draw_limb(axes=ax, lw=0.5, alpha=0.5)
                eomap_.draw_grid(axes=ax, grid_spacing=10. * u.deg, lw=0.5)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                warning_label = eovsa_preview_warning_label(eomap)
                if review_label:
                    ax.text(0.02, 0.98, review_label,
                            transform=ax.transAxes, color='#ff6b6b', ha='left', va='top', fontsize=9,
                            bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=2.0))
                if warning_label:
                    ax.text(0.02, 0.88 if review_label else 0.98, warning_label,
                            transform=ax.transAxes, color='#ffd166', ha='left', va='top', fontsize=9,
                            bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', pad=2.0))
                ax.text(0.02, 0.02,
                        eovsa_preview_label(eomap),
                        transform=ax.transAxes, color='w', ha='left', va='bottom', fontsize=9)
                ax.text(0.98, 0.02, 'Max Tb {:.0f} K'.format(np.nanmax(eomap.data)),
                        transform=ax.transAxes, color='w', ha='right', va='bottom', fontsize=9)
                ax.set_xlim(-1227, 1227)
                ax.set_ylim(-1227, 1227)

                for l, dpi in dpis_dict.items():
                    figname = os.path.join(imgoutdir, eovsa_preview_filename(l, s + 1, fits_tag=fits_tag))
                    fig.savefig(figname, dpi=int(dpi), pil_kwargs={"quality":85})
                    print('EOVSA image saved to {}'.format(figname))
            except Exception as err:
                print('Fail to plot {}'.format(eofile))
                print(err)
    if mkfig:
        pass
    else:
        plt.close(fig)
    return


def plot_sdo_func(sdofile, ax, dpis_dict, key, imgoutdir, fig):
    sdomap = smap.Map(sdofile)
    norm = colors.Normalize()
    sdomap_ = pmX.Sunmap(sdomap)
    if "HMI" in key:
        cmap = plt.get_cmap('gray')
    else:
        cmap = plt.get_cmap('sdoaia' + key.lstrip('0'))
    sdomap_.imshow(axes=ax, cmap=cmap, norm=norm)
    sdomap_.draw_limb(axes=ax, lw=0.5, alpha=0.5)
    sdomap_.draw_grid(axes=ax, grid_spacing=10. * u.deg, lw=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.text(0.02, 0.02,
            '{}/{} {}  {}'.format(sdomap.observatory, sdomap.instrument.split(' ')[0], sdomap.measurement,
                                  sdomap.date.strftime('%d-%b-%Y %H:%M UT')),
            transform=ax.transAxes, color='w', ha='left', va='bottom', fontsize=9)
    ax.set_xlim(-1227, 1227)
    ax.set_ylim(-1227, 1227)

    for l, dpi in dpis_dict.items():
        figname = os.path.join(imgoutdir, '{}{}.jpg'.format(l, key))
        if os.path.exists(figname):
            os.system('rm -rf ' + figname)
        fig.savefig(figname, dpi=int(dpi), pil_kwargs={"quality": 85})

def pltSdoQlookImage(datestr, dpis_dict, fig=None, ax=None, overwrite=False, verbose=False, clearcache=False, debug=False):
    plt.ioff()
    dateobj = datetime.strptime(datestr, "%Y-%m-%d")
    datestrdir = dateobj.strftime("%Y/%m/%d/")
    imgindir = os.path.join(imgfitstmpdir, datestr)
    imgoutdir = pltfigdir + datestrdir
    if not os.path.exists(imgindir):
        os.makedirs(imgindir)

    aiaDataSource = {"0094": 8,
                     "0193": 11,
                     "0335": 14,
                     # "4500": 17,
                     "0171": 10,
                     "0304": 13,
                     "0131": 9,
                     "1700": 16,
                     "0211": 12,
                     # "1600": 15,
                     "_HMIcont": 18,
                     "_HMImag": 19}

    if fig is None or ax is None:
        mkfig = True
    else:
        mkfig = False

    if mkfig:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    if verbose: print('Processing SDO images for date {}'.format(dateobj.strftime('%Y-%m-%d')))
    for key, sourceid in aiaDataSource.items():
        fexists = []
        for l, dpi in dpis_dict.items():
            figname = os.path.join(imgoutdir, '{}{}.jpg'.format(l, key))
            fexists.append(os.path.exists(figname))

        if overwrite or (False in fexists):
            sdourl = 'https://api.helioviewer.org/v2/getJP2Image/?date={}T20:00:00Z&sourceId={}'.format(datestr,
                                                                                                        sourceid)
            sdofile = os.path.join(imgindir, key + '.jp2')
            if overwrite and os.path.exists(sdofile):
                os.system('rm -rf {}'.format(sdofile))
            if not os.path.exists(sdofile):
                timeout_hits = 0
                while timeout_hits < HELIOVIEWER_TIMEOUT_LIMIT and not os.path.exists(sdofile):
                    try:
                        with urllib.request.urlopen(sdourl, timeout=HELIOVIEWER_TIMEOUT_S) as response:
                            with open(sdofile, 'wb') as outfp:
                                outfp.write(response.read())
                    except (socket.timeout, TimeoutError, urllib.error.URLError) as err:
                        timeout_hits += 1
                        print('The connection with {} has timed out (attempt {}/{}).'.format(
                            sdourl, timeout_hits, HELIOVIEWER_TIMEOUT_LIMIT))
                        print(err)
                        if timeout_hits >= HELIOVIEWER_TIMEOUT_LIMIT:
                            print('Skipping {} after {} timeouts.'.format(sdourl, timeout_hits))
                            break
                ax.cla()

            if not os.path.exists(sdofile): continue
            if not os.path.exists(imgoutdir): os.makedirs(imgoutdir)

            if debug:
                plot_sdo_func(sdofile, ax, dpis_dict, key, imgoutdir, fig)
            else:
                try:
                    plot_sdo_func(sdofile, ax, dpis_dict, key, imgoutdir, fig)
                except Exception as err:
                    print('Fail to plot {}'.format(sdofile))
                    print(err)
    if clearcache:
        os.system('rm -rf ' + imgindir)

    if mkfig:
        pass
    else:
        plt.close(fig)
    return


def pltBbsoQlookImage(datestr, dpis_dict, fig=None, ax=None, overwrite=False, verbose=False, clearcache=False):
    from astropy.io import fits
    from html.parser import HTMLParser
    class MyHTMLParser(HTMLParser):
        def __init__(self, prefix='bbso_halph_fr_', suffix='.fts'):
            HTMLParser.__init__(self)
            self.prefix = prefix
            self.suffix = suffix

        def handle_starttag(self, tag, attrs):
            if tag != 'a':
                return
            for name, value in attrs:
                if name == "href":
                    if value.startswith(self.prefix) and value.endswith(self.suffix):
                        self.links.append(value)

    def extract(url, prefix='bbso_halph_fr_', suffix='.fts'):
        import urllib.request
        with urllib.request.urlopen(url, timeout=QUERY_TIMEOUT_S) as response:
            f = response.read()

        parser = MyHTMLParser(prefix, suffix)
        parser.links = []
        parser.feed(str(f))
        return parser.links

    bbsodir = 'http://www.bbso.njit.edu/pub/archive/'
    plt.ioff()
    dateobj = datetime.strptime(datestr, "%Y-%m-%d")
    datestrdir = dateobj.strftime("%Y/%m/%d/")
    imgindir = os.path.join(imgfitstmpdir, datestr)
    imgoutdir = pltfigdir + datestrdir
    if not os.path.exists(imgindir):
        os.makedirs(imgindir)

    bbsoDataSource = {"_Halph_fr": ["bbso_halph_fr_", ".fts"]}

    if fig is None or ax is None:
        mkfig = True
    else:
        mkfig = False

    if mkfig:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    if verbose: print('Processing BBSO images for date {}'.format(dateobj.strftime('%Y-%m-%d')))
    for key, sourceid in bbsoDataSource.items():
        fexists = []
        for l, dpi in dpis_dict.items():
            figname = os.path.join(imgoutdir, '{}{}.jpg'.format(l, key))
            fexists.append(os.path.exists(figname))

        if overwrite or (False in fexists):
            bbsosite = os.path.join(bbsodir, datestrdir)
            filelist = extract(bbsosite, sourceid[0], sourceid[1])
            if filelist:
                tfilelist = Time(
                    [datetime.strptime(tf.replace(sourceid[0], '').replace(sourceid[1], ''), "%Y%m%d_%H%M%S") for tf in
                     filelist])
                bbsourl = os.path.join(bbsosite, filelist[
                    np.nanargmin(np.abs(np.array(tfilelist.mjd - (Time(dateobj).mjd + 20. / 24.))))])

                bbsofile = os.path.join(imgindir, key + '.fits')
                if not os.path.exists(bbsofile):
                    try:
                        with urllib.request.urlopen(bbsourl, timeout=QUERY_TIMEOUT_S) as response:
                            with open(bbsofile, 'wb') as outfp:
                                outfp.write(response.read())
                    except (socket.timeout, TimeoutError, urllib.error.URLError) as err:
                        print('The connection with {} has timed out. Skipped!'.format(bbsourl))
                        print(err)
                ax.cla()
                if not os.path.exists(bbsofile): continue
                if not os.path.exists(imgoutdir): os.makedirs(imgoutdir)
                try:
                    hdu = fits.open(bbsofile)[0]
                    header = hdu.header
                    header['WAVELNTH'] = 6562.8
                    header['WAVEUNIT'] = 'angstrom'
                    header['WAVE_STR'] = 'Halph'
                    header['CTYPE1'] = 'HPLN-TAN'
                    header['CUNIT1'] = 'arcsec'
                    header['CTYPE2'] = 'HPLT-TAN'
                    header['CUNIT2'] = 'arcsec'
                    header['DATE-OBS'] = header['DATE_OBS']
                    for k in ['CONTRAST', 'WAVE ERR']:
                        try:
                            header.remove(k)
                        except:
                            pass

                    bbsomap = smap.Map(hdu.data, header)
                    med = np.nanmean(bbsomap.data)
                    norm = colors.Normalize(vmin=med - 1500, vmax=med + 1500)
                    bbsomap_ = pmX.Sunmap(bbsomap)
                    cmap = plt.get_cmap('sdoaia304')
                    bbsomap_.imshow(axes=ax, cmap=cmap, norm=norm)
                    bbsomap_.draw_limb(axes=ax, lw=0.5, alpha=0.5)
                    bbsomap_.draw_grid(axes=ax, grid_spacing=10. * u.deg, lw=0.5)
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.text(0.02, 0.02,
                            '{}  {}'.format(bbsomap.instrument, bbsomap.date.strftime('%d-%b-%Y %H:%M UT')),
                            transform=ax.transAxes, color='w', ha='left', va='bottom', fontsize=9)
                    ax.set_xlim(-1227, 1227)
                    ax.set_ylim(-1227, 1227)
                    ax.set_facecolor('k')

                    for l, dpi in dpis_dict.items():
                        figname = os.path.join(imgoutdir, '{}{}.jpg'.format(l, key))
                        fig.savefig(figname, dpi=int(dpi), pil_kwargs={"quality":85})
                except Exception as err:
                    print('Fail to plot {}'.format(bbsofile))
                    print(err)
    if clearcache:
        os.system('rm -rf ' + imgindir)

    if mkfig:
        pass
    else:
        plt.close(fig)
    return

def main(dateobj=None, ndays=1, clearcache=False, ovwrite_eovsa=False, ovwrite_sdo=False,
         ovwrite_bbso=False, show_warning=False, debug=False, version='all', fits_tag='',
         include_fine_bands=True):
    """
    Main pipeline for plotting EOVSA daily full-disk images at multiple frequencies.

    :param dateobj: Starting datetime for processing. If None, defaults to two days before now.
    :type dateobj: datetime, optional
    :param ndays: Number of days to process (spanning from dateobj - ndays to dateobj); default is 1.
    :type ndays: int, optional
    :param clearcache: If True, remove temporary files after processing; default is False.
    :type clearcache: bool, optional
    :param ovwrite_eovsa: If True, overwrite existing EOVSA images; default is False.
    :type ovwrite_eovsa: bool, optional
    :param ovwrite_sdo: If True, overwrite existing SDO images; default is False.
    :type ovwrite_sdo: bool, optional
    :param ovwrite_bbso: If True, overwrite existing BBSO images; default is False.
    :type ovwrite_bbso: bool, optional
    :param show_warning: If True, show warnings during processing; default is False.
    :type show_warning: bool, optional
    :param debug: If True, run the pipeline in debugging mode; default is False.
    :type debug: bool, optional
    :param version: EOVSA product version to plot, or "all" for the public v1/v2 pair.
    :type version: str, optional
    :param fits_tag: Optional tag inserted after eovsa.synoptic_daily for alternate FITS products.
    :type fits_tag: str, optional
    :param include_fine_bands: If True (default), the v2 daily plotting path additionally
        discovers and plots any non-standard band-segmentation daily products found
        alongside the standard 7 bands, using frequency-interpolated color scaling.
        On dates where only the standard 7 bands exist this is a no-op (nothing extra
        is found), so default output is unchanged. Set False to disable discovery
        entirely (e.g. from --no-fine-bands) as a production kill-switch.
    :type include_fine_bands: bool, optional
    :raises Exception: If an error occurs during processing.
    :return: None
    :rtype: None
    """
    import warnings
    import numpy as np
    from datetime import timedelta
    from astropy.time import Time
    import matplotlib.pyplot as plt

    if not show_warning:
        import warnings
        warnings.filterwarnings("ignore")

    # Determine the end date for processing.
    ted = dateobj if dateobj is not None else (datetime.now() - timedelta(days=2))
    # Calculate the start date based on ndays.
    tst = Time(np.fix(Time(ted).mjd) - ndays, format='mjd').datetime
    tsep = datetime.strptime(SPW_EPOCH_SPLIT_DATE, "%Y-%m-%d")

    # vmaxs = [22.0e4, 8.0e4, 5.4e4, 3.5e4, 2.3e4, 1.8e4, 1.5e4]
    # vmins = [-9.0e3, -5.5e3, -3.4e3, -2.5e3, -2.5e3, -2.5e3, -2.5e3]
    vmaxs = [70.0e4, 30e4, 18e4, 13e4, 8e4, 6e4, 6e4]
    vmins = [-18.0e3, -8e3, -4.8e3, -3.4e3, -2.1e3, -1.6e3, -1.6e3]

    dpis = np.array([256, 512, 1024]) / 8
    dpis_dict_eo = {'t': dpis[0], 'l': dpis[1], 'f': dpis[2]}
    dpis = np.array([256, 512, 1024]) / 8
    dpis_dict_sdo = {'t': dpis[0], 'l': dpis[1], 'f': dpis[2]}
    dpis = np.array([256, 512, 1024]) / 8
    dpis_dict_bbso = {'t': dpis[0], 'l': dpis[1], 'f': dpis[2]}

    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    dateobs = tst
    while dateobs < ted:
        # Determine spectral window settings based on the observation date.
        if dateobs > tsep:
            spws = list(SPWS_52BAND_ALT)
            spws_v3 = list(SPWS_52BAND)
        else:
            spws = list(SPWS_34BAND)
            spws_v3 = spws

        datestr = dateobs.strftime("%Y-%m-%d")
        eovsa_versions = ['v1.0', 'v2.0'] if version == 'all' else [version]
        for eovsa_version in eovsa_versions:
            version_fits_tag = '' if eovsa_version == 'v1.0' else fits_tag
            if eovsa_version == 'v1.0':
                pltEovsaQlookImage(datestr, spws, vmaxs, vmins, dpis_dict_eo, fig, ax,
                                    overwrite=ovwrite_eovsa, verbose=True, version=eovsa_version,
                                    fits_tag=version_fits_tag)
            else:
                pltEovsaQlookImage_v3(datestr, spws_v3, vmaxs, vmins, dpis_dict_eo, fig, ax,
                                       overwrite=ovwrite_eovsa, verbose=True, version=eovsa_version,
                                       fits_tag=version_fits_tag, include_fine_bands=include_fine_bands)
        try:
            pltSdoQlookImage(datestr, dpis_dict_sdo, fig, ax,
                             overwrite=ovwrite_sdo, verbose=True, clearcache=clearcache, debug=debug)
        except Exception as err:
            print('Skipping optional SDO preview for date {} after error.'.format(datestr))
            print(err)
        try:
            pltBbsoQlookImage(datestr, dpis_dict_bbso, fig, ax,
                              overwrite=ovwrite_bbso, verbose=True, clearcache=clearcache)
        except Exception as err:
            print('Skipping optional BBSO preview for date {} after error.'.format(datestr))
            print(err)
        dateobs = dateobs + timedelta(days=1)

if __name__ == '__main__':
    import argparse
    import os
    from datetime import datetime, timedelta
    from astropy.time import Time

    parser = argparse.ArgumentParser(
        description='Pipeline for plotting EOVSA daily full-disk images at multiple frequencies.'
    )
    # Default date is set to one day before the current date at 20:00 UT,
    # formatted as YYYY-MM-DDT20:00.
    default_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT20:00')
    parser.add_argument(
        '--date', type=str, default=default_date,
        help='Date to process in YYYY-MM-DDT20:00 format, defaults to 20:00 UT one day before the current date.'
    )
    parser.add_argument(
        '--ndays', type=int, default=1,
        help='Process data spanning from DATE minus ndays to DATE (default: 1 days).'
    )
    parser.add_argument(
        '--clearcache', action='store_true',
        help='Remove temporary files after processing.'
    )
    parser.add_argument(
        '--ovwrite_eovsa', action='store_true',
        help='Overwrite existing EOVSA images.'
    )
    parser.add_argument(
        '--ovwrite_sdo', action='store_true',
        help='Overwrite existing SDO images.'
    )
    parser.add_argument(
        '--ovwrite_bbso', action='store_true',
        help='Overwrite existing BBSO images.'
    )
    parser.add_argument(
        '--show_warning', action='store_true',
        help='Show warnings during processing.'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Run the pipeline in debugging mode.'
    )
    parser.add_argument(
        '--version', type=str, default='all',
        help='EOVSA product version to plot into vX.Y preview folders, or "all" for the public v1/v2 pair.'
    )
    parser.add_argument(
        '--fits-tag', type=str, default='',
        help='Optional tag inserted after eovsa.synoptic_daily for alternate FITS products.'
    )
    parser.add_argument(
        '--no-fine-bands', dest='no_fine_bands', action='store_true',
        help='Disable discovery/plotting of non-standard band-segmentation daily products '
             '(fine-spectral chunks, custom groups) in the v2 daily plotting path. By default '
             '(flag absent) discovery is automatic: additional eovsa.synoptic_daily*.s*.tb.disk.fits '
             'tags beyond the standard 7 bands are found and plotted with frequency-interpolated '
             'color scaling; this flag is a kill-switch for production if that behavior is ever unwanted.'
    )
    # Optional positional date arguments: year month day (overrides --date if provided)
    parser.add_argument(
        'date_args', type=int, nargs='*',
        help='Optional date arguments: year month day. If provided, overrides --date.'
    )


    args = parser.parse_args()

    # Determine the processing date.
    if len(args.date_args) == 3:
        year, month, day = args.date_args
        dateobj = datetime(year, month, day, 20)  # Use 20:00 UT for the specified date.
    else:
        dateobj = Time(args.date).datetime

    print(f"Running pipeline_plt for date {dateobj.strftime('%Y-%m-%d')}.")
    print("Arguments:")
    print(f"  ndays: {args.ndays}")
    print(f"  clearcache: {args.clearcache}")
    print(f"  ovwrite_eovsa: {args.ovwrite_eovsa}")
    print(f"  ovwrite_sdo: {args.ovwrite_sdo}")
    print(f"  ovwrite_bbso: {args.ovwrite_bbso}")
    print(f"  show_warning: {args.show_warning}")
    print(f"  debug: {args.debug}")
    print(f"  version: {args.version}")
    print(f"  fits_tag: {args.fits_tag}")
    print(f"  no_fine_bands: {args.no_fine_bands}")

    # Run the main pipeline function with the datetime object.
    main(
        dateobj=dateobj,
        ndays=args.ndays,
        clearcache=args.clearcache,
        ovwrite_eovsa=args.ovwrite_eovsa,
        ovwrite_sdo=args.ovwrite_sdo,
        ovwrite_bbso=args.ovwrite_bbso,
        show_warning=args.show_warning,
        debug=args.debug,
        version=args.version,
        fits_tag=args.fits_tag,
        include_fine_bands=not args.no_fine_bands
    )
