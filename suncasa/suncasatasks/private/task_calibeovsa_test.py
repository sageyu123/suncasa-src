from ...casa_compat import check_dependencies

check_dependencies()

import platform
import matplotlib

if platform.system() == 'Linux':
    matplotlib.use('Agg')
import os
import shutil
import json
import numpy as np

from eovsapy.util import extract as eoextract
from eovsapy.util import Time
from eovsapy import cal_header as ch
from eovsapy import dbutil as db
from eovsapy import pipeline_cal as pc
from eovsapy.sqlutil import sql2refcalX, sql2phacalX
from .. import concateovsa
from suncasa.eovsa.update_log import EOVSA15_UPGRADE_DATE, DCM_IF_FILTER_UPGRADE_DATE

from ...casa_compat import import_casatools, import_casatasks

REFCAL_NPZ_MODES = ('triplet', 'smooth_model', 'bph_sbd')

tasks = import_casatasks('split', 'tclean', 'gencal', 'clearcal', 'applycal', 'flagdata', 'casalog', 'bandpass')
split = tasks.get('split')
tclean = tasks.get('tclean')
gencal = tasks.get('gencal')
clearcal = tasks.get('clearcal')
applycal = tasks.get('applycal')
flagdata = tasks.get('flagdata')
casalog = tasks.get('casalog')
bandpass = tasks.get('bandpass')

tools = import_casatools(['tbtool', 'mstool', 'qatool', 'iatool'])

tbtool = tools['tbtool']
mstool = tools['mstool']
qatool = tools['qatool']
iatool = tools['iatool']
tb = tbtool()
ms = mstool()
qa = qatool()
ia = iatool()


def _normalize_refcal_npz_mode(mode):
    mode = str(mode or 'smooth_model').strip().lower().replace('-', '_')
    if mode not in REFCAL_NPZ_MODES:
        raise ValueError(
            "refcal_npz_mode must be one of {0}; got {1!r}".format(
                ", ".join(REFCAL_NPZ_MODES),
                mode,
            )
        )
    return mode


def _refcal_npz_mode_label(mode):
    if mode == 'smooth_model':
        return 'smooth-model refcal mode'
    if mode == 'bph_sbd':
        return 'saved band-phase + SBD refcal mode'
    return 'legacy triplet refcal mode'


def _npz_json_scalar(data, key, default):
    if key not in data.files:
        return default
    try:
        return json.loads(str(np.asarray(data[key]).item()))
    except Exception:
        return default


def _merge_npz_promoted_anchor_arrays(data, promoted):
    if 'refcal__promoted_anchor_antennas' not in data.files:
        return promoted if isinstance(promoted, dict) else {}
    out = dict(promoted) if isinstance(promoted, dict) else {}
    antennas = np.asarray(data['refcal__promoted_anchor_antennas'], dtype=np.int32).reshape(-1)
    scan_ids = np.asarray(
        data['refcal__promoted_anchor_scan_ids'] if 'refcal__promoted_anchor_scan_ids' in data.files else [],
        dtype=np.int32,
    ).reshape(-1)
    timestamps = np.asarray(
        data['refcal__promoted_anchor_timestamp_lv'] if 'refcal__promoted_anchor_timestamp_lv' in data.files else [],
        dtype=np.float64,
    ).reshape(-1)
    active_ns = np.asarray(
        data['refcal__promoted_anchor_active_ns'] if 'refcal__promoted_anchor_active_ns' in data.files else [],
        dtype=np.float64,
    )
    for idx, ant_i in enumerate(antennas.tolist()):
        entry = dict(out.get(str(int(ant_i))) or {})
        if idx < scan_ids.size and int(scan_ids[idx]) >= 0:
            entry['from_phacal_scan_id'] = int(scan_ids[idx])
        if idx < timestamps.size and np.isfinite(timestamps[idx]):
            entry['from_phacal_timestamp_lv'] = float(timestamps[idx])
            try:
                entry.setdefault('from_phacal_timestamp_iso', Time(float(timestamps[idx]), format='lv').iso[:19])
            except Exception:
                pass
        if active_ns.ndim == 2 and idx < active_ns.shape[0]:
            row = np.asarray(active_ns[idx, :2], dtype=np.float64)
            if np.any(np.isfinite(row)):
                entry['anchor_active_ns'] = row.tolist()
        out[str(int(ant_i))] = entry
    return out


def _promoted_source_antennas_for_phacal(refcal, scan_id, nant):
    promoted = refcal.get('promoted_antennas') or {}
    if not isinstance(promoted, dict):
        return []
    out = []
    for ant_key, meta in promoted.items():
        if not isinstance(meta, dict):
            continue
        try:
            ant_i = int(ant_key)
            source_scan_id = int(meta.get('from_phacal_scan_id'))
        except (TypeError, ValueError):
            continue
        if source_scan_id == int(scan_id) and 0 <= ant_i < int(nant):
            out.append(ant_i)
    return sorted(set(out))


def _apply_promoted_source_phacal_self_anchor(refcal, phacal, scan_id):
    pslope = np.asarray(phacal.get('pslope', []), dtype=np.float64)
    poff = np.asarray(phacal.get('poff', []), dtype=np.float64)
    if pslope.ndim != 2 or poff.ndim != 2 or pslope.shape[0] == 0:
        return []
    updated = []
    for ant_i in _promoted_source_antennas_for_phacal(refcal, scan_id, pslope.shape[0]):
        if ant_i >= poff.shape[0]:
            continue
        pslope[ant_i, :min(2, pslope.shape[1])] = pslope[0, :min(2, pslope.shape[1])]
        poff[ant_i, :min(2, poff.shape[1])] = poff[0, :min(2, poff.shape[1])]
        updated.append(int(ant_i))
    return updated


def _load_npz_triplet(data, prefix, fallback_flag):
    keys = [
        prefix + 'model_phi_band_rad',
        prefix + 'model_tau_ib_ns',
        prefix + 'model_tau_mb_eff_ns',
        prefix + 'model_band_ref_freq_ghz',
    ]
    if not all(key in data.files for key in keys):
        return None
    phi = np.asarray(data[prefix + 'model_phi_band_rad'], dtype=np.float64)
    if prefix + 'model_flag' in data.files:
        model_flag = np.asarray(data[prefix + 'model_flag'])
    elif fallback_flag is not None and np.asarray(fallback_flag).shape == phi.shape:
        model_flag = np.asarray(fallback_flag)
    else:
        model_flag = np.zeros(phi.shape, dtype=np.int32)
    return {
        'phi_band_rad': phi,
        'tau_ib_ns': np.asarray(data[prefix + 'model_tau_ib_ns'], dtype=np.float64),
        'tau_mb_eff_ns': np.asarray(data[prefix + 'model_tau_mb_eff_ns'], dtype=np.float64),
        'band_ref_freq_ghz': np.asarray(data[prefix + 'model_band_ref_freq_ghz'], dtype=np.float64),
        'flag': model_flag,
    }


def _triplet_has_band(triplet, band_i):
    if not triplet or band_i < 0:
        return False
    phi = np.asarray(triplet.get('phi_band_rad', []), dtype=np.float64)
    tau_ib = np.asarray(triplet.get('tau_ib_ns', []), dtype=np.float64)
    tau_mb = np.asarray(triplet.get('tau_mb_eff_ns', []), dtype=np.float64)
    band_ref = np.asarray(triplet.get('band_ref_freq_ghz', []), dtype=np.float64)
    flag = np.asarray(triplet.get('flag', []))
    if (
        phi.ndim != 3
        or tau_ib.ndim not in (2, 3)
        or tau_mb.ndim != 2
        or band_i >= phi.shape[2]
        or band_i >= band_ref.size
        or not np.isfinite(band_ref[band_i])
    ):
        return False
    valid = np.isfinite(phi[:, :, band_i]) & np.isfinite(tau_mb)
    if tau_ib.ndim == 3:
        if band_i >= tau_ib.shape[2]:
            return False
        valid &= np.isfinite(tau_ib[:, :, band_i])
    else:
        valid &= np.isfinite(tau_ib)
    if flag.ndim == 3 and band_i < flag.shape[2]:
        valid &= flag[:, :, band_i] == 0
    return bool(np.any(valid))


def _triplet_value(triplet, key, ant_i, pol_i, band_i, default=np.nan):
    arr = np.asarray(triplet.get(key, []), dtype=np.float64)
    if arr.ndim == 3:
        if ant_i < arr.shape[0] and pol_i < arr.shape[1] and band_i < arr.shape[2]:
            return arr[ant_i, pol_i, band_i]
    elif arr.ndim == 2:
        if ant_i < arr.shape[0] and pol_i < arr.shape[1]:
            return arr[ant_i, pol_i]
    elif arr.ndim == 1:
        if band_i < arr.size:
            return arr[band_i]
    return default


def _triplet_flagged(triplet, ant_i, pol_i, band_i):
    flag = np.asarray(triplet.get('flag', []))
    return (
        flag.ndim == 3
        and ant_i < flag.shape[0]
        and pol_i < flag.shape[1]
        and band_i < flag.shape[2]
        and flag[ant_i, pol_i, band_i] == 1
    )


def _triplet_slot_usable(triplet, ant_i, pol_i, band_i):
    if not triplet or band_i < 0:
        return False
    phi = np.asarray(triplet.get('phi_band_rad', []), dtype=np.float64)
    tau_ib = np.asarray(triplet.get('tau_ib_ns', []), dtype=np.float64)
    tau_mb = np.asarray(triplet.get('tau_mb_eff_ns', []), dtype=np.float64)
    band_ref = np.asarray(triplet.get('band_ref_freq_ghz', []), dtype=np.float64)
    if (
        phi.ndim != 3
        or ant_i >= phi.shape[0]
        or pol_i >= phi.shape[1]
        or band_i >= phi.shape[2]
        or band_i >= band_ref.size
        or tau_mb.ndim != 2
        or ant_i >= tau_mb.shape[0]
        or pol_i >= tau_mb.shape[1]
        or _triplet_flagged(triplet, ant_i, pol_i, band_i)
    ):
        return False
    if tau_ib.ndim == 3:
        if ant_i >= tau_ib.shape[0] or pol_i >= tau_ib.shape[1] or band_i >= tau_ib.shape[2]:
            return False
        ib_ns = tau_ib[ant_i, pol_i, band_i]
    elif tau_ib.ndim == 2:
        if ant_i >= tau_ib.shape[0] or pol_i >= tau_ib.shape[1]:
            return False
        ib_ns = tau_ib[ant_i, pol_i]
    else:
        return False
    return bool(
        np.isfinite(phi[ant_i, pol_i, band_i])
        and np.isfinite(ib_ns)
        and np.isfinite(tau_mb[ant_i, pol_i])
        and np.isfinite(band_ref[band_i])
    )


def _operator_flagged(refcal, ant_i, pol_i, band_i):
    flags = np.asarray(refcal.get('operator_band_flag', []), dtype=np.uint8)
    return (
        flags.ndim == 3
        and ant_i < flags.shape[0]
        and pol_i < flags.shape[1]
        and band_i < flags.shape[2]
        and flags[ant_i, pol_i, band_i] != 0
    )


def _array_value_3d(values, ant_i, pol_i, band_i, default=np.nan):
    arr = np.asarray(values, dtype=np.float64)
    if (
        arr.ndim == 3
        and ant_i < arr.shape[0]
        and pol_i < arr.shape[1]
        and band_i < arr.shape[2]
    ):
        return arr[ant_i, pol_i, band_i]
    return default


def _array_flagged_3d(values, ant_i, pol_i, band_i):
    arr = np.asarray(values)
    return (
        arr.ndim == 3
        and ant_i < arr.shape[0]
        and pol_i < arr.shape[1]
        and band_i < arr.shape[2]
        and arr[ant_i, pol_i, band_i] != 0
    )


def _lo_data_ignored_for_ant(refcal, ant_i):
    ignored = np.asarray(refcal.get('lo_data_ignored_antennas', []), dtype=np.uint8).reshape(-1)
    return ant_i < ignored.size and ignored[ant_i] != 0


def _npz_triplet_for_band(refcal, band_i):
    triplets = refcal.get('gencal_triplets') or {}
    lo = triplets.get('lo')
    hi = triplets.get('hi') or triplets.get('legacy') or refcal.get('gencal_triplet')
    if _triplet_has_band(hi, band_i):
        return 'hi', hi
    if _triplet_has_band(lo, band_i):
        return 'lo', lo
    return None, None


def _apply_promoted_active_sbd(active_ns, promoted):
    if active_ns.ndim != 2 or not isinstance(promoted, dict):
        return 0
    updated = 0
    for ant_key, meta in promoted.items():
        if not isinstance(meta, dict):
            continue
        try:
            ant_i = int(ant_key)
        except (TypeError, ValueError):
            continue
        if ant_i < 0 or ant_i >= active_ns.shape[0]:
            continue
        values = np.asarray(meta.get('anchor_active_ns', []), dtype=np.float64).reshape(-1)
        if values.size == 0:
            continue
        npol = min(active_ns.shape[1], values.size, 2)
        if npol <= 0 or not np.any(np.isfinite(values[:npol])):
            continue
        finite = np.isfinite(values[:npol])
        active_ns[ant_i, :npol] = np.where(finite, values[:npol], active_ns[ant_i, :npol])
        updated += 1
    return updated


def _smooth_refcal_active_sbd(refcal):
    active = np.asarray(refcal.get('active_ns', []), dtype=np.float64)
    if active.ndim != 2 or active.shape[1] < 2 or active.size == 0:
        raise ValueError(
            "smooth-model refcal mode cannot obtain a reliable SBD source from the NPZ; "
            "missing refcal__active_ns/effective active in-band delay"
        )
    return active


def _smooth_refcal_sbd_for_band(refcal, active_sbd, ant_i, pol_i, band_i):
    triplets = refcal.get('gencal_triplets') or {}
    for triplet in (triplets.get('hi'), triplets.get('legacy'), refcal.get('gencal_triplet')):
        if not _triplet_has_band(triplet, band_i):
            continue
        if _triplet_flagged(triplet, ant_i, pol_i, band_i):
            continue
        ib_ns = _triplet_value(triplet, 'tau_ib_ns', ant_i, pol_i, band_i, default=np.nan)
        mb_ns = _triplet_value(triplet, 'tau_mb_eff_ns', ant_i, pol_i, band_i, default=np.nan)
        if np.isfinite(ib_ns) and np.isfinite(mb_ns):
            return float(ib_ns + mb_ns)
    if ant_i < active_sbd.shape[0] and pol_i < active_sbd.shape[1]:
        return float(active_sbd[ant_i, pol_i])
    return np.nan


def _gencal_sbd_phase_at_spw_reference(freq_ghz, sbd_ns):
    del freq_ghz, sbd_ns
    return 0.0


def _triplet_bph_sbd_terms_for_band(triplet, ant_i, pol_i, band_i):
    if not _triplet_slot_usable(triplet, ant_i, pol_i, band_i):
        return np.nan, np.nan
    phase_rad = _triplet_value(triplet, 'phi_band_rad', ant_i, pol_i, band_i, default=np.nan)
    ib_ns = _triplet_value(triplet, 'tau_ib_ns', ant_i, pol_i, band_i, default=np.nan)
    mb_ns = _triplet_value(triplet, 'tau_mb_eff_ns', ant_i, pol_i, band_i, default=np.nan)
    band_ref = _triplet_value(triplet, 'band_ref_freq_ghz', ant_i, pol_i, band_i, default=np.nan)
    if not all(np.isfinite(value) for value in (phase_rad, ib_ns, mb_ns, band_ref)):
        return np.nan, np.nan
    return float(phase_rad + 2.0 * np.pi * float(band_ref) * float(mb_ns)), float(ib_ns + mb_ns)


def _bph_candidate_for_band(refcal, band_phase, band_flag, ant_i, pol_i, band_i, freq_ghz, hi_sbd_ns, label):
    if _operator_flagged(refcal, ant_i, pol_i, band_i):
        return np.nan, np.nan, True, None
    source_prefix = '' if label == 'primary' else label + '_'
    lo_band = np.isfinite(freq_ghz) and float(freq_ghz) <= 3.0
    if lo_band and not _lo_data_ignored_for_ant(refcal, ant_i):
        lo_triplet = (refcal.get('gencal_triplets') or {}).get('lo')
        lo_phase, lo_sbd_ns = _triplet_bph_sbd_terms_for_band(lo_triplet, ant_i, pol_i, band_i)
        if np.isfinite(lo_phase) and np.isfinite(lo_sbd_ns):
            return (
                float(lo_phase),
                float(lo_sbd_ns),
                False,
                source_prefix + 'lo_model',
            )
        # HI smooth/extrapolated model on LO grid: the HI gencal triplet stores the
        # smooth model evaluated at every band center (including LO bands) via
        # require_model_mask=False during NPZ export.  Use it when the LO triplet
        # is absent or fully flagged — SBD stays hi_sbd_ns per the bph_sbd contract.
        hi_triplet = (
            (refcal.get('gencal_triplets') or {}).get('hi')
            or refcal.get('gencal_triplet')
        )
        hi_lo_phase, _ = _triplet_bph_sbd_terms_for_band(hi_triplet, ant_i, pol_i, band_i)
        if np.isfinite(hi_lo_phase) and np.isfinite(hi_sbd_ns):
            return (
                float(hi_lo_phase),
                float(hi_sbd_ns),
                False,
                source_prefix + 'hi_smooth_extrap',
            )
    phase = _array_value_3d(band_phase, ant_i, pol_i, band_i, default=np.nan)
    if np.isfinite(phase) and not _array_flagged_3d(band_flag, ant_i, pol_i, band_i):
        return float(phase), float(hi_sbd_ns), False, source_prefix + 'band_phase'
    return np.nan, np.nan, True, None


def _bph_sbd_phase_base_for_band(
        refcal, band_phase, band_flag, smooth_model_pha,
        ant_i, pol_i, band_i, freq_ghz, hi_sbd_ns):
    phase, sbd_ns, flagged, source = _bph_candidate_for_band(
        refcal, band_phase, band_flag, ant_i, pol_i, band_i, freq_ghz, hi_sbd_ns, 'primary'
    )
    if not flagged and source:
        return phase, sbd_ns, False, source
    secondary = refcal.get('secondary_bph_refcal')
    if isinstance(secondary, dict):
        phase, sbd_ns, flagged, source = _bph_candidate_for_band(
            secondary,
            secondary.get('band_phase_rad', []),
            secondary.get('band_phase_flag', []),
            ant_i,
            pol_i,
            band_i,
            freq_ghz,
            hi_sbd_ns,
            'secondary',
        )
        if not flagged and source:
            return phase, sbd_ns, False, source
    # HI smooth/extrapolated model on LO grid when both primary and secondary
    # candidates fail.  ``smooth_model_pha`` holds the NaN-preserving
    # refcal__model_pha from the NPZ, which carries HI smooth-model values at LO
    # band centers when the NPZ was saved with require_model_mask=False.  This
    # fallback fires precisely *because* no valid LO phase exists for the slot
    # (LO absent or LO-flagged), so it is gated only on a finite model value and
    # the operator flag -- NOT on the raw LO data flag, which is the trigger
    # condition itself.  A bad HI model is excluded via the operator flag.  SBD
    # stays hi_sbd_ns per the bph_sbd contract.
    lo_band = np.isfinite(freq_ghz) and float(freq_ghz) <= 3.0
    if lo_band and not _operator_flagged(refcal, ant_i, pol_i, band_i):
        hi_phase = _array_value_3d(smooth_model_pha, ant_i, pol_i, band_i, default=np.nan)
        if np.isfinite(hi_phase):
            return float(hi_phase), float(hi_sbd_ns), False, 'hi_smooth_extrap'
    return 0.0, np.nan, True, 'bph_missing'


def load_calwidget_v2_npz(npz_path):
    '''Load refcal + phacals from a calwidget v2 calibeovsa-ready NPZ.

    Returns a tuple ``(refcal, phacals)`` whose dict layout matches what
    :func:`eovsapy.sqlutil.sql2refcalX` and :func:`sql2phacalX` produce, so the
    rest of :func:`calibeovsa` can consume them unchanged.

    Promoted-antenna metadata (the per-antenna source phacal timestamp written
    by ``Promote to Refcal``) is attached to ``refcal['promoted_antennas']``.
    Newer NPZs also carry per-antenna phacal anchor scan/time arrays.
    '''
    with np.load(npz_path, allow_pickle=False) as data:
        kind = str(data['kind'])
        if kind != 'calibeovsa_v1':
            raise ValueError(
                'Unrecognized calwidget v2 npz kind {0!r} in {1}'.format(kind, npz_path)
            )
        vis = data['refcal__vis_real'] + 1j * data['refcal__vis_imag']
        raw_flag = np.asarray(data['refcal__flag'])
        flag = np.asarray(raw_flag).copy()
        # Prefer the widget's smooth analytic-fit phase (Chebyshev for Ant 1,
        # polynomial for Ant 2+) as the calibration phase — that IS the
        # calibration the widget intends to apply. Fall back to the raw
        # band-averaged vis phase for older NPZs without refcal__model_pha.
        # NaN entries in the model mark (ant, pol, band) combinations the
        # widget did not fit (missing bands, masked antennas); promote those
        # to flag=1 so downstream zeroing (pha[flag==1]=0) handles them.
        has_model_pha = 'refcal__model_pha' in data.files and data['refcal__model_pha'].size > 0
        if has_model_pha:
            model_pha = np.asarray(data['refcal__model_pha'], dtype=np.float64)
            fitted = np.isfinite(model_pha)
            pha = np.where(fitted, model_pha, 0.0)
            flag = np.where(fitted, flag, 1).astype(flag.dtype)
            pha_source = 'refcal__model_pha'
            # NaN-preserving copy of the exported smooth model. The bph_sbd LO
            # HI-extrapolation fallback reads this (not the zero-filled, raw-flag
            # gated ``pha``) so HI smooth-model values at LO band centers survive
            # for slots whose raw LO flag is 1 because no LO refcal was saved.
            model_pha_raw = model_pha
        else:
            pha = np.angle(vis)
            pha_source = 'angle(refcal__vis)'
            model_pha_raw = np.full(np.asarray(pha).shape, np.nan, dtype=np.float64)
        promoted = _merge_npz_promoted_anchor_arrays(
            data,
            _npz_json_scalar(data, 'refcal__promoted_antennas_json', {}),
        )
        refcal = {
            'pha': pha,
            'amp': np.abs(vis),
            'flag': flag,
            'sigma': np.asarray(data['refcal__sigma']),
            'fghz': np.asarray(data['refcal__fghz']),
            'timestamp': Time(float(data['refcal__timestamp_lv']), format='lv'),
            't_bg': Time(float(data['refcal__t_bg_lv']), format='lv'),
            't_ed': Time(float(data['refcal__t_ed_lv']), format='lv'),
            'promoted_antennas': promoted,
            'pha_source': pha_source,
            'model_pha_raw': model_pha_raw,
            'band_phase_rad': np.angle(vis),
            'band_phase_flag': np.asarray(raw_flag).copy(),
        }
        if 'refcal__active_ns' in data.files:
            active_ns = np.asarray(data['refcal__active_ns'], dtype=np.float64).copy()
            active_ns_source = 'refcal__active_ns'
            if _apply_promoted_active_sbd(active_ns, promoted):
                active_ns_source += '+refcal__promoted_anchor_active_ns'
            refcal['active_ns'] = active_ns
            refcal['active_ns_source'] = active_ns_source
        legacy_triplet = _load_npz_triplet(data, 'refcal__', flag)
        hi_triplet = _load_npz_triplet(data, 'refcal__hi_', flag) or legacy_triplet
        lo_triplet = _load_npz_triplet(data, 'refcal__lo_', flag)
        refcal['gencal_triplets'] = {
            'legacy': legacy_triplet,
            'hi': hi_triplet,
            'lo': lo_triplet,
        }
        if hi_triplet is not None:
            refcal['gencal_triplet'] = hi_triplet
        if 'refcal__operator_band_flag' in data.files:
            refcal['operator_band_flag'] = np.asarray(data['refcal__operator_band_flag'], dtype=np.uint8)
        elif 'refcal__manual_ant_flag_override' in data.files and hi_triplet is not None:
            manual = np.asarray(data['refcal__manual_ant_flag_override'], dtype=bool).reshape(-1)
            op = np.zeros(np.asarray(hi_triplet['phi_band_rad']).shape, dtype=np.uint8)
            limit = min(op.shape[0], manual.size)
            if limit:
                op[:limit, :, :] = manual[:limit, None, None]
            refcal['operator_band_flag'] = op
        if 'refcal__lo_data_ignored_antennas' in data.files:
            refcal['lo_data_ignored_antennas'] = np.asarray(
                data['refcal__lo_data_ignored_antennas'], dtype=np.uint8
            )
        phacal_ids = [int(v) for v in np.asarray(data['phacal_scan_ids']).tolist()]
        phacals = []
        for scan_id in phacal_ids:
            prefix = 'phacal_{0}'.format(scan_id)
            inner = {
                'pha': np.asarray(data[prefix + '__phacal_pha']),
                'amp': np.asarray(data[prefix + '__phacal_amp']),
                'flag': np.asarray(data[prefix + '__phacal_flag']),
                'sigma': np.asarray(data[prefix + '__phacal_sigma']),
                'fghz': np.asarray(data[prefix + '__phacal_fghz']),
                'timestamp': Time(float(data[prefix + '__t_pha_lv']), format='lv'),
                't_bg': Time(float(data[prefix + '__phacal_t_bg_lv']), format='lv'),
                't_ed': Time(float(data[prefix + '__phacal_t_ed_lv']), format='lv'),
            }
            phacal = {
                'pslope': np.asarray(data[prefix + '__pslope'], dtype=np.float64).copy(),
                't_pha': Time(float(data[prefix + '__t_pha_lv']), format='lv'),
                'flag': np.asarray(data[prefix + '__flag']),
                'poff': np.asarray(data[prefix + '__poff'], dtype=np.float64).copy(),
                't_ref': Time(float(data[prefix + '__t_ref_lv']), format='lv'),
                't_bg': Time(float(data[prefix + '__t_bg_lv']), format='lv'),
                't_ed': Time(float(data[prefix + '__t_ed_lv']), format='lv'),
                'phacal': inner,
            }
            if prefix + '__anchor_scan_id' in data.files:
                phacal['anchor_scan_id'] = np.asarray(data[prefix + '__anchor_scan_id'], dtype=np.int32)
            if prefix + '__anchor_timestamp_lv' in data.files:
                phacal['anchor_timestamp'] = Time(
                    np.asarray(data[prefix + '__anchor_timestamp_lv'], dtype=np.float64),
                    format='lv',
                )
            if prefix + '__anchor_is_promoted' in data.files:
                phacal['anchor_is_promoted'] = np.asarray(data[prefix + '__anchor_is_promoted'], dtype=np.uint8)
            _apply_promoted_source_phacal_self_anchor(refcal, phacal, scan_id)
            phacals.append(phacal)
    return refcal, phacals


def _attach_secondary_bph_refcal(refcal, secondary_npz):
    if not secondary_npz:
        return None
    secondary_refcal, _secondary_phacals = load_calwidget_v2_npz(secondary_npz)
    refcal['secondary_bph_refcal'] = secondary_refcal
    refcal['secondary_npz'] = str(secondary_npz)
    return secondary_refcal


def flag_phambd_by_spw(caltb, flagspw='0~1'):
    sp_st, sp_ed = flagspw.split('~')
    tb.open(caltb, nomodify=False)
    phambd_spw = tb.getcol('SPECTRAL_WINDOW_ID')
    spwindx = np.where(np.logical_and(phambd_spw >= int(sp_st), phambd_spw <= int(sp_ed)))[0]
    # print(spwindx)

    if 'CPARAM' in tb.colnames():
        data = tb.getcol('CPARAM')
        data[:, :, spwindx[0]:spwindx[-1] + 1][:] = complex(1)
        datakey = 'CPARAM'
    elif 'FPARAM' in tb.colnames():
        data = tb.getcol('FPARAM')
        data[:, :, spwindx[0]:spwindx[-1] + 1][:] = 0.0
        datakey = 'FPARAM'
    else:
        print(f'No calibration data found in {caltb}')
        tb.close()
        return False
    tb.putcol(datakey, data)

    tb.close()
    return True


def calibeovsa(vis=None, caltype=None, caltbdir='', interp=None, docalib=True, doflag=True, flagant='',
               flagspw='', doimage=False, imagedir=None, antenna='', timerange=None, spw=None, stokes=None,
               dosplit=False, outputvis=None, doconcat=False, concatvis=None, keep_orig_ms=True,
               keep_corrected_column=False, cal_npz=None, refcal_npz_mode='smooth_model', secondary_npz=None):
    '''

    :param vis: EOVSA visibility dataset(s) to be calibrated 
    :param caltype:
    :param interp:
    :param docalib:
    :param qlookimage:
    :param flagant:
    :param stokes:
    :param doconcat:
    :return:
    '''

    interp0 = interp
    refcal_npz_mode = _normalize_refcal_npz_mode(refcal_npz_mode)

    cal_npz_refcal = None
    cal_npz_phacals = None
    cal_src = 'SQL'
    if cal_npz:
        cal_npz_refcal, cal_npz_phacals = load_calwidget_v2_npz(cal_npz)
        if secondary_npz and refcal_npz_mode == 'bph_sbd':
            _attach_secondary_bph_refcal(cal_npz_refcal, secondary_npz)
        cal_src = 'calwidget v2 NPZ'
        print('Loaded refcal + {0} phacal(s) from calwidget v2 NPZ {1}'.format(
            len(cal_npz_phacals), cal_npz))
        print('Refcal NPZ apply mode selected: {0}'.format(_refcal_npz_mode_label(refcal_npz_mode)))
        if secondary_npz and refcal_npz_mode == 'bph_sbd':
            print('Secondary BPH NPZ selected: {0}'.format(secondary_npz))

    if type(vis) == str:
        vis = [vis]

    for idx, f in enumerate(vis):
        if f[-1] == '/':
            vis[idx] = f[:-1]
        vis[idx] = str(vis[idx])

    # check if the calibration table directory is defined
    # pipeline should always use "caltbdir = /data1/eovsa/caltable/"
    if not caltbdir:
        print('Task calibeovsa')
        print('Path for generating calibration tables not defined')
        print('Use current path')
        caltbdir = './'

    failed_vis = []
    for msfile in vis:
        casalog.origin('calibeovsa')
        if not caltype:
            casalog.post("Caltype not provided. Perform reference phase calibration and daily phase calibration.")
            caltype = ['refpha', 'phacal']
        if not os.path.exists(msfile):
            casalog.post("Input visibility does not exist. Skipping...")
            failed_vis.append(msfile)
            continue
        if msfile.endswith('/'):
            msfile = msfile[:-1]
        if not msfile[-3:] in ['.ms', '.MS']:
            casalog.post("Invalid visibility. Please provide a proper visibility file ending with .ms")
        # if not caltable:
        #    caltable=[os.path.basename(vis).replace('.ms','.'+c) for c in caltype]

        try:  # --- begin per-file try block ---
            # get band information
            tb.open(msfile + '/SPECTRAL_WINDOW')
            nspw = tb.nrows()
            bdname = tb.getcol('NAME')
            bd_nchan = tb.getcol('NUM_CHAN')
            bd = [int(b[4:]) - 1 for b in bdname]
            reffreqs = tb.getcol('REF_FREQUENCY')
            bandwidths = tb.getcol('TOTAL_BANDWIDTH')
            chan_freqs_spw0 = tb.getcol('CHAN_FREQ', startrow=0, nrow=1)
            cfreq_spw0 = np.mean(chan_freqs_spw0)
            cfreqs_spw = np.asarray([
                float(np.mean(tb.getcell('CHAN_FREQ', s))) for s in range(nspw)
            ], dtype=np.float64)

            tb.close()
            tb.open(msfile + '/ANTENNA')
            nant = tb.nrows()
            antname = tb.getcol('NAME')
            antlist = [str(ll) for ll in range(len(antname) - 1)]
            antennas = ','.join(antlist)
            tb.close()

            # get time stamp, use the beginning of the file
            tb.open(msfile + '/OBSERVATION')
            trs = {'BegTime': [], 'EndTime': []}
            for ll in range(tb.nrows()):
                tim0, tim1 = Time(tb.getcell('TIME_RANGE', ll) / 24 / 3600, format='mjd')
                trs['BegTime'].append(tim0)
                trs['EndTime'].append(tim1)
            tb.close()
            trs['BegTime'] = Time(trs['BegTime'])
            trs['EndTime'] = Time(trs['EndTime'])
            btime = np.min(trs['BegTime'])
            etime = np.max(trs['EndTime'])
            # ms.open(vis)
            # summary = ms.summary()
            # ms.close()
            # btime = Time(summary['BeginTime'], format='mjd')
            # etime = Time(summary['EndTime'], format='mjd')
            ## stop using ms.summary to avoid conflicts with importeovsa
            t_mid = Time((btime.mjd + etime.mjd) / 2., format='mjd')
            print("This scan observed from {} to {} UTC".format(btime.iso, etime.iso))
            gaintables = []
            spwmaps = []

            if not antenna.strip():
                antenna = '0~12'
            if not flagant.strip():
                flagant = '13~15'
            if t_mid.mjd >= EOVSA15_UPGRADE_DATE.mjd:
                antenna = '0~14'
                flagant = '15'

            if ('refpha' in caltype) or ('refamp' in caltype) or ('refcal' in caltype):
                if cal_npz_refcal is not None:
                    refcal = cal_npz_refcal
                else:
                    refcal = sql2refcalX(btime)
                # shape is 15 (nant) x 2 (npol) x 34 (nband)
                # EOVSA15 upgrade-related Note:
                # the number of antennas in refcal['pha'] is changed to 16 after EOVSA15 upgrade
                # The last 15-ant record is  2025-05-23 and the first 16-ant record is on 2025-06-07.
                # But because the pha is added to para_pha in a loop of nant-1, it should be fine.
                # No change is needed in the code below.
                triplet = refcal.get('gencal_triplet')
                use_npz_triplets = bool(cal_npz_refcal is not None and refcal_npz_mode == 'triplet' and triplet)
                use_npz_smooth_model = bool(cal_npz_refcal is not None and refcal_npz_mode == 'smooth_model')
                use_npz_bph_sbd = bool(cal_npz_refcal is not None and refcal_npz_mode == 'bph_sbd')
                smooth_sbd = None
                if use_npz_smooth_model or use_npz_bph_sbd:
                    if refcal.get('pha_source') != 'refcal__model_pha':
                        if use_npz_smooth_model:
                            raise ValueError(
                                'smooth-model refcal mode requires refcal__model_pha in the calwidget v2 NPZ'
                            )
                    smooth_sbd = _smooth_refcal_active_sbd(refcal)
                if use_npz_smooth_model or use_npz_bph_sbd or not use_npz_triplets:
                    pha = np.asarray(refcal['pha'], dtype=np.float64).copy()
                    phase_flag = np.asarray(refcal['flag'])
                    pha[np.where(phase_flag == 1)] = 0.
                amp = refcal['amp']
                amp[np.where(refcal['flag'] == 1)] = 1.
                t_ref = refcal['timestamp']
                # find the start and end time of the local day when refcal is registered
                try:
                    dhr = t_ref.LocalTime.utcoffset().total_seconds() / 60. / 60.
                except:
                    dhr = -7.
                bt = Time(np.fix(t_ref.mjd + dhr / 24.) - dhr / 24., format='mjd')
                et = Time(bt.mjd + 1., format='mjd')
                (yr, mon, day) = (bt.datetime.year, bt.datetime.month, bt.datetime.day)
                dirname = caltbdir + str(yr) + str(mon).zfill(2) + '/'
                if not os.path.exists(dirname):
                    os.mkdir(dirname)
                # check if there is any ROACH reboot between the reference calibration found and the current data
                t_rbts = db.get_reboot(Time([t_ref, btime]))
                if not t_rbts:
                    casalog.post("Reference calibration is derived from observation at " + t_ref.iso + f" [source: {cal_src}]")
                    print("Reference calibration is derived from observation at " + t_ref.iso + f" [source: {cal_src}]")
                else:
                    casalog.post(
                        "Oh crap! Roach reboot detected between the reference calibration time " + t_ref.iso + ' and the current observation at ' + btime.iso)
                    casalog.post("Aborting...")
                    print(
                        "Oh crap! Roach reboot detected between the reference calibration time " + t_ref.iso + ' and the current observation at ' + btime.iso)
                    print("Aborting...")

                para_pha = []
                para_amp = []
                para_sbd = []
                para_mbd = []
                calpha = np.zeros((nspw, nant - 1, 2))
                calamp = np.zeros((nspw, nant - 1, 2))
                mbd_ref_ghz = float(cfreq_spw0) * 1e-9
                phase_flag_spw = np.ones((nant - 1, 2, nspw), dtype=np.int32)
                selected_npz_models = set()
                band_phase = np.asarray(refcal.get('band_phase_rad', []), dtype=np.float64)
                band_phase_flag = np.asarray(refcal.get('band_phase_flag', []), dtype=np.int32)
                # NaN-preserving smooth model for the bph_sbd LO HI-extrapolation
                # fallback; carries HI smooth values at LO band centers (finite)
                # and NaN where the widget produced no model.
                model_pha_lo_fallback = np.asarray(refcal.get('model_pha_raw', []), dtype=np.float64)

                def phase_is_flagged(ant_i, pol_i, band_i):
                    return (
                        phase_flag.ndim == 3
                        and ant_i < phase_flag.shape[0]
                        and pol_i < phase_flag.shape[1]
                        and band_i < phase_flag.shape[2]
                        and phase_flag[ant_i, pol_i, band_i] == 1
                    )

                for s in range(nspw):
                    band_i = int(bd[s])
                    band_triplet_name = None
                    band_triplet = None
                    if use_npz_triplets:
                        band_triplet_name, band_triplet = _npz_triplet_for_band(refcal, band_i)
                        if band_triplet_name:
                            selected_npz_models.add(band_triplet_name)
                    for n in range(nant - 1):
                        for p in range(2):
                            if band_triplet:
                                phase_rad = _triplet_value(
                                    band_triplet, 'phi_band_rad', n, p, band_i, default=np.nan
                                )
                                ib_ns = 0.0
                                mb_ns = 0.0
                                band_ref = _triplet_value(
                                    band_triplet, 'band_ref_freq_ghz', n, p, band_i, default=np.nan
                                )
                                ib_ns = _triplet_value(
                                    band_triplet, 'tau_ib_ns', n, p, band_i, default=np.nan
                                )
                                mb_ns = _triplet_value(
                                    band_triplet, 'tau_mb_eff_ns', n, p, band_i, default=np.nan
                                )
                                flagged = (
                                    _triplet_flagged(band_triplet, n, p, band_i)
                                    or _operator_flagged(refcal, n, p, band_i)
                                    or not np.isfinite(phase_rad)
                                    or not np.isfinite(ib_ns)
                                    or not np.isfinite(mb_ns)
                                )
                                phase_flag_spw[n, p, s] = 1 if flagged else 0
                                if flagged:
                                    phase_rad = 0.0
                                    ib_ns = 0.0
                                    mb_ns = 0.0
                                if not np.isfinite(band_ref):
                                    band_ref = float(cfreqs_spw[s]) * 1e-9
                                # The three CASA tables combine to the benchmark model:
                                # phi + 2*pi*(freq-band_ref)*tau_ib + 2*pi*freq*tau_mb.
                                phase_rad = (
                                    float(phase_rad)
                                    + 2.0 * np.pi * (float(cfreqs_spw[s]) * 1e-9 - float(band_ref)) * float(ib_ns)
                                    + 2.0 * np.pi * mbd_ref_ghz * float(mb_ns)
                                )
                                para_sbd.append(float(ib_ns))
                                para_mbd.append(float(mb_ns))
                            elif use_npz_triplets:
                                phase_rad = 0.0
                                phase_flag_spw[n, p, s] = 1
                                para_sbd.append(0.0)
                                para_mbd.append(0.0)
                            elif use_npz_bph_sbd:
                                hi_sbd_ns = _smooth_refcal_sbd_for_band(refcal, smooth_sbd, n, p, band_i)
                                phase_rad, sbd_ns, flagged, source = _bph_sbd_phase_base_for_band(
                                    refcal,
                                    band_phase,
                                    band_phase_flag,
                                    model_pha_lo_fallback,
                                    n,
                                    p,
                                    band_i,
                                    float(cfreqs_spw[s]) * 1e-9,
                                    hi_sbd_ns,
                                )
                                if not np.isfinite(sbd_ns):
                                    flagged = True
                                    phase_rad = 0.0
                                phase_flag_spw[n, p, s] = 1 if flagged else 0
                                if flagged or not np.isfinite(phase_rad):
                                    phase_rad = 0.0
                                para_sbd.append(0.0 if flagged else float(sbd_ns))
                                selected_npz_models.add(source)
                            else:
                                phase_rad = pha[n, p, band_i]
                                flagged = phase_is_flagged(n, p, band_i)
                                if use_npz_smooth_model:
                                    flagged = flagged or _operator_flagged(refcal, n, p, band_i)
                                phase_flag_spw[n, p, s] = 1 if flagged else 0
                                if flagged or not np.isfinite(phase_rad):
                                    phase_rad = 0.0
                                if use_npz_smooth_model:
                                    sbd_ns = _smooth_refcal_sbd_for_band(refcal, smooth_sbd, n, p, band_i)
                                    if not np.isfinite(sbd_ns):
                                        if flagged:
                                            sbd_ns = 0.0
                                        else:
                                            raise ValueError(
                                                "smooth-model refcal mode cannot obtain a finite SBD value "
                                                "for antenna {0:d} pol {1:d}".format(n + 1, p)
                                            )
                                    para_sbd.append(0.0 if flagged else float(sbd_ns))
                            calpha[s, n, p] = phase_rad
                            calamp[s, n, p] = amp[n, p, band_i]
                            para_pha.append(np.degrees(phase_rad))
                            para_amp.append(amp[n, p, band_i])
                if use_npz_triplets:
                    print("NPZ refcal model namespaces selected by SPW: {0}".format(
                        ",".join(sorted(selected_npz_models)) if selected_npz_models else "none"
                    ))
                if use_npz_bph_sbd:
                    print("BPH+SBD refcal sources selected by SPW: {0}; SBD source={1}".format(
                        ",".join(sorted(selected_npz_models)) if selected_npz_models else "none",
                        refcal.get('active_ns_source', 'unknown'),
                    ))
                if use_npz_smooth_model:
                    print("Smooth-model refcal mode selected; phase source={0}; SBD source={1}".format(
                        refcal.get('pha_source', 'unknown'),
                        refcal.get('active_ns_source', 'unknown'),
                    ))

            if 'fluxcal' in caltype:
                calfac = pc.get_calfac(Time(t_mid.iso.split(' ')[0] + 'T23:59:59'))
                t_bp = Time(calfac['timestamp'], format='lv')
                if int(t_mid.mjd) == int(t_bp.mjd):
                    accalfac = calfac['accalfac']  # (ant x pol x freq)
                    # tpcalfac = calfac['tpcalfac']  # (ant x pol x freq)
                    caltb_autoamp = dirname + t_bp.isot[:-4].replace(':', '').replace('-', '') + '.bandpass'
                    if not os.path.exists(caltb_autoamp):
                        bandpass(vis=msfile, caltable=caltb_autoamp, solint='inf', refant='eo01', minblperant=0, minsnr=0,
                                 bandtype='B', docallib=False)
                        tb.open(caltb_autoamp, nomodify=False)  # (ant x spw)
                        bd_chanidx = np.hstack([[0], bd_nchan.cumsum()])
                        for ll in range(nspw):
                            antfac = np.sqrt(accalfac[:, :, bd_chanidx[ll]:bd_chanidx[ll + 1]])
                            # # antfac *= tpcalfac[:, :,bd_chanidx[ll]:bd_chanidx[ll + 1]]
                            antfac = np.moveaxis(antfac, 0, 2)
                            cparam = np.zeros((2, bd_nchan[ll], nant))
                            cparam[:, :, :-3] = 1.0 / antfac
                            tb.putcol('CPARAM', cparam + 0j, ll * nant, nant)
                            paramerr = tb.getcol('PARAMERR', ll * nant, nant)
                            paramerr = paramerr * 0
                            tb.putcol('PARAMERR', paramerr, ll * nant, nant)
                            bpflag = tb.getcol('FLAG', ll * nant, nant)
                            bpant1 = tb.getcol('ANTENNA1', ll * nant, nant)
                            bpflagidx, = np.where(bpant1 >= 13)
                            bpflag[:] = False
                            bpflag[:, :, bpflagidx] = True
                            tb.putcol('FLAG', bpflag, ll * nant, nant)
                            bpsnr = tb.getcol('SNR', ll * nant, nant)
                            bpsnr[:] = 100.0
                            bpsnr[:, :, bpflagidx] = 0.0
                            tb.putcol('SNR', bpsnr, ll * nant, nant)
                        tb.close()
                        msg_prompt = "Scaling calibration is derived for {}.".format(msfile)
                        casalog.post(msg_prompt)
                        print(msg_prompt)
                    gaintables.append(caltb_autoamp)
                    spwmaps.append([])
                else:
                    msg_prompt = "Caution: No TPCAL is available on {}. No scaling calibration is derived for {}.".format(
                        t_mid.datetime.strftime('%b %d, %Y'), msfile)
                    casalog.post(msg_prompt)
                    print(msg_prompt)

            if ('refpha' in caltype) or ('refcal' in caltype):
                # caltb_pha = os.path.basename(vis).replace('.ms', '.refpha')
                # check if the calibration table already exists
                if use_npz_triplets:
                    refcal_npz_suffix = '_npz_triplet'
                elif use_npz_bph_sbd:
                    refcal_npz_suffix = '_npz_bph_sbd'
                elif use_npz_smooth_model:
                    refcal_npz_suffix = '_npz_smooth_model'
                else:
                    refcal_npz_suffix = ''
                caltb_pha = dirname + t_ref.isot[:-4].replace(':', '').replace('-', '') + refcal_npz_suffix + '.refpha'
                if (use_npz_triplets or use_npz_smooth_model or use_npz_bph_sbd) and os.path.exists(caltb_pha):
                    shutil.rmtree(caltb_pha)
                if not os.path.exists(caltb_pha):
                    gencal(vis=msfile, caltable=caltb_pha, caltype='ph', antenna=antennas, pol='X,Y',
                           spw='0~' + str(nspw - 1), parameter=para_pha)
                    tb.open(caltb_pha, nomodify=False)
                    phaflag_ = phase_flag_spw
                    phaflag_new = np.full((nant, 2, nspw), True, dtype=np.bool_)
                    copy_nant = min(nant, phaflag_.shape[0])
                    if t_mid.mjd >= EOVSA15_UPGRADE_DATE.mjd:
                        phaflag_new[:copy_nant, ...] = phaflag_[:copy_nant, ...]
                    else:
                        copy_nant = min(nant - 1, phaflag_.shape[0])
                        phaflag_new[:copy_nant, ...] = phaflag_[:copy_nant, ...]
                    phaflag_new = np.moveaxis(phaflag_new, 0, 2).reshape(2, 1, nant * nspw)
                    tb.putcol('FLAG', phaflag_new)
                    tb.close()

                    # tb.open(caltb_pha, nomodify=False)
                    # phaparam = np.angle(tb.getcol('CPARAM'),deg=True)
                    # phaparam_ = np.degrees(refcal['pha'][:,:,np.array(bd)])
                    # phaparam2 = np.zeros((nant, 2, nspw))
                    # phaparam2[:-1,...] = phaparam_
                    # # phaparam2 = phaparam2.swapaxes(0,1).reshape(2,1,nant*nspw)
                    # phaparam2 = np.moveaxis(phaparam2,0,2).reshape(2,1,nant*nspw)
                    # tb.close()

                refcal_gaintables = [caltb_pha]
                gaintables.append(caltb_pha)
                spwmaps.append([])
                if use_npz_triplets or use_npz_smooth_model or use_npz_bph_sbd:
                    delay_table_kinds = [('sbd', para_sbd, 'sbd', True)]
                    if use_npz_triplets:
                        delay_table_kinds.append(('mbd', para_mbd, 'mbd', False))
                    for table_kind, params, caltype_name, per_spw in delay_table_kinds:
                        caltb_delay = dirname + t_ref.isot[:-4].replace(':', '').replace('-', '') + refcal_npz_suffix + '.ref' + table_kind
                        if os.path.exists(caltb_delay):
                            shutil.rmtree(caltb_delay)
                        if per_spw:
                            nparam_spw = (nant - 1) * 2
                            for s in range(nspw):
                                start = s * nparam_spw
                                gencal(vis=msfile, caltable=caltb_delay, caltype=caltype_name, antenna=antennas, pol='X,Y',
                                       spw=str(s), parameter=params[start:start + nparam_spw])
                        else:
                            gencal(vis=msfile, caltable=caltb_delay, caltype=caltype_name, antenna=antennas, pol='X,Y',
                                   spw='0~' + str(nspw - 1), parameter=params)
                        tb.open(caltb_delay, nomodify=False)
                        delayflag_new = np.full((nant, 2, nspw), True, dtype=np.bool_)
                        copy_nant = min(nant, phaflag_.shape[0])
                        if t_mid.mjd >= EOVSA15_UPGRADE_DATE.mjd:
                            delayflag_new[:copy_nant, ...] = phaflag_[:copy_nant, ...]
                        else:
                            copy_nant = min(nant - 1, phaflag_.shape[0])
                            delayflag_new[:copy_nant, ...] = phaflag_[:copy_nant, ...]
                        delayflag_new = np.moveaxis(delayflag_new, 0, 2).reshape(2, 1, nant * nspw)
                        tb.putcol('FLAG', delayflag_new)
                        tb.close()
                        gaintables.append(caltb_delay)
                        refcal_gaintables.append(caltb_delay)
                        spwmaps.append([])
                    print("Refcal NPZ gaintables ({0}): {1}".format(
                        refcal_npz_mode,
                        ", ".join(os.path.basename(path) for path in refcal_gaintables),
                    ))
            if ('refamp' in caltype) or ('refcal' in caltype):
                # caltb_amp = os.path.basename(vis).replace('.ms', '.refamp')
                caltb_amp = dirname + t_ref.isot[:-4].replace(':', '').replace('-', '') + '.refamp'
                if not os.path.exists(caltb_amp):
                    gencal(vis=msfile, caltable=caltb_amp, caltype='amp', antenna=antennas, pol='X,Y',
                           spw='0~' + str(nspw - 1), parameter=para_amp)
                    tb.open(caltb_amp, nomodify=False)
                    ampflag_ = np.asarray(refcal['flag'])[:, :, np.array(bd)]
                    operator_flag = np.asarray(refcal.get('operator_band_flag', []), dtype=np.uint8)
                    if operator_flag.ndim == 3:
                        op = operator_flag[:, :, np.array(bd)]
                        copy_nant = min(ampflag_.shape[0], op.shape[0])
                        copy_npol = min(ampflag_.shape[1], op.shape[1])
                        copy_nspw = min(ampflag_.shape[2], op.shape[2])
                        if copy_nant and copy_npol and copy_nspw:
                            ampflag_[:copy_nant, :copy_npol, :copy_nspw] = np.where(
                                op[:copy_nant, :copy_npol, :copy_nspw] != 0,
                                1,
                                ampflag_[:copy_nant, :copy_npol, :copy_nspw],
                            )
                    ampflag_new = np.full((nant, 2, nspw), True, dtype=np.bool_)
                    if t_mid.mjd >= EOVSA15_UPGRADE_DATE.mjd:
                        ampflag_new[...] = ampflag_
                    else:
                        ampflag_new[:-1, ...] = ampflag_
                    ampflag_new = np.moveaxis(ampflag_new, 0, 2).reshape(2, 1, nant * nspw)
                    tb.putcol('FLAG', ampflag_new)
                    tb.close()
                gaintables.append(caltb_amp)
                spwmaps.append([])

            # calibration for the change of delay center between refcal time and beginning of scan -- hopefully none!
            xml, buf = ch.read_calX(4, t=[t_ref, btime], verbose=False)
            if buf is not None:
                dly_t2 = Time(eoextract(buf[0], xml['Timestamp']), format='lv')
                dlycen_ns2 = eoextract(buf[0], xml['Delaycen_ns'])[:nant - 1]
                xml, buf = ch.read_calX(4, t=t_ref)
                dly_t1 = Time(eoextract(buf, xml['Timestamp']), format='lv')
                dlycen_ns1 = eoextract(buf, xml['Delaycen_ns'])[:nant - 1]
                dlycen_ns_diff = dlycen_ns2 - dlycen_ns1
                for n in range(2):
                    dlycen_ns_diff[:, n] -= dlycen_ns_diff[0, n]
                print('Multi-band delay is derived from delay center difference at {} & {} [source: SQL DCM]'.format(dly_t1.iso, dly_t2.iso))
                dlycen_pha0 = np.degrees(dlycen_ns_diff * 1e-9 * cfreq_spw0 * 2. * np.pi)
                # print('=====Delays relative to Ant 14=====')
                # for i, dl in enumerate(dlacen_ns_diff[:, 0] - dlacen_ns_diff[13, 0]):
                #     ant = antlist[i]
                #     print 'Ant eo{0:02d}: x {1:.2f} ns & y {2:.2f} ns'.format(int(ant) + 1, dl
                #           dlacen_ns_diff[i, 1] - dlacen_ns_diff[13, 1])
                # caltb_mbd0 = os.path.basename(vis).replace('.ms', '.mbd0')
                caltb_dlycen = dirname + dly_t2.isot[:-4].replace(':', '').replace('-', '') + '.dlycen'
                caltb_dlycen_pha0 = dirname + dly_t2.isot[:-4].replace(':', '').replace('-', '') + '.dlycen_pha0'
                if not os.path.exists(caltb_dlycen):
                    gencal(vis=msfile, caltable=caltb_dlycen, caltype='mbd', pol='X,Y', antenna=antennas,
                           parameter=dlycen_ns_diff.flatten().tolist())
                if not os.path.exists(caltb_dlycen_pha0):
                    gencal(vis=msfile, caltable=caltb_dlycen_pha0, caltype='ph', pol='X,Y', antenna=antennas,
                           parameter=dlycen_pha0.flatten().tolist())
                gaintables.append(caltb_dlycen)
                spwmaps.append(nspw * [0])
                gaintables.append(caltb_dlycen_pha0)
                spwmaps.append(nspw * [0])

            if 'phacal' in caltype:
                if cal_npz_phacals is not None:
                    phacals = np.array(cal_npz_phacals)
                else:
                    phacals = np.array(sql2phacalX([bt, et], nrecords=0, neat=True, verbose=False))
                if not phacals.any() or len(phacals) == 0:
                    print(f"Found no phacal records in {cal_src}, will skip phase calibration")
                else:
                    # first generate all phacal calibration tables if not already exist
                    t_phas = Time([phacal['t_pha'] for phacal in phacals])
                    # sort the array in ascending order by t_pha
                    sinds = t_phas.mjd.argsort()
                    t_phas = t_phas[sinds]
                    phacals = phacals[sinds]
                    caltbs_phambd = []
                    caltbs_phambd_pha0 = []
                    for i, phacal in enumerate(phacals):
                        # filter out phase cals with reference time stamp >30 min away from the provided refcal time
                        if (phacal['t_ref'].jd - refcal['timestamp'].jd) > 30. / 1440.:
                            del phacals[i]
                            del t_phas[i]
                            continue
                        else:
                            t_pha = phacal['t_pha']
                            phambd_ns = phacal['pslope']
                            for n in range(2):
                                phambd_ns[:, n] -= phambd_ns[0, n]
                            # set all flagged values to be zero
                            phambd_ns[np.where(phacal['flag'] == 1)] = 0.
                            caltb_phambd = dirname + t_pha.isot[:-4].replace(':', '').replace('-', '') + '.phambd'
                            caltbs_phambd.append(caltb_phambd)
                            if os.path.exists(caltb_phambd):
                                os.system('rm -rf ' + caltb_phambd)
                            # if not os.path.exists(caltb_phambd):
                            gencal(vis=msfile, caltable=caltb_phambd, caltype='mbd', pol='X,Y', antenna=antennas,
                                   parameter=phambd_ns[:nant-1,:].flatten().tolist())
                            if flagspw != '':
                                flag_phambd_by_spw(caltb_phambd, flagspw=flagspw)

                            # When applying the multi-band delays, they are referenced to the center of spw 0
                            # Make a corresponding calibration table for the reference phase at the center of spw 0
                            pha0 = np.degrees(phambd_ns * 1e-9 * cfreq_spw0 * 2. * np.pi)
                            caltb_phambd_pha0 = dirname + t_pha.isot[:-4].replace(':', '').replace('-', '') + '.phambd_pha0'
                            caltbs_phambd_pha0.append(caltb_phambd_pha0)
                            if os.path.exists(caltb_phambd_pha0):
                                os.system('rm -rf ' + caltb_phambd_pha0)
                            # if not os.path.exists(caltb_phambd_pha0):
                            gencal(vis=msfile, caltable=caltb_phambd_pha0, caltype='ph', pol='X,Y', antenna=antennas,
                                   parameter=pha0[:nant-1,:].flatten().tolist())
                            if flagspw != '':
                                flag_phambd_by_spw(caltb_phambd_pha0, flagspw=flagspw)

                    # now decides which table to apply depending on the interpolation method ("nearest" or "linear")
                    dt = np.min(np.abs(t_phas.mjd - t_mid.mjd)) * 24.
                    if interp0 == 'auto':
                        print(
                            f'interp method is set to auto. The interpolation method will be determined based on the time difference between the mid time of the scan and the nearest phase calibration table.')
                        print(f'The time difference threshold is set to 1 hour')
                        if dt < 1.:
                            interp = 'nearest'
                            print(f'The time difference is {dt:.1f} hours. Using nearest interp method.')
                        else:
                            interp = 'linear'
                            print(f'The time difference is {dt:.1f} hours. Using linear interp method.')
                    if interp == 'nearest':
                        tbind = np.argmin(np.abs(t_phas.mjd - t_mid.mjd))
                        print("Selected nearest phase calibration table at " + t_phas[tbind].iso + f" [source: {cal_src}]")
                        gaintables.append(caltbs_phambd[tbind])
                        ## Note: gencal generates the same solution for all spws, so no need to specify spwmap
                        # spwmaps.append(nspw * [0])
                        spwmaps.append([])
                        gaintables.append(caltbs_phambd_pha0[tbind])
                        # spwmaps.append(nspw * [0])
                        spwmaps.append([])
                    if interp == 'linear':
                        # bphacal = sql2phacalX(btime)
                        # ephacal = sql2phacalX(etime,reverse=True)
                        bt_ind, = np.where(t_phas.mjd < btime.mjd)
                        et_ind, = np.where(t_phas.mjd > etime.mjd)
                        if len(bt_ind) == 0 and len(et_ind) == 0:
                            print("No phacal found before or after the ms data within the day of observation")
                            print("Skipping daily phase calibration")
                        elif len(bt_ind) > 0 and len(et_ind) == 0:
                            gaintables.append(caltbs_phambd[bt_ind[-1]])
                            # spwmaps.append(nspw * [0])
                            spwmaps.append([])
                            gaintables.append(caltbs_phambd_pha0[bt_ind[-1]])
                            # spwmaps.append(nspw * [0])
                            spwmaps.append([])
                            print("Using phase calibration table at " + t_phas[bt_ind[-1]].iso + f" [source: {cal_src}]")
                        elif len(bt_ind) == 0 and len(et_ind) > 0:
                            gaintables.append(caltbs_phambd[et_ind[0]])
                            # spwmaps.append(nspw * [0])
                            spwmaps.append([])
                            gaintables.append(caltbs_phambd_pha0[et_ind[0]])
                            # spwmaps.append(nspw * [0])
                            spwmaps.append([])
                            print("Using phase calibration table at " + t_phas[et_ind[0]].iso + f" [source: {cal_src}]")
                        elif len(bt_ind) > 0 and len(et_ind) > 0:
                            bphacal = phacals[bt_ind[-1]]
                            ephacal = phacals[et_ind[0]]
                            # generate a new table interpolating between two daily phase calibrations
                            dt_obs = t_mid.mjd - bphacal['t_pha'].mjd
                            dt_pha = ephacal['t_pha'].mjd - bphacal['t_pha'].mjd
                            phambd_diff = ephacal['pslope'] - bphacal['pslope']
                            phambd_ns = bphacal['pslope'] + dt_obs / dt_pha * phambd_diff
                            for n in range(2):
                                phambd_ns[:, n] -= phambd_ns[0, n]
                            # set all flagged values to be zero
                            phambd_ns[np.where(bphacal['flag'] == 1)] = 0.
                            phambd_ns[np.where(ephacal['flag'] == 1)] = 0.
                            caltb_phambd_interp = dirname + t_mid.isot[:-4].replace(':', '').replace('-',
                                                                                                     '') + '.phambd'
                            caltb_phambd_interp_pha0 = caltb_phambd_interp + '_pha0'
                            pha0 = np.degrees(phambd_ns * 1e-9 * cfreq_spw0 * 2. * np.pi)
                            if os.path.exists(caltb_phambd_interp):
                                os.system('rm -rf ' + caltb_phambd_interp)
                            # if not os.path.exists(caltb_phambd_interp):
                            gencal(vis=msfile, caltable=caltb_phambd_interp, caltype='mbd', pol='X,Y', antenna=antennas,
                                   parameter=phambd_ns.flatten().tolist())
                            if flagspw != '':
                                flag_phambd_by_spw(caltb_phambd_interp, flagspw=flagspw)
                            if os.path.exists(caltb_phambd_interp_pha0):
                                os.system('rm -rf ' + caltb_phambd_interp_pha0)
                            # if not os.path.exists(caltb_phambd_interp_pha0):
                            gencal(vis=msfile, caltable=caltb_phambd_interp_pha0, caltype='ph', pol='X,Y',
                                   antenna=antennas, parameter=pha0.flatten().tolist())
                            if flagspw != '':
                                flag_phambd_by_spw(caltb_phambd_interp_pha0, flagspw=flagspw)
                            print("Using phase calibration table interpolated between records at " + bphacal[
                                't_pha'].iso + ' and ' + ephacal['t_pha'].iso + f" [source: {cal_src}]")
                            gaintables.append(caltb_phambd_interp)
                            # spwmaps.append(nspw * [0])
                            spwmaps.append([])
                            gaintables.append(caltb_phambd_interp_pha0)
                            # spwmaps.append(nspw * [0])
                            spwmaps.append([])

            if docalib:
                clearcal(msfile)
                applycal(vis=msfile, gaintable=gaintables, spwmap=spwmaps, applymode='calflag', calwt=False)
            if doflag:
                # flag zeros and NaNs
                flagdata(vis=msfile, mode='clip', clipzeros=True)
                if flagant:
                    try:
                        flagdata(vis=msfile, antenna=flagant)
                    except:
                        print("Something wrong with flagant. Abort...")

            if doimage:
                from matplotlib import pyplot as plt
                from suncasa.utils import helioimage2fits as hf
                from sunpy import map as smap

                if not stokes:
                    stokes = 'XX'
                if not timerange:
                    timerange = ''
                if not spw:
                    spw = '1~3'
                if not imagedir:
                    imagedir = '.'
                # (yr, mon, day) = (bt.datetime.year, bt.datetime.month, bt.datetime.day)
                # dirname = imagedir + str(yr) + '/' + str(mon).zfill(2) + '/' + str(day).zfill(2) + '/'
                # if not os.path.exists(dirname):
                #    os.makedirs(dirname)
                bds = [spw]
                nbd = len(bds)
                imgs = []
                for bd in bds:
                    if '~' in bd:
                        bdstr = bd.replace('~', '-')
                    else:
                        bdstr = str(bd).zfill(2)
                    imname = imagedir + '/' + os.path.basename(msfile).replace('.ms', '.bd' + bdstr)
                    print('Cleaning image: ' + imname)
                    try:
                        tclean(vis=msfile, imagename=imname, antenna=antenna, spw=bd, timerange=timerange, imsize=[512],
                               cell=['5.0arcsec'], stokes=stokes,
                               niter=500)
                    except:
                        print('clean not successfull for band ' + str(bd))
                    else:
                        imgs.append(imname + '.image')
                    junks = ['.flux', '.mask', '.model', '.psf', '.residual']
                    for junk in junks:
                        if os.path.exists(imname + junk):
                            shutil.rmtree(imname + junk)

                tranges = [btime.iso + '~' + etime.iso] * nbd
                fitsfiles = [img.replace('.image', '.fits') for img in imgs]
                hf.imreg(vis=msfile, timerange=tranges, imagefile=imgs, fitsfile=fitsfiles, usephacenter=False)
                plt.figure(figsize=(6, 6))
                for i, fitsfile in enumerate(fitsfiles):
                    plt.subplot(1, nbd, i + 1)
                    eomap = smap.Map(fitsfile)
                    sz = eomap.data.shape
                    if len(sz) == 4:
                        eomap.data = eomap.data.reshape((sz[2], sz[3]))
                    eomap.plot_settings['cmap'] = plt.get_cmap('jet')
                    eomap.plot()
                    eomap.draw_limb()
                    # the next line would cause trouble in higher versions of SunPy, as it requires WCS
                    # eomap.draw_grid()

                plt.show()

        except Exception as e:
            import traceback as _tb
            casalog.post(f"ERROR processing {msfile}: {e}. Skipping this file.")
            print(f"ERROR processing {msfile}: {e}")
            print(_tb.format_exc())
            failed_vis.append(msfile)
            continue

    # Remove failed files from vis before dosplit/doconcat
    for fv in failed_vis:
        if fv in vis:
            vis.remove(fv)

    if not vis:
        casalog.post("All input files failed calibration. No output produced.")
        print("All input files failed calibration. No output produced.")
        return None

    if dosplit:
        if not doconcat:
            if not outputvis:
                outputvis = [vis[n].split('.')[0] + '.corrected.ms' for n in range(len(vis))]
            for n in range(len(vis)):
                split(vis=vis[n], outputvis=outputvis[n], datacolumn='corrected')
                if not keep_orig_ms:
                    os.system('rm -rf {}'.format(vis[n]))
    else:
        outputvis = vis

    if doconcat:
        if not concatvis:
            msoutdir = os.path.dirname(vis[0])
            if len(vis) == 1:
                vis0 = os.path.basename(vis[0])
                concatvis = os.path.join(msoutdir, vis0.split('.')[0] + '.corrected.ms')
            if len(vis) > 1:
                visb = os.path.basename(vis[0])
                vise = os.path.basename(vis[-1])
                concatvis = os.path.join(msoutdir, visb.split('.')[0] + '-' + vise.split('.')[0][3:] + '.corrected.ms')
        if len(vis) == 1:
            split(vis=vis[0], outputvis=concatvis, datacolumn='corrected')
        if len(vis) > 1:
            cols2rm = "model" if keep_corrected_column else "model,corrected"
            concateovsa(vis, concatvis, datacolumn='corrected', keep_orig_ms=keep_orig_ms, cols2rm=cols2rm)
        return concatvis
    else:
        return outputvis
