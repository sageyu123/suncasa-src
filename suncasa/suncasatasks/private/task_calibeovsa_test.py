from ...casa_compat import check_dependencies

check_dependencies()

import platform
import matplotlib

if platform.system() == 'Linux':
    matplotlib.use('Agg')
import os
import shutil
import numpy as np

from eovsapy.util import extract as eoextract
from eovsapy.util import Time
from eovsapy import cal_header as ch
from eovsapy import dbutil as db
from eovsapy import pipeline_cal as pc
from eovsapy.sqlutil import sql2refcalX, sql2phacalX, sql2refcal_bphsbdX
from .. import concateovsa
from suncasa.eovsa.update_log import EOVSA15_UPGRADE_DATE, DCM_IF_FILTER_UPGRADE_DATE

from ...casa_compat import import_casatools, import_casatasks

from eovsapy.calibeovsa_bph_sbd import (
    _normalize_refcal_npz_mode,
    _refcal_npz_mode_label,
    _npz_json_scalar,
    _merge_npz_promoted_anchor_arrays,
    _promoted_source_antennas_for_phacal,
    _apply_promoted_source_phacal_self_anchor,
    _load_npz_triplet,
    _triplet_has_band,
    _triplet_value,
    _triplet_flagged,
    _triplet_slot_usable,
    _operator_flagged,
    _array_value_3d,
    _array_flagged_3d,
    _lo_data_ignored_for_ant,
    _npz_triplet_for_band,
    _apply_promoted_active_sbd,
    _smooth_refcal_active_sbd,
    _smooth_refcal_sbd_for_band,
    _gencal_sbd_phase_at_spw_reference,
    _triplet_bph_sbd_terms_for_band,
    _bph_candidate_for_band,
    _bph_sbd_phase_base_for_band,
    sample_refcal_bps_for_band,
    load_calwidget_v2_npz,
    _attach_secondary_bph_refcal,
    resolve_bph_sbd_tables,
    recenter_bph_rad,
    REFCAL_NPZ_MODES,
)

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


def _valid_sql_bphsbd_arrays(record):
    if not isinstance(record, dict):
        return None, "record missing"
    missing = [key for key in ("bph_rad", "sbd_ns", "flag") if key not in record]
    if missing:
        return None, "missing {0}".format(",".join(missing))
    try:
        bph = np.asarray(record["bph_rad"], dtype=np.float64)
        sbd = np.asarray(record["sbd_ns"], dtype=np.float64)
        flag = np.asarray(record["flag"], dtype=np.float64)
    except Exception as exc:
        return None, "array conversion failed: {0}".format(exc)
    if bph.ndim != 3 or sbd.ndim != 3 or flag.ndim != 3:
        return None, "expected 3-D bph/sbd/flag arrays"
    if bph.shape != sbd.shape or bph.shape != flag.shape:
        return None, "shape mismatch bph={0} sbd={1} flag={2}".format(bph.shape, sbd.shape, flag.shape)
    usable = np.isfinite(bph) & np.isfinite(sbd) & (flag == 0)
    if not np.any(usable):
        return None, "no finite unflagged BPH+SBD slots"
    return (bph, sbd, flag), None


def _attach_sql_bphsbd_refcal(refcal, record, arrays):
    bph, sbd, flag = arrays
    if not isinstance(refcal, dict):
        refcal = {}
    refcal["resolved_bph_rad"] = bph
    refcal["resolved_sbd_ns"] = sbd
    refcal["resolved_flag"] = flag
    refcal["sql_bphsbd_t_refcal"] = record.get("t_refcal")
    if "timestamp" not in refcal:
        refcal["timestamp"] = record.get("t_refcal") or record.get("timestamp")
    if "t_bg" not in refcal:
        refcal["t_bg"] = record.get("t_bg") or refcal.get("timestamp")
    if "t_ed" not in refcal:
        refcal["t_ed"] = record.get("t_ed") or refcal.get("timestamp")
    if "pha" not in refcal:
        refcal["pha"] = bph.copy()
    if "amp" not in refcal:
        refcal["amp"] = np.ones(bph.shape, dtype=np.float64)
    if "flag" not in refcal:
        refcal["flag"] = flag.copy()
    return refcal


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


def _bps_candidate_code_for_source(source):
    """Return the BPS candidate code for a resolved BPH source."""

    if source == 'band_phase':
        return 1
    if source == 'secondary_band_phase':
        return 2
    return 0


def _refcal_bps_payload_for_spw(
        refcal, candidate_code, phase_flag, band_id, target_freq_ghz, nant):
    """Build one SPW's phase-only residual BPS payload.

    :param refcal: Primary NPZ refcal dictionary.
    :type refcal: dict
    :param candidate_code: Per-antenna/polarization source codes; 1 selects
        primary measured BPH, 2 selects secondary measured BPH, and 0 is an
        identity tier.
    :type candidate_code: array-like
    :param phase_flag: Per-antenna/polarization BPH invalid flags.
    :type phase_flag: array-like
    :param band_id: One-based EOVSA band identifier.
    :type band_id: int
    :param target_freq_ghz: Science channel frequencies in GHz.
    :type target_freq_ghz: array-like
    :param nant: Number of antenna rows in the CASA table.
    :type nant: int
    :returns: ``(cparam, flag, applied_slots, measured_slots)`` with arrays
        shaped ``(nant, 2, nchan)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, int, int]
    """

    target = np.asarray(target_freq_ghz, dtype=np.float64).reshape(-1)
    cparam = np.ones((int(nant), 2, target.size), dtype=np.complex128)
    flag = np.zeros((int(nant), 2, target.size), dtype=np.bool_)
    codes = np.asarray(candidate_code)
    invalid = np.asarray(phase_flag)
    if (
        codes.ndim != 2
        or invalid.ndim != 2
        or codes.shape[1] < 2
        or invalid.shape[1] < 2
    ):
        flag[:] = True
        return cparam, flag, 0, 0

    covered_nant = min(int(nant), codes.shape[0], invalid.shape[0])
    secondary = refcal.get('secondary_bph_refcal')
    applied_slots = 0
    measured_slots = 0
    for ant_i in range(covered_nant):
        for pol_i in range(2):
            if invalid[ant_i, pol_i] != 0:
                flag[ant_i, pol_i, :] = True
                continue
            code = int(codes[ant_i, pol_i])
            if code not in (1, 2):
                continue
            measured_slots += 1
            candidate = refcal if code == 1 else (
                secondary if isinstance(secondary, dict) else {}
            )
            bps_phase, applied = sample_refcal_bps_for_band(
                candidate,
                ant_i,
                pol_i,
                band_id,
                target,
            )
            if applied:
                cparam[ant_i, pol_i, :] = np.exp(1j * bps_phase)
                applied_slots += 1
    flag[covered_nant:, :, :] = True
    return cparam, flag, applied_slots, measured_slots


def calibeovsa(vis=None, caltype=None, caltbdir='', interp=None, docalib=True, doflag=True, flagant='',
               flagspw='', doimage=False, imagedir=None, antenna='', timerange=None, spw=None, stokes=None,
               dosplit=False, outputvis=None, doconcat=False, concatvis=None, keep_orig_ms=True,
               keep_corrected_column=False, cal_npz=None, refcal_npz_mode='bph_sbd', secondary_npz=None,
               force_lo_hi_smooth_extrap=False):
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
        if force_lo_hi_smooth_extrap and refcal_npz_mode == 'bph_sbd':
            print('BPH+SBD LO phase base forced to HI smooth extrapolation for LO bands.')

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
            chan_freqs_per_spw = [
                np.asarray(tb.getcell('CHAN_FREQ', s), dtype=np.float64).reshape(-1)
                for s in range(nspw)
            ]

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
                    bphsbd_arrays = None
                    try:
                        bphsbd_rec = sql2refcal_bphsbdX(btime)
                    except Exception as exc:
                        bphsbd_rec = None
                        bphsbd_reason = "query failed: {0}".format(exc)
                    else:
                        bphsbd_arrays, bphsbd_reason = _valid_sql_bphsbd_arrays(bphsbd_rec)
                    if bphsbd_rec is not None and bphsbd_arrays is not None:
                        try:
                            refcal = sql2refcalX(btime)
                            type8_time = refcal.get("timestamp")
                            refcal = _attach_sql_bphsbd_refcal(refcal, bphsbd_rec, bphsbd_arrays)
                            # Preserve the type-8 refcal time so the daily phacal time filter can
                            # compare phacal t_ref against the refcal they were actually solved
                            # against, rather than the superseding BPH+SBD t_refcal.
                            refcal["type8_timestamp"] = type8_time
                            msg_prompt = "SQL BPH+SBD refcal tables found; superseding type-8 phase calibration"
                            if type8_time is not None:
                                msg_prompt += " from type-8 refcal at {0}".format(type8_time.iso)
                            t_refcal = bphsbd_rec.get("t_refcal")
                            if t_refcal is not None:
                                msg_prompt += " with BPH+SBD t_refcal {0}".format(t_refcal.iso)
                            msg_prompt += "."
                        except Exception as exc:
                            refcal = _attach_sql_bphsbd_refcal({}, bphsbd_rec, bphsbd_arrays)
                            msg_prompt = (
                                "SQL BPH+SBD refcal tables found and used without type-8 metadata "
                                "because sql2refcalX failed: {0}."
                            ).format(exc)
                        cal_src = "SQL BPH+SBD"
                    else:
                        refcal = sql2refcalX(btime)
                        msg_prompt = (
                            "SQL refcal BPH+SBD tables not usable ({0}); using SQL type-8 BPH only."
                        ).format(
                            bphsbd_reason or "record missing"
                        )
                    casalog.post(msg_prompt)
                    print(msg_prompt)
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
                resolved_source = None
                if use_npz_bph_sbd:
                    resolved_nband = max(
                        52,
                        int(np.asarray(refcal.get('fghz', [])).size),
                        max(bd) + 1 if bd else 0,
                    )
                    (
                        resolved_bph,
                        resolved_sbd,
                        resolved_flag,
                        resolved_source,
                        resolved_ref_freq,
                    ) = resolve_bph_sbd_tables(
                        refcal,
                        nant=nant - 1,
                        nband=resolved_nband,
                        force_lo_hi_smooth_extrap=force_lo_hi_smooth_extrap,
                        return_sources=True,
                        return_ref_freqs=True,
                    )
                else:
                    resolved_bph = np.asarray(
                        refcal.get('resolved_bph_rad', []),
                        dtype=np.float64,
                    )
                    resolved_sbd = np.asarray(
                        refcal.get('resolved_sbd_ns', []),
                        dtype=np.float64,
                    )
                    resolved_flag = np.asarray(
                        refcal.get('resolved_flag', []),
                        dtype=np.float64,
                    )
                    resolved_ref_freq = np.full(
                        resolved_bph.shape,
                        np.nan,
                        dtype=np.float64,
                    )
                    saved_band_ref = np.asarray(
                        refcal.get('fghz', []),
                        dtype=np.float64,
                    ).reshape(-1)
                    if resolved_ref_freq.ndim == 3:
                        nband_ref = min(
                            resolved_ref_freq.shape[2],
                            saved_band_ref.size,
                        )
                        resolved_ref_freq[:, :, :nband_ref] = (
                            saved_band_ref[:nband_ref][None, None, :]
                        )
                use_resolved_tables = bool(
                    (use_npz_bph_sbd or not force_lo_hi_smooth_extrap)
                    and resolved_bph.ndim == 3
                    and resolved_sbd.ndim == 3
                    and resolved_flag.ndim == 3
                    and resolved_ref_freq.ndim == 3
                    and resolved_ref_freq.shape == resolved_bph.shape
                )
                smooth_sbd = None
                if use_npz_smooth_model or use_npz_bph_sbd:
                    if refcal.get('pha_source') != 'refcal__model_pha':
                        if use_npz_smooth_model:
                            raise ValueError(
                                'smooth-model refcal mode requires refcal__model_pha in the calwidget v2 NPZ'
                            )
                    smooth_sbd = _smooth_refcal_active_sbd(refcal)
                if use_npz_smooth_model or use_npz_bph_sbd or use_resolved_tables or not use_npz_triplets:
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
                bps_candidate_spw = np.zeros((nant - 1, 2, nspw), dtype=np.uint8)
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
                            if use_resolved_tables:
                                if (
                                    n < resolved_bph.shape[0]
                                    and p < resolved_bph.shape[1]
                                    and band_i < resolved_bph.shape[2]
                                    and n < resolved_sbd.shape[0]
                                    and p < resolved_sbd.shape[1]
                                    and band_i < resolved_sbd.shape[2]
                                    and n < resolved_flag.shape[0]
                                    and p < resolved_flag.shape[1]
                                    and band_i < resolved_flag.shape[2]
                                    and n < resolved_ref_freq.shape[0]
                                    and p < resolved_ref_freq.shape[1]
                                    and band_i < resolved_ref_freq.shape[2]
                                ):
                                    ph = resolved_bph[n, p, band_i]
                                    sb = resolved_sbd[n, p, band_i]
                                    fl = resolved_flag[n, p, band_i]
                                    source_ref_freq_ghz = (
                                        resolved_ref_freq[n, p, band_i]
                                    )
                                else:
                                    ph = 0.0
                                    sb = np.nan
                                    fl = 1.0
                                    source_ref_freq_ghz = np.nan
                                flagged = (
                                    bool(fl)
                                    or not np.isfinite(sb)
                                    or not np.isfinite(source_ref_freq_ghz)
                                )
                                phase_flag_spw[n, p, s] = 1 if flagged else 0
                                phase_rad = (
                                    0.0
                                    if (flagged or not np.isfinite(ph))
                                    else recenter_bph_rad(
                                        ph,
                                        sb,
                                        source_ref_freq_ghz,
                                        float(cfreqs_spw[s]) * 1e-9,
                                    )
                                )
                                if not np.isfinite(phase_rad):
                                    flagged = True
                                    phase_flag_spw[n, p, s] = 1
                                    phase_rad = 0.0
                                para_sbd.append(0.0 if flagged else float(sb))
                                source = 'resolved_tables'
                                if use_npz_bph_sbd:
                                    if (
                                        isinstance(resolved_source, np.ndarray)
                                        and resolved_source.ndim == 3
                                        and n < resolved_source.shape[0]
                                        and p < resolved_source.shape[1]
                                        and band_i < resolved_source.shape[2]
                                    ):
                                        source = str(
                                            resolved_source[n, p, band_i]
                                        )
                                    if not flagged:
                                        bps_candidate_spw[n, p, s] = (
                                            _bps_candidate_code_for_source(source)
                                        )
                                selected_npz_models.add(source)
                            elif band_triplet:
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
                                    None,
                                    n,
                                    p,
                                    band_i,
                                    float(cfreqs_spw[s]) * 1e-9,
                                    hi_sbd_ns,
                                    force_lo_hi_smooth_extrap=force_lo_hi_smooth_extrap,
                                )
                                if not np.isfinite(sbd_ns):
                                    flagged = True
                                    phase_rad = 0.0
                                phase_flag_spw[n, p, s] = 1 if flagged else 0
                                if flagged or not np.isfinite(phase_rad):
                                    phase_rad = 0.0
                                para_sbd.append(0.0 if flagged else float(sbd_ns))
                                if not flagged:
                                    bps_candidate_spw[n, p, s] = (
                                        _bps_candidate_code_for_source(source)
                                    )
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
                if use_npz_bph_sbd and not use_resolved_tables:
                    print("BPH+SBD refcal sources selected by SPW: {0}; SBD source={1}".format(
                        ",".join(sorted(selected_npz_models)) if selected_npz_models else "none",
                        refcal.get('active_ns_source', 'unknown'),
                    ))
                if use_resolved_tables:
                    print("{0} resolved BPH+SBD refcal tables selected by SPW: {1}".format(
                        "NPZ" if cal_npz_refcal is not None else "SQL",
                        ",".join(sorted(selected_npz_models)) if selected_npz_models else "none",
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
                if use_resolved_tables:
                    refcal_npz_suffix = '_npz_resolved_bph_sbd' if cal_npz_refcal is not None else '_sql_bph_sbd'
                elif use_npz_triplets:
                    refcal_npz_suffix = '_npz_triplet'
                elif use_npz_bph_sbd:
                    refcal_npz_suffix = '_npz_bph_sbd'
                elif use_npz_smooth_model:
                    refcal_npz_suffix = '_npz_smooth_model'
                else:
                    refcal_npz_suffix = ''
                caltb_pha = dirname + t_ref.isot[:-4].replace(':', '').replace('-', '') + refcal_npz_suffix + '.refpha'
                if (use_npz_triplets or use_npz_smooth_model or use_npz_bph_sbd or use_resolved_tables) and os.path.exists(caltb_pha):
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
                if use_npz_bph_sbd:
                    caltb_bps = (
                        dirname
                        + t_ref.isot[:-4].replace(':', '').replace('-', '')
                        + refcal_npz_suffix
                        + '.refbps'
                    )
                    if os.path.exists(caltb_bps):
                        shutil.rmtree(caltb_bps)
                    bandpass(
                        vis=msfile,
                        caltable=caltb_bps,
                        solint='inf',
                        refant='eo01',
                        minblperant=0,
                        minsnr=0,
                        bandtype='B',
                        docallib=False,
                    )
                    bps_applied_slots = 0
                    bps_measured_slots = 0
                    tb.open(caltb_bps, nomodify=False)
                    for ll in range(nspw):
                        nchan_ll = int(bd_nchan[ll])
                        freq_ghz = (
                            np.asarray(
                                chan_freqs_per_spw[ll],
                                dtype=np.float64,
                            ).reshape(-1)
                            * 1e-9
                        )
                        if freq_ghz.size != nchan_ll:
                            raise ValueError(
                                'SPW {0:d} NUM_CHAN={1:d} but CHAN_FREQ has {2:d} values'.format(
                                    ll,
                                    nchan_ll,
                                    freq_ghz.size,
                                )
                            )
                        cp, fl, applied_count, measured_count = (
                            _refcal_bps_payload_for_spw(
                                refcal,
                                bps_candidate_spw[:, :, ll],
                                phase_flag_spw[:, :, ll],
                                int(bd[ll]) + 1,
                                freq_ghz,
                                nant,
                            )
                        )
                        bps_applied_slots += int(applied_count)
                        bps_measured_slots += int(measured_count)
                        cp_table = np.moveaxis(cp, 0, 2)
                        flag_table = np.moveaxis(fl, 0, 2)
                        tb.putcol('CPARAM', cp_table, ll * nant, nant)
                        tb.putcol('FLAG', flag_table, ll * nant, nant)
                        tb.putcol(
                            'SNR',
                            np.where(flag_table, 0.0, 100.0),
                            ll * nant,
                            nant,
                        )
                        paramerr = tb.getcol('PARAMERR', ll * nant, nant)
                        tb.putcol('PARAMERR', paramerr * 0, ll * nant, nant)
                    tb.close()
                    gaintables.append(caltb_bps)
                    refcal_gaintables.append(caltb_bps)
                    spwmaps.append([])
                    msg_prompt = (
                        'Validated residual BPS applied to {0:d} of {1:d} '
                        'measured-BPH antenna/pol/SPW slots; table={2}'
                    ).format(
                        bps_applied_slots,
                        bps_measured_slots,
                        os.path.basename(caltb_bps),
                    )
                    casalog.post(msg_prompt)
                    print(msg_prompt)
                if use_npz_triplets or use_npz_smooth_model or use_npz_bph_sbd or use_resolved_tables:
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
                    print("Refcal model gaintables ({0}): {1}".format(
                        'resolved_tables' if use_resolved_tables else refcal_npz_mode,
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
                # Drop phacals whose reference refcal time is >30 min after the refcal they were
                # solved against. In the SQL BPH+SBD path refcal['timestamp'] is the BPH+SBD
                # t_refcal, which can differ from the type-8 refcal the phacals reference, so
                # compare against the captured type-8 time when available to avoid dropping every
                # phacal. Build a keep mask -- `del` on a numpy/Time array raises
                # "ValueError: cannot delete array elements".
                if phacals.any() and len(phacals) > 0:
                    phacal_ref_time = refcal.get('type8_timestamp') or refcal['timestamp']
                    keep = np.array(
                        [(phacal['t_ref'].jd - phacal_ref_time.jd) <= 30. / 1440. for phacal in phacals],
                        dtype=bool,
                    )
                    if not np.all(keep):
                        print("Filtered out {0} phacal(s) with reference time >30 min after refcal {1}".format(
                            int(np.count_nonzero(~keep)), phacal_ref_time.iso))
                    phacals = phacals[keep]
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
                                   parameter=phambd_ns[:nant-1, :].flatten().tolist())
                            if flagspw != '':
                                flag_phambd_by_spw(caltb_phambd_interp, flagspw=flagspw)
                            if os.path.exists(caltb_phambd_interp_pha0):
                                os.system('rm -rf ' + caltb_phambd_interp_pha0)
                            # if not os.path.exists(caltb_phambd_interp_pha0):
                            gencal(vis=msfile, caltable=caltb_phambd_interp_pha0, caltype='ph', pol='X,Y',
                                   antenna=antennas, parameter=pha0[:nant-1, :].flatten().tolist())
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


def _run_refcal_provenance_case(monkeypatch, tmp_path, legacy, bphsbd,
                                applycal_error=None, bps=None,
                                refcal_sql_mode='auto',
                                gencal_calls=None):
    from . import task_calibeovsa as production

    scan_start = Time('2026-07-09 13:51:55.500')
    scan_end = Time('2026-07-09 15:09:56.500')

    class FakeTable:
        def __init__(self):
            self.path = ''

        def open(self, path, *args, **kwargs):
            self.path = str(path)

        def close(self):
            self.path = ''

        def nrows(self):
            if self.path.endswith('/ANTENNA'):
                return 2
            return 1

        def getcol(self, name, *args, **kwargs):
            if self.path.endswith('/SPECTRAL_WINDOW'):
                return {
                    'NAME': np.array(['band01']),
                    'NUM_CHAN': np.array([1]),
                    'REF_FREQUENCY': np.array([1.0e9]),
                    'TOTAL_BANDWIDTH': np.array([1.0e8]),
                    'CHAN_FREQ': np.array([[1.0e9]]),
                }[name]
            if self.path.endswith('/ANTENNA') and name == 'NAME':
                return np.array(['eo01', 'eo02'])
            if self.path.endswith('.refbps') and name == 'PARAMERR':
                return np.zeros((2, 1, 2), dtype=np.float64)
            raise AssertionError('Unexpected getcol({0}) for {1}'.format(name, self.path))

        def getcell(self, name, row):
            if self.path.endswith('/SPECTRAL_WINDOW') and name == 'CHAN_FREQ':
                return np.array([1.0e9])
            if self.path.endswith('/SPECTRAL_WINDOW') and name == 'CHAN_WIDTH':
                return np.array([1.0e8])
            if self.path.endswith('/OBSERVATION') and name == 'TIME_RANGE':
                return np.array([scan_start.mjd, scan_end.mjd]) * 86400.0
            raise AssertionError('Unexpected getcell({0}) for {1}'.format(name, self.path))

        def putcol(self, *args, **kwargs):
            return None

    class FakeCasaLog:
        def origin(self, *args, **kwargs):
            return None

        def post(self, *args, **kwargs):
            return None

    msfile = tmp_path / 'UDB20260709135125.ms'
    msfile.mkdir()
    caltbdir = tmp_path / 'caltable'
    caltbdir.mkdir()

    applied = []
    provenance = []

    def fake_applycal(**kwargs):
        applied.append(kwargs)
        if applycal_error is not None:
            raise applycal_error

    monkeypatch.setattr(production, 'tb', FakeTable())
    monkeypatch.setattr(production, 'casalog', FakeCasaLog())
    monkeypatch.setattr(production, 'sql2refcalX', lambda *_args, **_kwargs: dict(legacy))
    monkeypatch.setattr(production, 'sql2refcal_bphsbdX', lambda *_args, **_kwargs: dict(bphsbd))
    monkeypatch.setattr(production, 'sql2refcal_bpsX', lambda *_args, **_kwargs: bps)
    monkeypatch.setattr(production.db, 'get_reboot', lambda *_args, **_kwargs: [])
    monkeypatch.setattr(production.ch, 'read_calX', lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(
        production,
        'gencal',
        lambda **kwargs: (
            gencal_calls.append(kwargs)
            if gencal_calls is not None
            else None
        ),
    )
    monkeypatch.setattr(production, 'bandpass', lambda **_kwargs: None)
    monkeypatch.setattr(production, 'clearcal', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(production, 'applycal', fake_applycal)

    result = production.calibeovsa(
        str(msfile),
        caltype=['refpha'],
        caltbdir=str(caltbdir) + '/',
        docalib=True,
        doflag=False,
        refcal_sql_mode=refcal_sql_mode,
        refcal_provenance=provenance,
    )

    return str(msfile), scan_start, result, applied, provenance


def test_calibeovsa_reports_the_legacy_refcal_actually_applied(monkeypatch, tmp_path):
    """A stale companion must report the legacy record selected at apply time."""
    legacy_record_time = Time('2026-07-08 07:12:00.000')
    legacy_refcal_time = Time('2026-07-08 12:51:52.000')
    legacy = {
        'pha': np.zeros((2, 2, 1), dtype=np.float64),
        'amp': np.ones((2, 2, 1), dtype=np.float64),
        'flag': np.zeros((2, 2, 1), dtype=np.int32),
        'timestamp': legacy_record_time,
        't_bg': legacy_refcal_time,
        't_ed': Time('2026-07-08 13:39:53.000'),
    }
    stale_bphsbd = {
        'bph_rad': np.zeros((2, 2, 1), dtype=np.float64),
        'sbd_ns': np.zeros((2, 2, 1), dtype=np.float64),
        'flag': np.zeros((2, 2, 1), dtype=np.int32),
        'timestamp': legacy_record_time,
        't_refcal': legacy_refcal_time,
    }

    msfile, scan_start, result, applied, provenance = _run_refcal_provenance_case(
        monkeypatch, tmp_path, legacy, stale_bphsbd)

    assert result == [msfile]
    assert len(applied) == 1
    assert provenance == [{
        'vis': msfile,
        'lookup_time_utc': scan_start.iso,
        'source': 'sql_legacy_type8',
        'mode': 'legacy',
        'sql_record_time_utc': legacy_record_time.iso,
        'refcal_time_utc': legacy_refcal_time.iso,
        'refcal_date_utc': '2026-07-08',
        'applied': True,
    }]


def test_calibeovsa_reports_the_fresh_bph_sbd_refcal_actually_applied(monkeypatch, tmp_path):
    """A fresh companion must expose its own locator and true refcal time."""
    sql_record_time = Time('2026-07-09 07:12:00.000')
    refcal_time = Time('2026-07-09 12:52:53.000')
    legacy = {
        'pha': np.zeros((2, 2, 1), dtype=np.float64),
        'amp': np.ones((2, 2, 1), dtype=np.float64),
        'flag': np.zeros((2, 2, 1), dtype=np.int32),
        'timestamp': sql_record_time,
        't_bg': refcal_time,
        't_ed': Time('2026-07-09 13:46:27.000'),
    }
    bphsbd = {
        'bph_rad': np.zeros((2, 2, 1), dtype=np.float64),
        'sbd_ns': np.zeros((2, 2, 1), dtype=np.float64),
        'flag': np.zeros((2, 2, 1), dtype=np.int32),
        'timestamp': sql_record_time,
        't_refcal': refcal_time,
    }

    msfile, scan_start, result, applied, provenance = _run_refcal_provenance_case(
        monkeypatch, tmp_path, legacy, bphsbd)

    assert result == [msfile]
    assert len(applied) == 1
    assert provenance == [{
        'vis': msfile,
        'lookup_time_utc': scan_start.iso,
        'source': 'sql_bph_sbd',
        'mode': 'bph_sbd',
        'sql_record_time_utc': sql_record_time.iso,
        'refcal_time_utc': refcal_time.iso,
        'refcal_date_utc': '2026-07-09',
        'bps_status': 'skipped',
        'bps_reason': (
            'no type-16 records exist in the lookup local day; selected '
            'historical type-14-only BPH+SBD'
        ),
        'bps_sql_record_time_utc': None,
        'bps_product_digest': None,
        'applied': True,
    }]


def test_calibeovsa_does_not_report_a_refcal_when_applycal_fails(monkeypatch, tmp_path):
    """A failed per-MS apply path must not escape as applied provenance."""
    sql_record_time = Time('2026-07-09 07:12:00.000')
    refcal_time = Time('2026-07-09 12:52:53.000')
    legacy = {
        'pha': np.zeros((2, 2, 1), dtype=np.float64),
        'amp': np.ones((2, 2, 1), dtype=np.float64),
        'flag': np.zeros((2, 2, 1), dtype=np.int32),
        'timestamp': sql_record_time,
        't_bg': refcal_time,
        't_ed': Time('2026-07-09 13:46:27.000'),
    }
    bphsbd = {
        'bph_rad': np.zeros((2, 2, 1), dtype=np.float64),
        'sbd_ns': np.zeros((2, 2, 1), dtype=np.float64),
        'flag': np.zeros((2, 2, 1), dtype=np.int32),
        'timestamp': sql_record_time,
        't_refcal': refcal_time,
    }

    _, _, result, applied, provenance = _run_refcal_provenance_case(
        monkeypatch,
        tmp_path,
        legacy,
        bphsbd,
        applycal_error=RuntimeError('applycal failed'),
    )

    assert result is None
    assert len(applied) == 1
    assert provenance == []


def _sql_bps_route_records(type16_parent_digest=None):
    """Return a small, fully paired SQL BPH+SBD+BPS test product."""

    sql_record_time = Time('2026-07-09 07:12:00.000')
    refcal_time = Time('2026-07-09 12:52:53.000')
    type14_digest = 'a' * 64
    legacy = {
        'pha': np.zeros((2, 2, 1), dtype=np.float64),
        'amp': np.ones((2, 2, 1), dtype=np.float64),
        'flag': np.zeros((2, 2, 1), dtype=np.int32),
        'fghz': np.array([1.05], dtype=np.float64),
        'timestamp': sql_record_time,
        't_bg': refcal_time,
        't_ed': Time('2026-07-09 13:46:27.000'),
    }
    bphsbd = {
        'bph_rad': np.zeros((2, 2, 1), dtype=np.float64),
        'sbd_ns': np.zeros((2, 2, 1), dtype=np.float64),
        'flag': np.zeros((2, 2, 1), dtype=np.int32),
        'timestamp': sql_record_time,
        't_refcal': refcal_time,
        'type14_buffer_digest': type14_digest,
    }
    primary_phase = np.zeros((2, 2, 2), dtype=np.float64)
    primary_phase[0, 0] = [-0.2, 0.2]
    primary_phase[0, 1] = [0.1, -0.1]
    candidate = np.zeros((2, 2, 1), dtype=np.uint8)
    candidate[0, :, 0] = 1
    authorized = np.zeros((2, 2, 1), dtype=np.uint8)
    authorized[0, :, 0] = 1
    bps = {
        'timestamp': sql_record_time,
        't_refcal': refcal_time,
        't_bphsbd': sql_record_time,
        'product_digest': 'b' * 64,
        'type14_digest': (
            type14_digest
            if type16_parent_digest is None
            else type16_parent_digest
        ),
        'bph_ref_frequency_ghz': np.array([1.05], dtype=np.float64),
        'bps_candidate_code': candidate,
        'bps_authorized': authorized,
        'bps_primary': {
            'frequency_ghz': np.array([1.0, 1.1], dtype=np.float64),
            'band': np.array([1, 1], dtype=np.int32),
            'phase_rad': primary_phase,
            'channel_valid': np.ones(primary_phase.shape, dtype=np.uint8),
        },
        'bps_secondary': {
            'frequency_ghz': np.zeros(0, dtype=np.float64),
            'band': np.zeros(0, dtype=np.int32),
            'phase_rad': np.zeros((2, 2, 0), dtype=np.float64),
            'channel_valid': np.zeros((2, 2, 0), dtype=np.uint8),
        },
    }
    return legacy, bphsbd, bps


def test_calibeovsa_applies_fresh_matched_sql_residual_bps(monkeypatch, tmp_path):
    """A paired type 16 must add the residual B table to the SQL route."""

    legacy, bphsbd, bps = _sql_bps_route_records()
    msfile, _, result, applied, provenance = _run_refcal_provenance_case(
        monkeypatch,
        tmp_path,
        legacy,
        bphsbd,
        bps=bps,
    )

    assert result == [msfile]
    assert len(applied) == 1
    assert any(
        str(table).endswith('.refbps')
        for table in applied[0]['gaintable']
    )
    assert provenance[0]['source'] == 'sql_bph_sbd'
    assert provenance[0]['bps_status'] == 'applied'
    assert provenance[0]['bps_reason'] is None
    assert provenance[0]['bps_product_digest'] == 'b' * 64
    assert provenance[0]['bps_applied_slots'] == 2
    assert provenance[0]['bps_authorized_slots'] == 2


def test_calibeovsa_skips_mismatched_sql_residual_bps(monkeypatch, tmp_path):
    """Auto mode must not activate a type 14 from a partial family."""

    legacy, bphsbd, bps = _sql_bps_route_records(
        type16_parent_digest='c' * 64,
    )
    msfile, _, result, applied, provenance = _run_refcal_provenance_case(
        monkeypatch,
        tmp_path,
        legacy,
        bphsbd,
        bps=bps,
    )

    assert result == [msfile]
    assert len(applied) == 1
    assert not any(
        str(table).endswith('.refbps')
        for table in applied[0]['gaintable']
    )
    assert provenance[0]['source'] == 'sql_legacy_type8'


def test_explicit_bph_sbd_rejects_incomplete_sql_family(monkeypatch, tmp_path):
    """Explicit full-route mode must fail when type 16 does not match."""

    legacy, bphsbd, bps = _sql_bps_route_records(
        type16_parent_digest='c' * 64,
    )
    _, _, result, applied, provenance = _run_refcal_provenance_case(
        monkeypatch,
        tmp_path,
        legacy,
        bphsbd,
        bps=bps,
        refcal_sql_mode='bph_sbd',
    )

    assert result is None
    assert applied == []
    assert provenance == []


def test_sql_bps_pairing_searches_past_newer_orphan(monkeypatch):
    """A newer partial-save orphan must not hide an exact older companion."""

    from . import task_calibeovsa as production

    _legacy, bphsbd, matching = _sql_bps_route_records()
    orphan = dict(matching)
    orphan['type14_digest'] = 'c' * 64
    calls = []

    def fake_loader(trange, **kwargs):
        calls.append((trange, kwargs))
        return [orphan, matching]

    monkeypatch.setattr(production, 'sql2refcal_bpsX', fake_loader)
    arrays, reason = production._valid_sql_bphsbd_arrays(bphsbd)
    assert reason is None

    companion, reason = production._sql_bps_companion_for_bphsbd(
        Time('2026-07-09 18:00:00.000'),
        bphsbd,
        arrays,
    )

    assert reason is None
    assert companion is not None
    assert companion['type14_digest'] == 'a' * 64
    assert len(calls) == 1
    assert calls[0][1]['nrecords'] == 0
    assert calls[0][1]['neat'] is False


def test_sql_family_selects_older_complete_pair_past_newer_partials():
    """The family selector must compare every same-day type-14/type-16 pair."""

    from . import task_calibeovsa as production

    _legacy, older_type14, older_type16 = _sql_bps_route_records()
    newer_time = Time('2026-07-09 08:12:00.000')
    newer_type14 = dict(older_type14)
    newer_type14.update({
        'timestamp': newer_time,
        'type14_buffer_digest': 'd' * 64,
    })
    newer_type16 = dict(older_type16)
    newer_type16.update({
        'timestamp': newer_time,
        't_bphsbd': newer_time,
        'type14_digest': 'e' * 64,
    })

    selected = production._select_sql_refcal_family(
        [older_type14, newer_type14],
        [older_type16, newer_type16],
        Time('2026-07-09 18:00:00.000'),
    )

    assert selected['status'] == 'complete'
    assert selected['bphsbd_record']['type14_buffer_digest'] == 'a' * 64
    assert selected['bps_companion']['type14_digest'] == 'a' * 64


def test_sql_family_marks_unmatched_type16_day_partial():
    """Any same-day type 16 blocks historical type-14-only activation."""

    from . import task_calibeovsa as production

    _legacy, bphsbd, bps = _sql_bps_route_records(
        type16_parent_digest='c' * 64,
    )
    selected = production._select_sql_refcal_family(
        bphsbd,
        bps,
        Time('2026-07-09 18:00:00.000'),
    )

    assert selected['status'] == 'partial'
    assert selected['type16_records_present'] is True
    assert selected['bphsbd_record'] is None
    assert 'no complete digest-matched' in selected['reason']


def test_sql_family_allows_historical_type14_only_when_day_has_no_type16():
    """Auto-mode compatibility is limited to genuinely pre-type-16 days."""

    from . import task_calibeovsa as production

    _legacy, bphsbd, _bps = _sql_bps_route_records()
    selected = production._select_sql_refcal_family(
        bphsbd,
        None,
        Time('2026-07-09 18:00:00.000'),
    )

    assert selected['status'] == 'historical_type14'
    assert selected['type16_records_present'] is False
    assert selected['bphsbd_record'] is bphsbd


def test_complete_sql_family_uses_type16_bph_pivot_not_type8_fghz(
        monkeypatch, tmp_path):
    """Type-8 band frequencies must not move a complete-family BPH phase."""

    legacy, bphsbd, bps = _sql_bps_route_records()
    legacy['fghz'] = np.array([9.0], dtype=np.float64)
    bphsbd['sbd_ns'] = np.ones((2, 2, 1), dtype=np.float64)
    gencal_calls = []

    msfile, _, result, _, _ = _run_refcal_provenance_case(
        monkeypatch,
        tmp_path,
        legacy,
        bphsbd,
        bps=bps,
        refcal_sql_mode='bph_sbd',
        gencal_calls=gencal_calls,
    )

    assert result == [msfile]
    refpha = [
        call for call in gencal_calls
        if call.get('caltype') == 'ph'
        and str(call.get('caltable', '')).endswith('.refpha')
    ]
    assert len(refpha) == 1
    np.testing.assert_allclose(refpha[0]['parameter'], [-18.0, -18.0])


def test_companion_sql_phacal_filter_rejects_old_sentinel_and_keeps_match():
    """Companion SQL modes must not admit old one-sided ``t_ref`` rows."""
    from . import task_calibeovsa as production

    refcal_time = Time('2025-03-26 10:44:50.000')
    phacals = np.array([
        {'t_ref': Time('1903-12-31 23:59:59.000')},
        {'t_ref': Time('2025-03-26 10:30:00.000')},
        {'t_ref': Time('2025-03-26 11:20:00.000')},
    ], dtype=object)

    strict = production._phacal_reference_keep_mask(
        phacals,
        refcal_time,
        require_absolute_match=True,
    )
    legacy = production._phacal_reference_keep_mask(
        phacals,
        refcal_time,
        require_absolute_match=False,
    )

    np.testing.assert_array_equal(strict, [False, True, False])
    np.testing.assert_array_equal(legacy, [True, True, False])


def test_refcal_bps_payload_uses_same_measured_candidate_and_neutral_tiers():
    """BPS follows measured-BPH provenance and leaves other tiers neutral."""
    from . import task_calibeovsa as production

    primary_phase = np.zeros((2, 2, 2), dtype=np.float64)
    primary_phase[0, 0] = [-0.2, 0.2]
    secondary_phase = np.zeros((2, 2, 2), dtype=np.float64)
    secondary_phase[0, 1] = [0.3, -0.3]
    primary = {
        'bps_frequency_ghz': np.array([1.0, 1.1], dtype=np.float64),
        'bps_band': np.array([1, 1], dtype=np.int32),
        'bps_phase_rad': primary_phase,
        'bps_valid': np.ones((2, 2, 1), dtype=np.uint8),
    }
    primary['secondary_bph_refcal'] = {
        'bps_frequency_ghz': np.array([1.0, 1.1], dtype=np.float64),
        'bps_band': np.array([1, 1], dtype=np.int32),
        'bps_phase_rad': secondary_phase,
        'bps_valid': np.ones((2, 2, 1), dtype=np.uint8),
    }
    candidate_code = np.array([[1, 2], [0, 1]], dtype=np.uint8)
    phase_flag = np.array([[0, 0], [0, 1]], dtype=np.uint8)

    cparam, flag, applied_slots, measured_slots = (
        production._refcal_bps_payload_for_spw(
            primary,
            candidate_code,
            phase_flag,
            1,
            np.array([1.0, 1.1], dtype=np.float64),
            3,
        )
    )

    assert production._bps_candidate_code_for_source('band_phase') == 1
    assert production._bps_candidate_code_for_source('secondary_band_phase') == 2
    assert production._bps_candidate_code_for_source('lo_model') == 1
    assert production._bps_candidate_code_for_source('hi_model') == 1
    assert production._bps_candidate_code_for_source('secondary_lo_model') == 2
    assert production._bps_candidate_code_for_source('secondary_hi_model') == 2
    assert production._bps_candidate_code_for_source('hi_smooth_extrap') == 0
    assert applied_slots == 2
    assert measured_slots == 2
    np.testing.assert_allclose(cparam[0, 0], np.exp(1j * np.array([-0.2, 0.2])))
    np.testing.assert_allclose(cparam[0, 1], np.exp(1j * np.array([0.3, -0.3])))
    np.testing.assert_allclose(cparam[1], 1.0 + 0j)
    np.testing.assert_allclose(np.abs(cparam), 1.0)
    np.testing.assert_array_equal(flag[0], False)
    np.testing.assert_array_equal(flag[1, 0], False)
    np.testing.assert_array_equal(flag[1, 1], True)
    np.testing.assert_array_equal(flag[2], True)


def test_refcal_bps_payload_keeps_valid_bph_sbd_on_sampler_failure():
    """Unsupported optional BPS must leave valid BPH+SBD unflagged."""
    from . import task_calibeovsa as production

    refcal = {
        'bps_frequency_ghz': np.array([1.0, 1.1], dtype=np.float64),
        'bps_band': np.array([1, 1], dtype=np.int32),
        'bps_phase_rad': np.zeros((2, 2, 2), dtype=np.float64),
        'bps_valid': np.ones((2, 2, 1), dtype=np.uint8),
    }
    candidate_code = np.array([[0, 0], [1, 0]], dtype=np.uint8)
    phase_flag = np.zeros((2, 2), dtype=np.uint8)

    cparam, flag, applied_slots, authorized_slots = (
        production._refcal_bps_payload_for_spw(
            refcal,
            candidate_code,
            phase_flag,
            1,
            np.array([0.9, 1.0], dtype=np.float64),
            2,
        )
    )

    assert authorized_slots == 1
    assert applied_slots == 0
    np.testing.assert_allclose(cparam[1, 0], 1.0 + 0j)
    np.testing.assert_array_equal(flag[1, 0], False)


def test_promoted_hi_model_source_does_not_apply_original_measured_bps():
    """A promoted HI row must not reuse BPS from its original refcal data."""
    from . import task_calibeovsa as production

    hi_triplet = {
        'phi_band_rad': np.array(
            [[[0.2], [0.3]], [[0.8], [0.9]]],
            dtype=np.float64,
        ),
        'tau_ib_ns': np.ones((2, 2, 1), dtype=np.float64) * 0.2,
        'tau_mb_eff_ns': np.array(
            [[0.04, 0.05], [0.07, 0.08]],
            dtype=np.float64,
        ),
        'band_ref_freq_ghz': np.array([4.0], dtype=np.float64),
        'flag': np.zeros((2, 2, 1), dtype=np.uint8),
    }
    refcal = {
        'band_phase_rad': np.array(
            [[[0.1], [0.2]], [[2.4], [-2.2]]],
            dtype=np.float64,
        ),
        'band_phase_flag': np.zeros((2, 2, 1), dtype=np.uint8),
        'band_phase_quality_flag': np.zeros(
            (2, 2, 1), dtype=np.uint8
        ),
        'smooth_bph_quality_flag': np.zeros(
            (2, 2, 1), dtype=np.uint8
        ),
        'active_ns': np.array(
            [[8.0, 7.0], [10.0, 9.0]], dtype=np.float64
        ),
        'delay_flag': np.zeros((2, 2), dtype=np.uint8),
        'fghz': np.array([4.0], dtype=np.float64),
        'gencal_triplets': {'hi': hi_triplet},
        'operator_band_flag': np.zeros(
            (2, 2, 1), dtype=np.uint8
        ),
        'promoted_antennas': {
            '1': {'anchor_active_ns': [10.0, 9.0]}
        },
        'bps_frequency_ghz': np.array(
            [3.95, 4.05], dtype=np.float64
        ),
        'bps_band': np.array([1, 1], dtype=np.int32),
        'bps_phase_rad': np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[-0.4, 0.4], [0.3, -0.3]],
            ],
            dtype=np.float64,
        ),
        'bps_valid': np.ones((2, 2, 1), dtype=np.uint8),
    }
    _, _, resolved_flag, source = resolve_bph_sbd_tables(
        refcal,
        nant=2,
        nband=1,
        return_sources=True,
    )
    codes = np.zeros((2, 2), dtype=np.uint8)
    for ant_i in range(2):
        for pol_i in range(2):
            codes[ant_i, pol_i] = (
                production._bps_candidate_code_for_slot(
                    refcal,
                    source[ant_i, pol_i, 0],
                    ant_i,
                    pol_i,
                    0,
                )
            )

    cparam, flag, applied_slots, measured_slots = (
        production._refcal_bps_payload_for_spw(
            refcal,
            codes,
            resolved_flag[:, :, 0],
            1,
            np.array([3.95, 4.05], dtype=np.float64),
            3,
        )
    )

    np.testing.assert_array_equal(source[1, :, 0], ['hi_model', 'hi_model'])
    np.testing.assert_array_equal(codes[1], 0)
    np.testing.assert_allclose(cparam[1], 1.0 + 0j)
    np.testing.assert_array_equal(flag[1], False)
    self_applied = int(np.count_nonzero(codes[0] == 1))
    assert measured_slots == self_applied
    assert applied_slots == self_applied

    ant1_promoted = dict(refcal)
    ant1_promoted['promoted_antennas'] = {
        '0': {'anchor_active_ns': [8.0, 7.0]}
    }
    _, _, ant1_flag, ant1_source = resolve_bph_sbd_tables(
        ant1_promoted,
        nant=2,
        nband=1,
        return_sources=True,
    )
    ant1_codes = np.zeros((2, 2), dtype=np.uint8)
    for ant_i in range(2):
        for pol_i in range(2):
            ant1_codes[ant_i, pol_i] = (
                production._bps_candidate_code_for_slot(
                    ant1_promoted,
                    ant1_source[ant_i, pol_i, 0],
                    ant_i,
                    pol_i,
                    0,
                )
            )
    ant1_cparam, _, _, _ = production._refcal_bps_payload_for_spw(
        ant1_promoted,
        ant1_codes,
        ant1_flag[:, :, 0],
        1,
        np.array([3.95, 4.05], dtype=np.float64),
        3,
    )
    np.testing.assert_array_equal(
        ant1_source[1, :, 0],
        ['hi_model', 'hi_model'],
    )
    np.testing.assert_array_equal(ant1_codes[1], 0)
    np.testing.assert_allclose(ant1_cparam[1], 1.0 + 0j)


def _legacy_band14_grids():
    width_hz = np.full(8, 40.625e6, dtype=np.float64)
    lower_edge_hz = 5.325e9 + np.arange(8) * width_hz
    return lower_edge_hz, width_hz, lower_edge_hz + width_hz / 2.0


def _single_band_bps_refcal(center_hz):
    phase = np.linspace(-0.35, 0.35, center_hz.size)
    return {
        'bps_frequency_ghz': center_hz * 1e-9,
        'bps_band': np.full(center_hz.size, 14, dtype=np.int32),
        'bps_phase_rad': np.broadcast_to(
            phase,
            (1, 2, phase.size),
        ).copy(),
        'bps_valid': np.ones((1, 2, 14), dtype=np.uint8),
    }


def test_legacy_eovsa_band14_grid_recovers_centers_for_bps():
    """Legacy impteovsa lower edges must be sampled at physical centers."""

    from . import task_calibeovsa as production

    lower_edges_hz, channel_width_hz, expected_centers_hz = (
        _legacy_band14_grids()
    )
    target_ghz = production._eovsa_bps_target_channel_centers(
        'band14',
        lower_edges_hz,
        channel_width_hz,
    ) * 1e-9

    _, flag, applied_slots, measured_slots = (
        production._refcal_bps_payload_for_spw(
            _single_band_bps_refcal(expected_centers_hz),
            np.ones((1, 2), dtype=np.uint8),
            np.zeros((1, 2), dtype=np.uint8),
            14,
            target_ghz,
            1,
        )
    )

    np.testing.assert_allclose(target_ghz, expected_centers_hz * 1e-9)
    np.testing.assert_array_equal(flag, False)
    assert applied_slots == measured_slots == 2


def test_standard_eovsa_band14_centers_are_not_shifted_twice():
    """A standards-compliant EOVSA MS must pass through unchanged."""

    from . import task_calibeovsa as production

    _, channel_width_hz, centers_hz = _legacy_band14_grids()

    target_hz = production._eovsa_bps_target_channel_centers(
        'band14',
        centers_hz,
        channel_width_hz,
    )

    np.testing.assert_array_equal(target_hz, centers_hz)


def test_raw_legacy_lower_edges_still_fail_closed_in_bps_sampler():
    """The strict sampler must not extrapolate an unrecovered target grid."""

    from . import task_calibeovsa as production

    lower_edges_hz, _, centers_hz = _legacy_band14_grids()

    phase, applied = production.sample_refcal_bps_for_band(
        _single_band_bps_refcal(centers_hz),
        0,
        0,
        14,
        lower_edges_hz * 1e-9,
    )

    np.testing.assert_array_equal(phase, 0.0)
    assert applied is False


def test_recovered_legacy_grid_preserves_npz_sql_bps_14_of_14_parity():
    """NPZ and normalized SQL candidates must apply the same 14 slots."""

    from eovsapy import chan_util_52
    from . import task_calibeovsa as production

    authorized_slots = (
        (2, 1, 7),
        (4, 0, 14),
        (4, 0, 15),
        (4, 1, 7),
        (5, 0, 7),
        (5, 0, 8),
        (5, 0, 14),
        (5, 0, 21),
        (5, 1, 11),
        (5, 1, 15),
        (7, 0, 7),
        (7, 0, 25),
        (7, 1, 7),
        (7, 1, 13),
    )
    bands = sorted({band for _, _, band in authorized_slots})
    authorization = np.zeros((8, 2, 52), dtype=np.uint8)
    for ant_i, pol_i, band_id in authorized_slots:
        authorization[ant_i, pol_i, band_id - 1] = 1

    frequency_parts = []
    band_parts = []
    phase_parts = []
    target_by_band = {}
    for band_id in bands:
        lower_edges_hz = np.asarray(
            chan_util_52.start_freq(band_id),
            dtype=np.float64,
        ) * 1e9
        channel_width_hz = np.asarray(
            chan_util_52.sci_bw(band_id),
            dtype=np.float64,
        ) * 1e9
        centers_ghz = production._eovsa_bps_target_channel_centers(
            'band{0:02d}'.format(band_id),
            lower_edges_hz,
            channel_width_hz,
        ) * 1e-9
        target_by_band[band_id] = centers_ghz
        frequency_parts.append(centers_ghz)
        band_parts.append(
            np.full(centers_ghz.size, band_id, dtype=np.int32)
        )
        phase_parts.append(
            np.broadcast_to(
                np.linspace(-0.3, 0.3, centers_ghz.size),
                (8, 2, centers_ghz.size),
            ).copy()
        )

    frequency = np.concatenate(frequency_parts)
    channel_band = np.concatenate(band_parts)
    phase = np.concatenate(phase_parts, axis=2)
    npz_refcal = {
        'bps_frequency_ghz': frequency,
        'bps_band': channel_band,
        'bps_phase_rad': phase,
        'bps_valid': authorization,
    }
    sql_refcal, reason = production._normalized_sql_bps_group(
        {
            'frequency_ghz': frequency,
            'band': channel_band,
            'phase_rad': phase,
            'channel_valid': np.ones(phase.shape, dtype=np.uint8),
        },
        (8, 2),
        52,
    )
    assert reason is None
    sql_refcal['bps_valid'] = authorization

    totals = {'npz': [0, 0], 'sql': [0, 0]}
    for band_id in bands:
        codes = authorization[:, :, band_id - 1]
        payloads = {}
        for source, refcal in (
                ('npz', npz_refcal),
                ('sql', sql_refcal)):
            cparam, flag, applied, measured = (
                production._refcal_bps_payload_for_spw(
                    refcal,
                    codes,
                    np.zeros((8, 2), dtype=np.uint8),
                    band_id,
                    target_by_band[band_id],
                    8,
                )
            )
            totals[source][0] += applied
            totals[source][1] += measured
            payloads[source] = (cparam, flag)
        np.testing.assert_allclose(
            payloads['npz'][0],
            payloads['sql'][0],
        )
        np.testing.assert_array_equal(
            payloads['npz'][1],
            payloads['sql'][1],
        )

    assert totals == {'npz': [14, 14], 'sql': [14, 14]}
