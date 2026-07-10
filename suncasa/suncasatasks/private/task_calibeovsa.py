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
from eovsapy.sqlutil import sql2refcalX, sql2phacalX, sql2refcal_bphsbdX, sql2refcal_bphaseX
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
    smooth_phase_bandpass_for_freqs,
    load_calwidget_v2_npz,
    _attach_secondary_bph_refcal,
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


def _single_sql_record(record):
    if isinstance(record, list):
        return record[-1] if record else None
    return record


def _sql_lookup_local_day_bounds(tim):
    try:
        dhr = tim.LocalTime.utcoffset().total_seconds() / 60. / 60.
    except Exception:
        dhr = -7.
    btime = Time(np.fix(tim.mjd + dhr / 24.) - dhr / 24., format='mjd')
    return btime, Time(btime.mjd + 1., format='mjd')


def _sql_record_in_lookup_day(record, lookup_time, label):
    if not isinstance(record, dict):
        return False, "{0} record missing".format(label)
    record_time = record.get("timestamp")
    if record_time is None:
        return False, "{0} record has no SQL timestamp".format(label)
    bday, eday = _sql_lookup_local_day_bounds(lookup_time)
    if bday.mjd <= record_time.mjd < eday.mjd:
        return True, None
    return False, (
        "{0} SQL timestamp {1} is outside lookup local day {2} to {3}"
    ).format(label, record_time.iso, bday.iso, eday.iso)


def _sql_smb_record_to_refcal(sql_lookup_time):
    try:
        smb_rec = sql2refcal_bphaseX(sql_lookup_time)
    except Exception as exc:
        return None, "query failed: {0}".format(exc)
    smb_rec = _single_sql_record(smb_rec)
    fresh, fresh_reason = _sql_record_in_lookup_day(
        smb_rec, sql_lookup_time, "SQL smooth-bandpass"
    )
    if not fresh:
        return None, fresh_reason
    smb_phase = (np.asarray(smb_rec['phase_rad'], dtype=np.float64)
                 if smb_rec is not None else np.zeros(0))
    smb_freq = (np.asarray(smb_rec['freq_ghz'], dtype=np.float64).reshape(-1)
                if smb_rec is not None else np.zeros(0))
    if smb_rec is None or smb_phase.ndim != 3 or smb_phase.size == 0 or smb_freq.size == 0:
        return None, "no usable caltype-15 smooth phase bandpass arrays"
    try:
        refcal = sql2refcalX(sql_lookup_time)
    except Exception as exc:
        raise ValueError(
            'SQL smooth-bandpass mode needs a type-8 refcal for metadata, '
            'but sql2refcalX failed for {0}: {1}'.format(sql_lookup_time.iso, exc))
    type8_time = refcal.get('timestamp')
    smb_flag = np.asarray(smb_rec['flag'], dtype=np.float64)
    if smb_flag.shape == smb_phase.shape:
        smb_phase = np.where(smb_flag != 0, np.nan, smb_phase)
    refcal['model_phase_fine'] = smb_phase
    refcal['fine_frequency_ghz'] = smb_freq
    refcal.pop('lo_model_phase_fine', None)
    refcal.pop('lo_model_fine_frequency_ghz', None)
    refcal['type8_timestamp'] = type8_time
    refcal['sql_smb_timestamp'] = smb_rec.get('timestamp')
    t_refcal = smb_rec.get('t_refcal')
    if t_refcal is not None:
        # Anchor caltable naming and the phacal filter to the SMB record's
        # t_refcal.  This mirrors the BPH+SBD anchoring below.
        refcal['sql_smb_t_refcal'] = t_refcal
        refcal['timestamp'] = t_refcal
    msg_prompt = ('SQL smooth phase bandpass (caltype 15) found; applying as a '
                  'per-channel phase-only B table')
    if t_refcal is not None:
        msg_prompt += ' with t_refcal {0}'.format(t_refcal.iso)
    msg_prompt += '.'
    return refcal, msg_prompt


def _time_iso_or_none(value):
    if value is None:
        return None
    return value.iso if hasattr(value, 'iso') else str(value)


def _refcal_provenance_entry(msfile, lookup_time, cal_src, refcal,
                             refcal_npz_mode, is_npz):
    if is_npz:
        source = 'calwidget_v2_npz'
        mode = refcal_npz_mode
        sql_record_time = None
        refcal_time = refcal.get('t_bg') or refcal.get('timestamp')
    elif cal_src == 'SQL BPH+SBD':
        source = 'sql_bph_sbd'
        mode = 'bph_sbd'
        sql_record_time = refcal.get('sql_bphsbd_timestamp')
        refcal_time = refcal.get('sql_bphsbd_t_refcal') or refcal.get('timestamp')
    elif cal_src == 'SQL smooth bandpass (caltype 15)':
        source = 'sql_smooth_bandpass'
        mode = 'smooth_bandpass'
        sql_record_time = refcal.get('sql_smb_timestamp')
        refcal_time = refcal.get('sql_smb_t_refcal') or refcal.get('timestamp')
    else:
        source = 'sql_legacy_type8'
        mode = 'legacy'
        sql_record_time = refcal.get('type8_timestamp') or refcal.get('timestamp')
        refcal_time = refcal.get('t_bg') or refcal.get('timestamp')

    refcal_time_utc = _time_iso_or_none(refcal_time)
    return {
        'vis': str(msfile),
        'lookup_time_utc': _time_iso_or_none(lookup_time),
        'source': source,
        'mode': mode,
        'sql_record_time_utc': _time_iso_or_none(sql_record_time),
        'refcal_time_utc': refcal_time_utc,
        'refcal_date_utc': refcal_time_utc[:10] if refcal_time_utc else None,
        'applied': False,
    }


def _attach_sql_bphsbd_refcal(refcal, record, arrays):
    bph, sbd, flag = arrays
    if not isinstance(refcal, dict):
        refcal = {}
    refcal["resolved_bph_rad"] = bph
    refcal["resolved_sbd_ns"] = sbd
    refcal["resolved_flag"] = flag
    refcal["sql_bphsbd_t_refcal"] = record.get("t_refcal")
    refcal["sql_bphsbd_timestamp"] = record.get("timestamp")
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


def calibeovsa(vis=None, caltype=None, caltbdir='', interp=None, docalib=True, doflag=True, flagant='',
               flagspw='', doimage=False, imagedir=None, antenna='', timerange=None, spw=None, stokes=None,
               dosplit=False, outputvis=None, doconcat=False, concatvis=None, keep_orig_ms=True,
               keep_corrected_column=False, cal_npz=None, refcal_npz_mode='bph_sbd', secondary_npz=None,
               force_lo_hi_smooth_extrap=False, refcal_sql_mode='auto', sql_cal_time=None,
               refcal_provenance=None):
    '''

    :param vis: EOVSA visibility dataset(s) to be calibrated 
    :param caltype:
    :param interp:
    :param docalib:
    :param qlookimage:
    :param flagant:
    :param stokes:
    :param doconcat:
    :param refcal_provenance: Optional mutable list receiving one refcal
        provenance record per successfully processed input MS.
    :return:
    '''

    interp0 = interp
    refcal_npz_mode = _normalize_refcal_npz_mode(refcal_npz_mode)
    refcal_sql_mode = (refcal_sql_mode or 'auto').strip().lower()
    if refcal_sql_mode not in ('auto', 'bph_sbd', 'smb', 'legacy'):
        raise ValueError(
            "refcal_sql_mode must be 'auto', 'bph_sbd', 'smb', or 'legacy', got {0!r}".format(
                refcal_sql_mode
            )
        )

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
            pending_refcal_provenance = None
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
            # Per-spw channel frequencies (Hz), kept for the smooth_bandpass mode
            # which evaluates the smooth phase model at every science channel.
            chan_freqs_per_spw = [
                np.asarray(tb.getcell('CHAN_FREQ', s), dtype=np.float64).reshape(-1) for s in range(nspw)
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
            sql_lookup_time = btime
            if sql_cal_time is not None:
                sql_lookup_time = Time(sql_cal_time)
                print("SQL calibration lookup time override: {0} for scan beginning {1}".format(
                    sql_lookup_time.iso, btime.iso))
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
                sql_smb_active = False
                if cal_npz_refcal is not None:
                    refcal = cal_npz_refcal
                elif refcal_sql_mode == 'legacy':
                    refcal = sql2refcalX(sql_lookup_time)
                    refcal['type8_timestamp'] = refcal.get('timestamp')
                    cal_src = 'SQL legacy type-8 BPH'
                    msg_prompt = (
                        'SQL legacy type-8 BPH refcal selected; skipping SQL BPH+SBD '
                        'and smooth-bandpass companion records.'
                    )
                    casalog.post(msg_prompt)
                    print(msg_prompt)
                elif refcal_sql_mode == 'smb':
                    # No-NPZ smooth-bandpass path: read the caltype-15 smooth phase
                    # bandpass record and apply it as a per-channel B table. The
                    # record stores the complete canonical channel-phase model
                    # plus per-channel flags, so we map it onto model_phase_fine /
                    # fine_frequency_ghz (flagged channels -> NaN so the
                    # interpolation drops them) and do NOT set the LO fine fields
                    # (no re-combination).  The SQL15 B table is the full phase
                    # correction for this mode; do not append a separate SBD table.
                    smb_result = _sql_smb_record_to_refcal(sql_lookup_time)
                    if smb_result[0] is None:
                        raise ValueError(
                            'refcal_sql_mode="smb" requested but no usable caltype-15 smooth '
                            'phase bandpass record was found in SQL for {0}: {1}.'.format(
                                sql_lookup_time.iso, smb_result[1]))
                    refcal, msg_prompt = smb_result
                    sql_smb_active = True
                    cal_src = 'SQL smooth bandpass (caltype 15)'
                    casalog.post(msg_prompt)
                    print(msg_prompt)
                else:
                    bphsbd_arrays = None
                    try:
                        bphsbd_rec = sql2refcal_bphsbdX(sql_lookup_time)
                    except Exception as exc:
                        bphsbd_rec = None
                        bphsbd_reason = "query failed: {0}".format(exc)
                    else:
                        bphsbd_rec = _single_sql_record(bphsbd_rec)
                        bphsbd_fresh, bphsbd_reason = _sql_record_in_lookup_day(
                            bphsbd_rec, sql_lookup_time, "SQL BPH+SBD"
                        )
                        if bphsbd_fresh:
                            bphsbd_arrays, bphsbd_reason = _valid_sql_bphsbd_arrays(bphsbd_rec)
                    if bphsbd_rec is not None and bphsbd_arrays is not None:
                        try:
                            refcal = sql2refcalX(sql_lookup_time)
                            type8_time = refcal.get("timestamp")
                            refcal = _attach_sql_bphsbd_refcal(refcal, bphsbd_rec, bphsbd_arrays)
                            # Keep the legacy type-8 time for diagnostics only.  SQL BPH+SBD
                            # phacals are solved against the BPH+SBD t_refcal, so the phacal
                            # filter below must not use this legacy timestamp in this mode.
                            refcal["type8_timestamp"] = type8_time
                            msg_prompt = "SQL BPH+SBD refcal tables found; superseding type-8 phase calibration"
                            if type8_time is not None:
                                msg_prompt += " from type-8 refcal at {0}".format(type8_time.iso)
                            t_refcal = bphsbd_rec.get("t_refcal")
                            if t_refcal is not None:
                                # BPH+SBD supersedes type-8, so make its t_refcal the
                                # authoritative runtime refcal time. sql2refcalX above
                                # already populated refcal['timestamp'] with the legacy
                                # type-8 time and _attach_sql_bphsbd_refcal's
                                # `if "timestamp" not in refcal` guard left it untouched,
                                # which made the phacal filter, the DCM delay-center
                                # t_ref lookup, and caltable naming all anchor to the
                                # stale type-8 time. type8_timestamp above preserves the
                                # legacy time for diagnostics.
                                refcal["timestamp"] = t_refcal
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
                        if refcal_sql_mode == 'auto':
                            smb_result = _sql_smb_record_to_refcal(sql_lookup_time)
                            if smb_result[0] is not None:
                                refcal, msg_prompt = smb_result
                                sql_smb_active = True
                                cal_src = 'SQL smooth bandpass (caltype 15)'
                                msg_prompt = (
                                    "SQL BPH+SBD tables not usable ({0}); "
                                ).format(bphsbd_reason or "record missing") + msg_prompt
                            else:
                                refcal = sql2refcalX(sql_lookup_time)
                                cal_src = 'SQL legacy type-8 BPH'
                                msg_prompt = (
                                    "SQL BPH+SBD tables not usable ({0}); SQL smooth-bandpass "
                                    "not usable ({1}); using legacy SQL type-8 BPH + phacal MBD."
                                ).format(
                                    bphsbd_reason or "record missing",
                                    smb_result[1] or "record missing",
                                )
                        else:
                            refcal = sql2refcalX(sql_lookup_time)
                            cal_src = 'SQL legacy type-8 BPH'
                            msg_prompt = (
                                "SQL refcal BPH+SBD tables not usable ({0}); "
                                "using legacy SQL type-8 BPH + phacal MBD."
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
                # Phase-only per-channel bandpass mode: the refcal phase is applied
                # as a B table sampling the smooth model across frequency.  SQL
                # SMB caltype-15 is already the complete channel-phase model.
                use_npz_smooth_bandpass = bool(cal_npz_refcal is not None and refcal_npz_mode == 'smooth_bandpass')
                if use_npz_smooth_bandpass:
                    _bp_fine = np.asarray(refcal.get('model_phase_fine', []), dtype=np.float64)
                    if _bp_fine.ndim != 3 or _bp_fine.size == 0:
                        raise ValueError(
                            'smooth_bandpass refcal mode requires refcal__model_phase_fine in the calwidget v2 NPZ'
                        )
                # No-NPZ SQL smooth-bandpass mode (caltype 15) shares the same
                # per-channel B-table builder as the NPZ smooth_bandpass mode.
                use_sql_smooth_bandpass = bool(cal_npz_refcal is None and sql_smb_active)
                use_smooth_bandpass = bool(use_npz_smooth_bandpass or use_sql_smooth_bandpass)
                resolved_bph = np.asarray(refcal.get('resolved_bph_rad', []), dtype=np.float64)
                resolved_sbd = np.asarray(refcal.get('resolved_sbd_ns', []), dtype=np.float64)
                resolved_flag = np.asarray(refcal.get('resolved_flag', []), dtype=np.float64)
                use_resolved_tables = bool(
                    not force_lo_hi_smooth_extrap
                    and resolved_bph.ndim == 3
                    and resolved_sbd.ndim == 3
                    and resolved_flag.ndim == 3
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
                pending_refcal_provenance = _refcal_provenance_entry(
                    msfile,
                    sql_lookup_time,
                    cal_src,
                    refcal,
                    refcal_npz_mode,
                    cal_npz_refcal is not None,
                )
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
                                ):
                                    ph = resolved_bph[n, p, band_i]
                                    sb = resolved_sbd[n, p, band_i]
                                    fl = resolved_flag[n, p, band_i]
                                else:
                                    ph = 0.0
                                    sb = np.nan
                                    fl = 1.0
                                flagged = bool(fl) or not np.isfinite(sb)
                                phase_flag_spw[n, p, s] = 1 if flagged else 0
                                phase_rad = 0.0 if (flagged or not np.isfinite(ph)) else float(ph)
                                para_sbd.append(0.0 if flagged else float(sb))
                                selected_npz_models.add('resolved_tables')
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
                                # Single physical in-band delay per (ant, pol): bph_sbd
                                # applies the one active_ns across all bands (HI + LO),
                                # mirroring resolve_bph_sbd_tables. smooth_model (below)
                                # keeps the per-band slope via _smooth_refcal_sbd_for_band.
                                hi_sbd_ns = (float(smooth_sbd[n, p])
                                             if (smooth_sbd is not None and getattr(smooth_sbd, 'ndim', 0) == 2
                                                 and n < smooth_sbd.shape[0] and p < smooth_sbd.shape[1])
                                             else np.nan)
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
                phaflag_ = phase_flag_spw
                if (not use_smooth_bandpass) and (not os.path.exists(caltb_pha)):
                    gencal(vis=msfile, caltable=caltb_pha, caltype='ph', antenna=antennas, pol='X,Y',
                           spw='0~' + str(nspw - 1), parameter=para_pha)
                    tb.open(caltb_pha, nomodify=False)
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

                if use_smooth_bandpass:
                    # Phase-only bandpass: scaffold a per-channel B table (same
                    # pattern as the auto-amp bandpass below), then overwrite CPARAM
                    # with the unit-amplitude smooth phase model sampled at every
                    # science channel. This replaces the per-spw ph table. The smooth
                    # model comes from the NPZ (refcal__model_phase_fine) or, on the
                    # no-NPZ SQL path, from the caltype-15 record mapped onto
                    # model_phase_fine / fine_frequency_ghz above.
                    smb_suffix = '_sql_smooth_bandpass' if use_sql_smooth_bandpass else '_npz_smooth_bandpass'
                    caltb_bphase = dirname + t_ref.isot[:-4].replace(':', '').replace('-', '') + smb_suffix + '.refbphase'
                    if os.path.exists(caltb_bphase):
                        shutil.rmtree(caltb_bphase)
                    bandpass(vis=msfile, caltable=caltb_bphase, solint='inf', refant='eo01', minblperant=0,
                             minsnr=0, bandtype='B', docallib=False)
                    tb.open(caltb_bphase, nomodify=False)
                    for ll in range(nspw):
                        nchan_ll = int(bd_nchan[ll])
                        freq_ghz = np.asarray(chan_freqs_per_spw[ll], dtype=np.float64).reshape(-1) * 1e-9
                        bp_phase, bp_valid = smooth_phase_bandpass_for_freqs(refcal, freq_ghz)
                        cp = np.ones((nant, 2, nchan_ll), dtype=np.complex128)
                        fl = np.zeros((nant, 2, nchan_ll), dtype=np.bool_)
                        a = min(nant, int(bp_phase.shape[0])) if np.asarray(bp_phase).ndim == 3 else 0
                        if a > 0 and bp_phase.shape[1] >= 2 and bp_phase.shape[2] == nchan_ll:
                            cp[:a] = np.where(bp_valid[:a, :2, :], np.exp(1j * bp_phase[:a, :2, :]), 1.0 + 0j)
                            fl[:a] = ~bp_valid[:a, :2, :]
                        else:
                            fl[:] = True
                        # mirror auto-amp: flag the trailing non-solar antennas
                        fl[13:, :, :] = True
                        tb.putcol('CPARAM', np.moveaxis(cp, 0, 2), ll * nant, nant)
                        tb.putcol('FLAG', np.moveaxis(fl, 0, 2), ll * nant, nant)
                        snr = np.full((2, nchan_ll, nant), 100.0)
                        snr[:, :, 13:] = 0.0
                        tb.putcol('SNR', snr, ll * nant, nant)
                        paramerr = tb.getcol('PARAMERR', ll * nant, nant)
                        tb.putcol('PARAMERR', paramerr * 0, ll * nant, nant)
                    tb.close()
                    refcal_gaintables = [caltb_bphase]
                    gaintables.append(caltb_bphase)
                    spwmaps.append([])
                    print("Refcal model gaintables (smooth_bandpass, {0}): {1}".format(
                        'SQL' if use_sql_smooth_bandpass else 'NPZ', os.path.basename(caltb_bphase)))
                else:
                    refcal_gaintables = [caltb_pha]
                    gaintables.append(caltb_pha)
                    spwmaps.append([])
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
                # Drop phacals whose reference refcal time is >30 min after the
                # refcal they were solved against. SQL companion records supersede
                # type-8 phase calibration, so compare against their t_refcal.
                # Legacy type-8 fallback may still use the captured type-8
                # timestamp. Build a keep mask -- `del` on a numpy/Time array
                # raises "ValueError: cannot delete array elements".
                if phacals.any() and len(phacals) > 0:
                    if cal_src == "SQL BPH+SBD":
                        phacal_ref_time = refcal.get('sql_bphsbd_t_refcal') or refcal['timestamp']
                    elif cal_src == 'SQL smooth bandpass (caltype 15)':
                        phacal_ref_time = refcal.get('sql_smb_t_refcal') or refcal['timestamp']
                    else:
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
                if pending_refcal_provenance is not None:
                    pending_refcal_provenance['applied'] = True
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

            if pending_refcal_provenance is not None and refcal_provenance is not None:
                refcal_provenance.append(pending_refcal_provenance)

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
