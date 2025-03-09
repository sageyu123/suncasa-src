##################### generated by xml-casa (v2) from calibeovsa.xml ################
##################### eedbae6635ef6957829bc80cbf8bb7ff ##############################
from __future__ import absolute_import
import numpy
from casatools.typecheck import CasaValidator as _val_ctor
_pc = _val_ctor( )
from casatools.coercetype import coerce as _coerce
from casatools.errors import create_error_string
from .private.task_calibeovsa import calibeovsa as _calibeovsa_t
from casatasks.private.task_logging import start_log as _start_log
from casatasks.private.task_logging import end_log as _end_log
from casatasks.private.task_logging import except_log as _except_log

class _calibeovsa:
    """
    calibeovsa ---- Calibrating EOVSA one or more measurement sets using calibration products in the SQL database.

    
    Calibrating EOVSA one or more measurement sets using calibration products in the SQL database. This task currently only works on pipeline.
    

    --------- parameter descriptions ---------------------------------------------

    vis          input EOVSA (uncalibrated) measurement set(s).
    caltype      Types of calibrations to perform
    caltbdir     Directory to place calibration tables.
    interp       Temporal interpolation for phacal table(s) (nearest, linear or auto)
    docalib      If False, only create the calibration tables but do not perform applycal.
    doflag       If true then perform flagging.
    flagant      Antennas to be flagged. Follow CASA syntax of "antenna".
    flagspw      Spectral windows to be flagged. Follow CASA syntax of "spw". Note this flag only applies to phacal tables.
    doimage      If True, produce a quicklook image after calibration (sunpy must be installed).
    imagedir     directory to place output images. Default current directory.
    antenna      antenna/baselines to be used for imaging. Follow CASA syntax of "antenna".
    timerange    Timerange to be imaged. Follow CASA syntax of "timerange". Default is the entire duration of the ms.
    spw          spectral windows to be imaged. Follow CASA syntax of "spw".
    stokes       stokes to be imaged. Follow CASA syntax of "stokes".
    dosplit      If True, plit the corrected data column as output visibility file.
    outputvis    Name of output visibility file. Default is the name of the first vis file ended with ".corrected.ms".
    doconcat     If True, and if more than one visibility dataset provided, concatenate all into one visibility.
    concatvis    Name of output visibility file. Default is the name of the first + last vis file ended with ".corrected.ms".
    keep_orig_ms Keep the original seperated ms datasets after split?

    --------- examples -----------------------------------------------------------

    
    Calibrating EOVSA one or more measurement sets using calibration products in the SQL database.
    
    Detailed Keyword arguments:
    
    vis -- Name of input EOVSA measurement set dataset(s)
    default: none. Must be supplied
    example: vis = 'IDB20160524000518.ms'
    example: vis = ['IDB20160524000518.ms','IDB20160524000528.ms']
    
    caltype -- list. Type of calibrations to be applied.
    'refpha': reference phase calibration
    'refamp': reference amplitude calibration (not used anymore)
    'phacal': daily phase calibration
    'fluxcal': flux calibration based on total-power measurements
    default value: ['refpha','phacal']
    *** note fluxcal is already implemented in udb_corr when doing importeovsa, should not be used anymore ****
    *** pipeline only uses ['refpha','phacal']
    
    caltbdir -- string. Place to hold calibration tables. Default is current directory. Pipeline should use /data1/eovsa/caltable
    
    interp -- string. How interpolation is done for phacal? 'nearest', 'linear', or 'auto'
    
    docalib -- boolean. Default True. If False, only create the calibration tables but do not perform applycal
    
    doflag -- boolean. Default True. Peforming flags?
    
    flagant -- string. Follow CASA antenna selection syntax. Default '13~15'.

    flagspw -- string. Follow CASA spw selection syntax. Default '0~1'.
    
    doimage -- boolean. Default False. If true, make a quicklook image using the specified time range and specified spw range
    
    imagedir -- string. Directory to place the output image.
    
    antenna -- string. Default '0~12'. Antenna/baselines to be used for imaging. Follow CASA antenna selection syntax.
    
    timerange -- string. Default '' (the whole duration of the visibility data). Follow CASA timerange syntax.
    e.g., '2017/07/11/20:16:00~2017/07/11/20:17:00'
    
    spw -- string. Default '1~3'. Follow CASA spw selection syntax.
    
    stokes -- string. Which stokes for the quicklook image. CASA syntax. Default 'XX'
    
    dosplit -- boolean. Split the corrected data column?
    
    outputvis -- string. Output visibility file after split
    
    doconcat -- boolean. If more than one visibility dataset provided, concatenate all into one or make separate outputs if True
    
    concatvis -- string. Output visibility file after concatenation
    
    keep_orig_ms -- boolean. Default True. Inherited from suncasa.eovsa.concateovsa.
    Keep the original seperated ms datasets after concatenation?
    
    


    """

    _info_group_ = """Calibration"""
    _info_desc_ = """Calibrating EOVSA one or more measurement sets using calibration products in the SQL database."""

    def __call__( self, vis='', caltype=[  ], caltbdir='', interp='nearest', docalib=True, doflag=True, flagant='13~15', flagspw='0~1', doimage=False, imagedir='.', antenna='0~12', timerange='', spw='1~3', stokes='XX', dosplit=False, outputvis='', doconcat=False, concatvis='', keep_orig_ms=True ):
        schema = {'vis': {'anyof': [{'type': 'cStr', 'coerce': _coerce.to_str}, {'type': 'cStrVec', 'coerce': [_coerce.to_list,_coerce.to_strvec]}]}, 'caltype': {'anyof': [{'type': 'cStr', 'coerce': _coerce.to_str}, {'type': 'cStrVec', 'coerce': [_coerce.to_list,_coerce.to_strvec]}]}, 'caltbdir': {'type': 'cStr', 'coerce': _coerce.to_str}, 'interp': {'type': 'cStr', 'coerce': _coerce.to_str}, 'docalib': {'type': 'cBool'}, 'doflag': {'type': 'cBool'}, 'flagant': {'type': 'cStr', 'coerce': _coerce.to_str}, 'flagspw': {'type': 'cStr', 'coerce': _coerce.to_str}, 'doimage': {'type': 'cBool'}, 'imagedir': {'type': 'cStr', 'coerce': _coerce.to_str}, 'antenna': {'type': 'cStr', 'coerce': _coerce.to_str}, 'timerange': {'type': 'cStr', 'coerce': _coerce.to_str}, 'spw': {'type': 'cStr', 'coerce': _coerce.to_str}, 'stokes': {'type': 'cStr', 'coerce': _coerce.to_str}, 'dosplit': {'type': 'cBool'}, 'outputvis': {'anyof': [{'type': 'cStr', 'coerce': _coerce.to_str}, {'type': 'cStrVec', 'coerce': [_coerce.to_list,_coerce.to_strvec]}]}, 'doconcat': {'type': 'cBool'}, 'concatvis': {'type': 'cStr', 'coerce': _coerce.to_str}, 'keep_orig_ms': {'type': 'cBool'}}
        doc = {'vis': vis, 'caltype': caltype, 'caltbdir': caltbdir, 'interp': interp, 'docalib': docalib, 'doflag': doflag, 'flagant': flagant, 'flagspw': flagspw, 'doimage': doimage, 'imagedir': imagedir, 'antenna': antenna, 'timerange': timerange, 'spw': spw, 'stokes': stokes, 'dosplit': dosplit, 'outputvis': outputvis, 'doconcat': doconcat, 'concatvis': concatvis, 'keep_orig_ms': keep_orig_ms}
        assert _pc.validate(doc,schema), create_error_string(_pc.errors)
        _logging_state_ = _start_log( 'calibeovsa', [ 'vis=' + repr(_pc.document['vis']), 'caltype=' + repr(_pc.document['caltype']), 'caltbdir=' + repr(_pc.document['caltbdir']), 'interp=' + repr(_pc.document['interp']), 'docalib=' + repr(_pc.document['docalib']), 'doflag=' + repr(_pc.document['doflag']), 'flagant=' + repr(_pc.document['flagant']), 'flagspw=' + repr(_pc.document['flagspw']), 'doimage=' + repr(_pc.document['doimage']), 'imagedir=' + repr(_pc.document['imagedir']), 'antenna=' + repr(_pc.document['antenna']), 'timerange=' + repr(_pc.document['timerange']), 'spw=' + repr(_pc.document['spw']), 'stokes=' + repr(_pc.document['stokes']), 'dosplit=' + repr(_pc.document['dosplit']), 'outputvis=' + repr(_pc.document['outputvis']), 'doconcat=' + repr(_pc.document['doconcat']), 'concatvis=' + repr(_pc.document['concatvis']), 'keep_orig_ms=' + repr(_pc.document['keep_orig_ms']) ] )
        task_result = None
        try:
            task_result = _calibeovsa_t( _pc.document['vis'], _pc.document['caltype'], _pc.document['caltbdir'], _pc.document['interp'], _pc.document['docalib'], _pc.document['doflag'], _pc.document['flagant'], _pc.document['flagspw'], _pc.document['doimage'], _pc.document['imagedir'], _pc.document['antenna'], _pc.document['timerange'], _pc.document['spw'], _pc.document['stokes'], _pc.document['dosplit'], _pc.document['outputvis'], _pc.document['doconcat'], _pc.document['concatvis'], _pc.document['keep_orig_ms'] )
        except Exception as exc:
            _except_log('calibeovsa', exc)
            raise
        finally:
            task_result = _end_log( _logging_state_, 'calibeovsa', task_result )
        return task_result

calibeovsa = _calibeovsa( )

