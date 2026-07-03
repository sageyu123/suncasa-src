"""
Standalone debug/test script for the EOVSA synoptic imaging pipeline (WSClean version).

This script sets up test parameters and runs pipeline_run() for a specific date.
It was extracted from the debug_mode block that was previously embedded in
eovsa_synoptic_imaging_pipeline_wsclean.py.

Usage:
    python debug_pipeline_wsclean.py
"""
import os
import numpy as np
from datetime import datetime
from glob import glob

from eovsapy.util import Time
from eovsapy.dump_tsys import findfiles
from suncasa.suncasatasks import importeovsa
from suncasa.suncasatasks.private.task_calibeovsa import calibeovsa
from suncasa.eovsa.update_log import EOVSA15_UPGRADE_DATE
from suncasa.eovsa.eovsa_synoptic_imaging_pipeline_wsclean import pipeline_run

# ---- Configuration ----
udbdir = '/data1/eovsa/fits/UDB/'
workdir = './'
slfcaltbdir = None
imgoutdir = './'
pols = 'XX'
overwrite = True
overwrite_caltb = True
hanning = False
do_sbdcal = False

# Choose a date to process
datein = datetime(2021, 11, 24, 20, 0, 0)

# ---- Prepare MS file ----
trange = Time(datein)
if trange.mjd == np.fix(trange.mjd):
    trange = Time(trange.mjd + 0.5, format='mjd')

tdatetime = trange.to_datetime()
dhr = trange.LocalTime.utcoffset().total_seconds() / 60 / 60 / 24
btime = Time(np.fix(trange.mjd + dhr) - dhr, format='mjd')
etime = Time(btime.mjd + 1, format='mjd')
trange = Time([btime, etime])
print('Selected timerange in UTC: ', trange.iso)

inpath = '{}{}/'.format(udbdir, tdatetime.strftime("%Y"))
outpath = './'
vis = os.path.join(outpath, 'UDB' + tdatetime.strftime("%Y%m%d") + '.ms')

if not os.path.exists(vis):
    sclist = findfiles(trange, projid='NormalObserving', srcid='Sun')
    udbfilelist = sclist['scanlist']
    udbfilelist = [os.path.basename(ll) for ll in udbfilelist]
    udbfilelist_set = set(udbfilelist)
    msfiles = []
    msfiles = udbfilelist_set.intersection(msfiles)
    filelist = udbfilelist_set - msfiles
    filelist = sorted(list(filelist))
    ncpu = 1
    importeovsa(idbfiles=[inpath + ll for ll in filelist], ncpu=ncpu, timebin="0s", width=1,
                visprefix=outpath, nocreatms=False,
                doconcat=False, modelms="", doscaling=False, keep_nsclms=False, udb_corr=True)

    msfiles = [os.path.basename(ll).split('.')[0] for ll in glob('{}UDB*.ms'.format(outpath)) if
               ll.endswith('.ms') or ll.endswith('.ms.tar.gz')]
    udbfilelist_set = set(udbfilelist)
    msfiles = udbfilelist_set.intersection(msfiles)
    filelist = udbfilelist_set - msfiles
    filelist = sorted(list(filelist))

    invis = [outpath + ll + '.ms' for ll in sorted(list(msfiles))]
    vis_out = os.path.join(os.path.dirname(invis[0]), os.path.basename(invis[0])[:11] + '.ms')
    interp = 'auto'
    flagant = '13~15' if btime.mjd >= EOVSA15_UPGRADE_DATE.mjd else '15'
    vis = calibeovsa(invis, caltype=['refpha', 'phacal'], caltbdir='./', interp=interp,
                     doflag=True,
                     flagant=flagant,
                     doimage=False, doconcat=True,
                     concatvis=vis_out, keep_orig_ms=True)
else:
    print(f'Input MS file {vis} already exists. Skipping import from UDB.')

# ---- Run the pipeline ----
pipeline_run(
    vis=vis,
    workdir=workdir,
    slfcaltbdir=slfcaltbdir,
    imgoutdir=imgoutdir,
    pols=pols,
    overwrite=overwrite,
    overwrite_caltb=overwrite_caltb,
    hanning=hanning,
    do_sbdcal=do_sbdcal,
)
