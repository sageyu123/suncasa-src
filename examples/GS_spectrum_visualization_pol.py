import lmfit
import matplotlib.pyplot as plt
import numpy as np
import sunpy
from pygsfit.utils import gstools
import os
import math
# import GScodes  # initialization library - located either in the current directory or in the system path
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors

# libname = os.path.join(os.path.dirname(GScodes.__file__),
#                        'MWTransferArr.so')  # name of the executable library - located where Python can find it

libname = os.path.join(os.path.dirname(os.path.realpath(gstools.__file__)),
                       '../binaries/MWTransferArr_arm64.so')

def mwspec2min_1src(params, freqghz, toTb=True, arcsec2cm=0.725e8, returnpol=False):
    # params are defined by lmfit.Paramters()
    '''
    params: parameters defined by lmfit.Paramters()
    freqghz: frequencies in GHz
    ssz: pixel size in arcsec
    tb: reference brightness temperature in K
    tb_err: uncertainties of reference brightness temperature in K
    '''

    from scipy import interpolate
    GET_MW = gstools.initGET_MW(libname)  # load the library

    ssz = float(params['ssz'].value)  # # source area in arcsec^2
    depth = float(params['depth'].value)  # total source depth in arcsec
    Bmag = float(params['Bmag'].value)  # magnetic field strength in G
    Tth = float(params['Tth'].value)  # thermal temperature in MK
    nth = float(params['nth'].value)  # thermal density in 1e10 cm^{-3}
    nrlh = 10. ** float(params['lognrlh'].value)  # total nonthermal density above 0.1 MeV
    delta = float(params['delta'].value)  # powerlaw index
    theta = float(params['theta'].value)  # viewing angle in degrees
    Emin = float(params['Emin'].value)  # low energy cutoff of nonthermal electrons in MeV
    Emax = float(params['Emax'].value)  # high energy cutoff of nonthermal electrons in MeV
    E_hi = 0.1
    nrl = nrlh * (Emin ** (1. - delta) - Emax * (1. - delta)) / (E_hi ** (1. - delta) - Emax ** (1. - delta))

    Nf = 100  # number of frequencies
    NSteps = 1  # number of nodes along the line-of-sight

    N_E = 15  # number of energy nodes
    N_mu = 15  # number of pitch-angle nodes

    Lparms = np.zeros(11, dtype='int32')  # array of dimensions etc.
    Lparms[0] = NSteps
    Lparms[1] = Nf
    Lparms[2] = N_E
    Lparms[3] = N_mu

    Rparms = np.zeros(5, dtype='double')  # array of global floating-point parameters
    Rparms[0] = ssz * arcsec2cm ** 2  # Area, cm^2
    # Rparms[0] = 1e20  # area, cm^2
    Rparms[1] = 1e9  # starting frequency to calculate spectrum, Hz
    Rparms[2] = 0.02  # logarithmic step in frequency
    Rparms[3] = 0  # f^C
    Rparms[4] = 0  # f^WH

    ParmLocal = np.zeros(24, dtype='double')  # array of voxel parameters - for a single voxel
    ParmLocal[0] = depth * arcsec2cm / NSteps  # voxel depth, cm
    ParmLocal[1] = Tth * 1e6  # T_0, K
    ParmLocal[2] = nth * 1e10  # n_0 - thermal electron density, cm^{-3}
    ParmLocal[3] = Bmag  # B - magnetic field, G

    Parms = np.zeros((24, NSteps), dtype='double', order='F')  # 2D array of input parameters - for multiple voxels
    for i in range(NSteps):
        Parms[:, i] = ParmLocal  # most of the parameters are the same in all voxels
        # if NSteps > 1:
        #     Parms[4, i] = 50.0 + 30.0 * i / (NSteps - 1)  # the viewing angle varies from 50 to 80 degrees along the LOS
        # else:
        #     Parms[4, i] = 50.0  # the viewing angle varies from 50 to 80 degrees along the LOS
        Parms[4, i] = theta

    # parameters of the electron distribution function
    n_b = nrl  # n_b - nonthermal electron density, cm^{-3}
    mu_c = np.cos(np.pi * 70 / 180)  # loss-cone boundary
    dmu_c = 0.2  # Delta_mu

    E_arr = np.logspace(np.log10(Emin), np.log10(Emax), N_E, dtype='double')  # energy grid (logarithmically spaced)
    mu_arr = np.linspace(-1.0, 1.0, N_mu, dtype='double')  # pitch-angle grid

    f0 = np.zeros((N_E, N_mu), dtype='double')  # 2D distribution function array - for a single voxel

    # computing the distribution function (equivalent to PLW & GLC)
    A = n_b / (2.0 * np.pi) * (delta - 1.0) / (Emin ** (1.0 - delta) - Emax ** (1.0 - delta))
    B = 0.5 / (mu_c + dmu_c * np.sqrt(np.pi) / 2 * math.erf((1.0 - mu_c) / dmu_c))
    for i in range(N_E):
        for j in range(N_mu):
            amu = abs(mu_arr[j])
            f0[i, j] = A * B * E_arr[i] ** (-delta) * (1.0 if amu < mu_c else np.exp(-((amu - mu_c) / dmu_c) ** 2))

    f_arr = np.zeros((N_E, N_mu, NSteps), dtype='double',
                     order='F')  # 3D distribution function array - for multiple voxels
    for k in range(NSteps):
        f_arr[:, :, k] = f0  # electron distribution function is the same in all voxels

    RL = np.zeros((7, Nf), dtype='double', order='F')  # input/output array

    # calculating the emission for array distribution (array -> on)
    res = GET_MW(Lparms, Rparms, Parms, E_arr, mu_arr, f_arr, RL)

    if res:
        # retrieving the results
        f = RL[0]
        I_L = RL[5]
        I_R = RL[6]

        # if showplt:
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(1, 1)
        #     ax.plot(f, I_L + I_R)
        #     ax.set_xscale('log')
        #     ax.set_yscale('log')
        #     ax.set_title('Total intensity (array)')
        #     ax.set_xlabel('Frequency, GHz')
        #     ax.set_ylabel('Intensity, sfu')

        logf = np.log10(f)
        logfreqghz = np.log10(freqghz)

        if returnpol:
            flx_model = I_L
            flx_model = np.nan_to_num(flx_model) + 1e-11
            logflx_model = np.log10(flx_model)
            interpfunc_L = interpolate.interp1d(logf, logflx_model, kind='linear')
            logmflx = interpfunc_L(logfreqghz)
            mflx_L = 10. ** logmflx

            flx_model = I_R
            flx_model = np.nan_to_num(flx_model) + 1e-11
            logflx_model = np.log10(flx_model)
            interpfunc_R = interpolate.interp1d(logf, logflx_model, kind='linear')
            logmflx = interpfunc_R(logfreqghz)
            mflx_R = 10. ** logmflx
            if toTb:
                tb_L = gstools.sfu2tb(np.array(freqghz) * 1.e9, mflx_L, ssz)
                tb_R = gstools.sfu2tb(np.array(freqghz) * 1.e9, mflx_R, ssz)
                return tb_R, tb_L
            else:
                return mflx_R, mflx_L
        else:
            flx_model = I_L + I_R
            flx_model = np.nan_to_num(flx_model) + 1e-11
            logflx_model = np.log10(flx_model)
            interpfunc = interpolate.interp1d(logf, logflx_model, kind='linear')
            logmflx = interpfunc(logfreqghz)
            mflx = 10. ** logmflx
            if toTb:
                tb = gstools.sfu2tb(np.array(freqghz) * 1.e9, mflx, ssz)
                return tb
            else:
                return mflx


    else:
        print("Calculation error!")


freqghz = np.logspace(np.log10(1), np.log10(18), 100)
E_hi = 0.1
Tth = 10.  # Thermal temperature in MK

nfrm = 40
bmag = np.linspace(150., 300., nfrm)
nrl = np.linspace(1.0e7, 2.0e7, nfrm)
delta = np.linspace(4.0, 2.5, nfrm)
Emin = np.linspace(0.02, 0.04, nfrm)
Emax = np.linspace(1.0, 3.0, nfrm)
nth = np.linspace(0.1, 1, nfrm)
theta = np.linspace(75., 45., nfrm)

paramsdict = {}
paramsdict['Bmag'] = np.hstack([bmag, bmag[:-1][::-1]])
paramsdict['nrl'] = np.hstack([nrl, nrl[:-1][::-1]])
paramsdict['delta'] = np.hstack([delta, delta[:-1][::-1]])
paramsdict['Emin'] = np.hstack([Emin, Emin[:-1][::-1]])
paramsdict['Emax'] = np.hstack([Emax, Emax[:-1][::-1]])
paramsdict['nth'] = np.hstack([nth, nth[:-1][::-1]])
paramsdict['theta'] = np.hstack([theta, theta[:-1][::-1]])
paramskeys = paramsdict.keys()
delta = 4.0
Emin = 0.02
Emax = 1.0
nrl = 1.0e7
nrlh = nrl * (E_hi ** (1. - delta) - Emax ** (1. - delta)) / (Emin ** (1. - delta) - Emax ** (1. - delta)) + 1e-8
lognrlh = np.log10(nrlh)

import time

# gs_kw = dict(height_ratios=[1,1],)
# fig, axs = plt.subplots(figsize=(4.5, 5), gridspec_kw=gs_kw, nrows=8)

fig = plt.figure(constrained_layout=True, figsize=(6, 6))
widths = [0.2, 0.8, 1]
heights = [1] * 7 + [35]
spec = fig.add_gridspec(ncols=3, nrows=8, width_ratios=widths,
                        height_ratios=heights)

caxs = []
txtaxs1 = []
txtaxs2 = []
for row in range(7):
    txtaxs1.append(fig.add_subplot(spec[row, 0]))
    txtaxs2.append(fig.add_subplot(spec[row, 1]))
    caxs.append(fig.add_subplot(spec[row, 2]))

ax = fig.add_subplot(spec[-1, :])

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_facecolor('#CCC')
ax.set_xlim([1, 18])
ax.set_ylim([1e6, 1e9])
ax.set_xticks([1, 5, 10])
ax.set_xticklabels([1, 5, 10])
ax.set_xticks([1, 5, 10])
# ax.set_yticks([])
# ax.set_yticks([0.01, 0.1, 1, 10, 100, 1000])
ax.set_ylabel('Brightness Temperature [M]')
ax.set_xlabel('Frequency [GHz]')
# fig.tight_layout()

# line_L = ax.plot([], [], ':k')
# lines_L = []
colormaps = ['Reds', 'Purples', 'Greens', 'Blues', 'Oranges', 'bone_r', 'pink_r']
# colormaps = ['sdoaia131', 'sdoaia171', 'sdoaia335', 'sdoaia211', 'sdoaia94', 'sdoaia304', 'sdoaia193']
figid = 1
k2text = {'Bmag': r'B',
          'nrl': r'n$_{nth}$',
          'delta': r'$\delta$',
          'Emin': r'E$_{min}$',
          'Emax': r'E$_{max}$',
          'nth': r'n$_{th}$',
          'theta': r'$\theta$'}

k2plaintext = {'Bmag': r'Magnetic Field Strength',
               'nrl': r'Number Density of Nonthermal Electrons',
               'delta': r'Power-law Index',
               'Emin': r'Low-energy cutoff',
               'Emax': r'High-energy cutoff',
               'nth': r'Number Density of Thermal Electrons',
               'theta': r'Viewing Angle'}

k2unit = {'Bmag': r' G',
          'nrl': r' cm$^{-3}$',
          'delta': r'',
          'Emin': r' MeV',
          'Emax': r' MeV',
          'nth': r' cm$^{-3}$',
          'theta': r'$^\circ$'}

k2vformat = {'Bmag': r'= {:.1f}',
             'nrl': r'= {:.2e}',
             'delta': r'= {:.2f}',
             'Emin': r'= {:.4f}',
             'Emax': r'= {:.1f}',
             'nth': r'= {:.2e}',
             'theta': r'= {:.1f}'}

vtexts = []
ktexts = []
vlines = []
plaintxt = ax.text(0.015, 0.12, '', ha='left', va='top',
                   transform=ax.transAxes, color='w', backgroundcolor='none', fontweight='bold', fontsize=10, wrap=True)
plaintxt._get_wrap_line_width = lambda: 400.
for kidx, k in enumerate(tqdm(['Bmag', 'nrl', 'delta', 'Emin', 'Emax', 'nth', 'theta'])):
    v = paramsdict[k]
    vmin = np.nanmin(v)
    vmax = np.nanmax(v)
    cmap = plt.get_cmap(colormaps[kidx])
    p_color = cmap(np.log10((v - vmin) / (vmax - vmin) * 7 + 3))

    if k == 'nth':
        vdisplay = v[0] * 1e10
        vmindisplay = vmin * 1e10
        vmaxdisplay = vmax * 1e10
    else:
        vdisplay = v[0]
        vmindisplay = vmin
        vmaxdisplay = vmax

    cax = caxs[kidx]
    cb = colorbar.ColorbarBase(cax, norm=colors.Normalize(vmin=vmindisplay, vmax=vmaxdisplay), cmap=cmap,
                               orientation='horizontal',
                               ticks=[], format='%4.1f', alpha=1)

    # plt.text(0.5, 1.05, 'MW', ha='center', va='bottom', transform=cax.transAxes, color='k', fontweight='normal')
    # plt.text(0.5, 1.01, '[GHz]', ha='center', va='bottom', transform=cax.transAxes, color='k',
    #          fontweight='normal')

    cax.yaxis.set_visible(False)
    cax.xaxis.set_visible(False)
    tax = txtaxs1[kidx]
    tax.axis('off')
    ktexts.append(tax.text(0.0, 0.5, k2text[k], ha='left', va='center',
                           transform=tax.transAxes))
    tax = txtaxs2[kidx]
    tax.axis('off')
    vtexts.append(tax.text(0.0, 0.5, k2vformat[k].format(vdisplay) + k2unit[k], ha='left', va='center',
                           transform=tax.transAxes))
    vlines.append(cax.axvline(vdisplay, lw=8, color='gray'))
    # cax.tick_params(axis="y", pad=-20., length=0, colors='k', labelsize=7)
    # cax.axvline(vmin, ymin=1.0, ymax=1.2, color='k', clip_on=False)
    # cax.axvline(vmax, ymin=1.0, ymax=1.2, color='k', clip_on=False)
    cax.text(-0.1, 0.5, k2vformat[k].format(vmindisplay).replace('= ', ''), fontsize=9, transform=cax.transAxes,
             va='center', ha='right')
    cax.text(1.1, 0.5, k2vformat[k].format(vmaxdisplay).replace('= ', ''), fontsize=9, transform=cax.transAxes,
             va='center', ha='left')

lines_R = []
lines_L = []
for kidx, k in enumerate(tqdm(['Bmag', 'nrl', 'delta', 'Emin', 'Emax', 'nth', 'theta'])):
    # if kidx > 0: break
    # if figid > 1: break

    line_R = ax.plot([], [], lw=1.0)
    # k = 'Bmag'
    v = paramsdict[k]
    params = lmfit.Parameters()
    params.add('lnf', value=-2., vary=False)  # ln of fractional error adjustment
    params.add('ssz', value=15. ** 2, vary=False)  # pixel size in arcsec
    params.add('depth', value=10., min=5.0, max=300., vary=False)  # Column depth, 1e8 cm (1 Mm)
    params.add('Bmag', value=150., min=150., max=300., vary=False)  # Magnetic field strength in G
    params.add('Tth', value=10., min=5, max=30., vary=False)  # Thermal temperature in MK
    params.add('nth', value=0.1, min=0.1, max=1.0, vary=False)  # Thermal density in 1e10 cm^{-3}
    params.add('lognrlh', value=np.log10(nrlh), min=1.0, max=10.0, vary=False)  # Non-thermal density in cm^{-3}
    params.add('delta', value=4.0, min=2.5, max=4.0, vary=False)  # Power law index for nonthermal electrons
    params.add('theta', value=75., min=45., max=75., vary=False)  # theta
    params.add('Emin', value=0.02, min=0.001, max=0.05, vary=False)  # Emin
    params.add('Emax', value=1.0, min=1.0, max=3.0, vary=False)  # Emax
    # palpha = np.logspace(-1., 0, len(v))
    # palpha = np.geomspace(0.05, 1, len(v))
    # p_color = plt.get_cmap('gray_r')(np.linspace(0.2,1,len(v)))
    vmin = np.nanmin(v)
    vmax = np.nanmax(v)
    cmap = plt.get_cmap(colormaps[kidx])
    p_color = cmap(np.log10((v - vmin) / (vmax - vmin) * 7 + 3))
    vtexts[kidx].set_fontweight('bold')
    ktexts[kidx].set_fontweight('bold')
    plaintxt.set_text(r'Changing {} {}'.format(k2plaintext[k], k2text[k]))
    # plaintxt.set_color(cmap(0.8))
    # plaintxt.set_backgroundcolor(cmap(0.8))
    ctxt = list(cmap(0.8))
    ctxt[-1] = 0.75
    plaintxt.set_bbox(dict(facecolor=ctxt, edgecolor='none'))
    for pidx, pvalue in enumerate(v):
        if pidx > 0:
            palpha = np.geomspace(0.05, 1.0, len(lines_R))
            # lines_R[-1][0].set_alpha(palpha[pidx])
            for lidx, l in enumerate(lines_R):
                l[0].set_alpha(palpha[lidx])
            # lines_R[-1][0].set_alpha(0.2)
            # lines_R[-1][0].set_color(p_color[pidx])
        if k == 'nrl':
            nrlh = pvalue * (
                    E_hi ** (1. - params['delta'].value) - params['Emax'].value ** (1. - params['delta'].value)) / (
                           params['Emin'].value ** (1. - params['delta'].value) - params['Emax'].value ** (
                           1. - params['delta'].value)) + 1e-8
            params['lognrlh'].value = np.log10(nrlh)
        elif k == 'Emin':
            params[k].value = pvalue
            vnrl = nrl * (pvalue ** (1. - delta) - Emax ** (1. - delta)) / (Emin ** (1. - delta) - Emax ** (1. - delta))
            vtexts[1].set_text(k2vformat['nrl'].format(vnrl) + k2unit['nrl'])
            vlines[1].set_data(([vnrl] * 2, [0, 1]))
        else:
            params[k].value = pvalue
        if k == 'nth':
            vdisplay = pvalue * 1e10
        else:
            vdisplay = pvalue
        # print(k, vdisplay)
        vtexts[kidx].set_text(k2vformat[k].format(vdisplay) + k2unit[k])
        vlines[kidx].set_data(([vdisplay] * 2, [0, 1]))
        # tb_R, tb_L  = mwspec2min_1src(params, freqghz)
        # lines_R.append(ax.plot(freqghz, tb_R,lw=1.0))
        # lines_L.append(ax.plot(freqghz, tb_L, ':k'))
        tb_R, tb_L = mwspec2min_1src(params, freqghz, returnpol=True)
        lines_R.append(ax.plot(freqghz, tb_R, lw=1.0, c=p_color[pidx]))
        lines_L.append(ax.plot(freqghz, tb_L, ':', lw=1.0, c=p_color[pidx]))
        if figid == 1:
            spec.tight_layout(fig)
            fig.savefig('fig.GSspec.{:04d}.png'.format(figid), dpi=150)
        fig.savefig('fig.GSspec.{:04d}.png'.format(figid), dpi=150)
        figid += 1
    ktexts[kidx].set_fontweight('normal')
    vtexts[kidx].set_fontweight('normal')
    # time.sleep(1)

from suncasa.utils import DButil

DButil.img2movie('fig.GSspec.', outname='fig.GSspec', overwrite=True, fps=24)
os.system('rm -rf fig.GSspec.????.png')
# gt.set_params(params=params)
#
# gt.fit()
