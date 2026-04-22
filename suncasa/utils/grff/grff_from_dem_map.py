### running env loadpyenv3.7

import copy
import ctypes
import glob
import os
import pickle
import platform
import time
import warnings
from datetime import datetime
from datetime import timedelta
from itertools import chain
import numpy.ma as ma
import astropy.time as atime
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import sunpy.map
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from aiapy.calibrate import degradation
from aiapy.calibrate import register, update_pointing
from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from demregpy import dn2dem
from numpy.ctypeslib import ndpointer
from sunpy import io as sio
from tqdm import tqdm

# import sys

# sys.path.insert(-1, '/Users/fisher/Library/Mobile Documents/com~apple~CloudDocs/work/research_project/LWS_FST_2022')
warnings.simplefilter('ignore')
# matplotlib.rcParams['font.size'] = 16

start_tim = time.time()


def getbeam(fitsfile, showplt=False):
    ## calculate the beam size in pixel units
    hdulist = fits.open(fitsfile)
    hdu = hdulist[0]
    header = hdu.header
    fwhm2sigma = 2 * np.sqrt(2 * np.log(2))
    bmaj = header['bmaj'] * 3600 / header['cdelt1']  ## unit in pixel
    bmin = header['bmin'] * 3600 / header['cdelt1']  ## unit in pixel
    bpa = np.deg2rad(90 - header['bpa'])  ## unit in rad
    hdulist.close()
    bmkernel = Gaussian2DKernel(x_stddev=bmaj / fwhm2sigma, y_stddev=bmin / fwhm2sigma, theta=bpa)
    if showplt:
        plt.figure()
        plt.imshow(bmkernel, origin='lower')
        plt.contour(np.array(bmkernel.array), levels=[0.5 * np.nanmax(bmkernel)], colors='red')
        plt.show()
    return bmkernel


def convolve_eovsabeam(res_tbmap, eofitsfiles, tbmapcovfile='tbmaps_convolved.p'):
    tbmap = res_tbmap['tbmap']
    tbmap_convolved = np.zeros_like(tbmap)
    bms = []
    for s, eofits in enumerate(tqdm(eofitsfiles, desc='beam convovling eovsa sim images')):
        bmkernel = getbeam(eofits, showplt=False)
        tbmap_convolved[s] = convolve(tbmap[s], bmkernel)
        bms.append(bmkernel)
    res_dict = {'freqghz': res_tbmap['freqghz'], 'tbmap': tbmap_convolved, 'bms': bms}
    with open(tbmapcovfile, 'wb') as pf:
        pickle.dump(res_dict, pf)
    return res_dict


def make_sub_map(cur_map, fov=None, ref_map=None):
    if not ref_map:
        ref_map = cur_map
    bole = SkyCoord(fov[0][0] * u.arcsec, fov[0][1] * u.arcsec, frame=ref_map.coordinate_frame)
    tori = SkyCoord(fov[1][0] * u.arcsec, fov[1][1] * u.arcsec, frame=ref_map.coordinate_frame)
    sub_map = cur_map.submap(bottom_left=bole, top_right=tori)
    return sub_map


def get_pixel(cur_map, x, y):
    coor = SkyCoord(x * u.arcsec, y * u.arcsec, frame=cur_map.coordinate_frame)
    pix_coor = cur_map.world_to_pixel(coor)
    x = int(pix_coor.x.value)
    y = int(pix_coor.y.value)
    return [x, y]


# dem in 2d
def cal_dem_map(aiafitsfiles, outfile='demreg.p', fov=[], scalefactor=1.0, overwrite=False):
    # Can either get the example data via fido/vso or adapt the example to your own AIA data set

    # Only want the 6 coronal channels - this might also download 304A
    # wvsrch=a.Wavelength(94*u.angstrom, 335*u.angstrom)
    #
    # result = Fido.search(a.Time('2011-01-27T00:00:00', '2011-01-27T00:00:09'), a.Instrument("aia"), wvsrch)
    # file_list = Fido.fetch(result,path='/Volumes/Data/20220511/dem_test/20110127')
    # aiafitsfiles = glob.glob('/Volumes/Data/20220511/dem_test/20110127/*aia*.fits')
    dodemcalc = True
    if os.path.exists(outfile):
        if overwrite == False:
            dodemcalc = False

    aia_tresp_en_file = '/Users/fisher/Library/Mobile Documents/com~apple~CloudDocs/work/research_project/LWS_FST_2022/grff_python/aia_tresp_en.dat'
    # fov = [[-1100, -400], [-900, -200]]
    # time_string = '2011-01-27T00:00:00.000'
    # hdulist = fits.open(aiafitsfiles)
    # hdulist.close()
    kw_list = ['Z.131.', 'Z.171.', 'Z.211.', 'Z.94.', 'Z.335.', 'Z.193.']
    iter_st = time.time()
    amaps = sunpy.map.Map(aiafitsfiles)
    time_string = amaps[0].meta['t_obs']
    print('cur_tiem is ', time_string)
    # Get the wavelengths of the maps and get index of sort for this list of maps
    wvn0 = [m.meta['wavelnth'] for m in amaps]
    # print(wvn0)
    srt_id = sorted(range(len(wvn0)), key=wvn0.__getitem__)
    print('sorted_list is : ', srt_id)
    amaps = [amaps[i] for i in srt_id]
    amap171 = amaps[2]
    # print([m.meta['wavelnth'] for m in amaps])
    channels = [94, 131, 171, 193, 211, 335] * u.angstrom
    # cctime = atime.Time(time_string, scale='utc')
    cctime = atime.Time(time_string, format='isot')
    nc = len(channels)
    degs = np.empty(nc)
    for i in np.arange(nc):
        degs[i] = degradation(channels[i], cctime, calibration_version=10)
    if scalefactor == 1.0:
        aprep = []
        for m in amaps:
            m_temp = update_pointing(m)
            aprep.append(register(m_temp))
    else:
        aprep = amaps
    # Get the durations for the DN/px/s normalisation and
    # wavenlength to check the order - should already be sorted above
    wvn = [m.meta['wavelnth'] for m in aprep]
    durs = [m.meta['exptime'] for m in aprep]
    # Convert to numpy arrays as make things easier later
    durs = np.array(durs)
    print(durs)
    wvn = np.array(wvn)
    worder = np.argsort(wvn)
    print(worder)

    trin = io.readsav(aia_tresp_en_file)
    tresp_logt = np.array(trin['logt'])
    nt = len(tresp_logt)
    nf = len(trin['tr'][:])
    trmatrix = np.zeros((nt, nf))
    for i in range(0, nf):
        trmatrix[:, i] = trin['tr'][i]
    gains = np.array([18.3, 17.6, 17.7, 18.3, 18.3, 17.6])
    dn2ph = gains * np.array([94, 131, 171, 193, 211, 335]) / 3397.
    rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18])
    temps = np.logspace(5.7, 7.6, num=42)
    # Temperature bin mid-points for DEM plotting
    mlogt = ([np.mean([(np.log10(temps[i])), np.log10((temps[i + 1]))]) \
              for i in np.arange(0, len(temps) - 1)])

    # --------------------------------------------------------------------------------------------

    if len(fov) > 0:
        tmp_aprep = []
        for api, cap in enumerate(aprep):
            tmp_aprep.append(make_sub_map(cur_map=cap, fov=fov))
        aprep = tmp_aprep

    if dodemcalc:
        cmap_shape = aprep[0].data.shape
        data_cube = np.zeros((cmap_shape[0], cmap_shape[1], 6))
        num_pix = (1 / scalefactor) ** 2

        for mi, m in enumerate(aprep):
            data_cube[:, :, mi] = m.data / degs[mi] / durs[mi]
            durs_cube = np.broadcast_to(durs, (cmap_shape[0], cmap_shape[1], 6))
            dn2ph_cube = np.broadcast_to(dn2ph, (cmap_shape[0], cmap_shape[1], 6))
            degs_cube = np.broadcast_to(degs, (cmap_shape[0], cmap_shape[1], 6))
            rdnse_cube = np.broadcast_to(rdnse, (cmap_shape[0], cmap_shape[1], 6))

        shotnoise = (dn2ph_cube * data_cube * num_pix) ** 0.5 / dn2ph_cube / num_pix / degs_cube
        edata_cube = (shotnoise ** 2 + rdnse_cube ** 2) ** 0.5 / durs_cube
        pe_time = time.time()
        print('it take to {} to prep in iter_time'.format(pe_time - iter_st))
        print('{} * {}, {} pixels to calculate in total'.format(data_cube.shape[0], data_cube.shape[1],
                                                                data_cube.shape[0] * data_cube.shape[1]))
        dem, edem, elogt, chisq, dn_reg = dn2dem(data_cube, edata_cube, trmatrix, tresp_logt, temps)
        res_dict = {'dem': dem,
                    'edem': edem,
                    'elogt': elogt,
                    'chisq': chisq,
                    'dn_reg': dn_reg,
                    'temps': temps,
                    'fov': fov}
        with open(outfile, 'wb') as pf:
            pickle.dump(res_dict, pf)
    else:
        with open(outfile, 'rb') as pf:
            res_dict = pickle.load(pf, encoding='latin1')
        dem = res_dict['dem']
        # edem = res_dict['edem']
        # elogt = res_dict['elogt']
        # chisq = res_dict['chisq']
        # dn_reg = res_dict['dn_reg']
        # temps = res_dict['temps']
        # fov = res_dict['fov']
    fig = plt.figure(figsize=(8, 9))
    for j in range(10):
        fig = plt.subplot(3, 4, j + 1)
        plt.imshow(np.log10(dem[:, :, j * 4] + 1e-20), 'inferno', vmin=19, vmax=23, origin='lower')
        ax = plt.gca()
        ax.set_title('%.1f' % (5.6 + j * 2 * 0.1))
        fig = plt.subplot(3, 4, 12)
        plt.imshow(aprep[0].data, origin='lower')

    return res_dict


def initGET_MW(libname):
    """
    Python wrapper for Codes for computing the solar gyroresonance and free-free radio emissions; both the isothermal
    plasma and the sources described by the differential emission measure (DEM) and differential density metric (DDM) are supported. By Alexey Kuznetsov, February 2021.
    Original code: https://github.com/kuznetsov-radio/GRFF

    This is for the single thread version
    @param libname: path for locating compiled shared library
    @return: An executable for calling the GS codes in single thread
    """
    _intp = ndpointer(dtype=ctypes.c_int32, flags='F')
    _doublep = ndpointer(dtype=ctypes.c_double, flags='F')

    mwfunc = ctypes.cdll.LoadLibrary(libname).GET_MW_SLICE
    mwfunc.argtypes = [ctypes.c_int,
                       ctypes.c_voidp * 7]
    # mwfunc.argtypes = [_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
    mwfunc.restype = ctypes.c_int

    return mwfunc


def ff_gyre_func(dem_res, temps, freq):
    # Get funcs depends on the os
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        libname = '/Users/fisher/Library/Mobile Documents/com~apple~CloudDocs/work/research_project/LWS_FST_2022/grff_python/GRFF_DEM_Transfer.so'
    if platform.system() == 'Windows':
        libname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               '../binaries/MWTransferArr64.dll')
    GET_MW = initGET_MW(libname)  # load the library

    # DEFINE Lparms: array of dimensions etc.
    (npx, npy) = (dem_res.shape[1], dem_res.shape[2])
    Np = npx * npy  # number of pixels
    Nz = 1  # number of Voxel
    Nf = len(freq)
    Nt = len(temps)
    use_dem = 0  # whether to use DEM -- 0 (on) or 1 (off)
    use_ddm = 1  # whether to use DEM -- 0 (on) or 1 (off)
    Lparms = np.zeros(6, dtype=ctypes.c_int)  # array of dimensions etc.
    Lparms[0] = Np
    Lparms[1] = Nz
    Lparms[2] = Nf
    Lparms[3] = Nt
    Lparms[4] = 0  # global DEM on (0)/off(1) key
    Lparms[5] = 1  # global DDM on (0)/off(1) key
    Lparms = Lparms.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # DEFINE global floating-point parameters
    pixel_size = 0.600698  # pixel size in arcsec
    arcsec2cm = 7.3e7
    p_area = (pixel_size * arcsec2cm) ** 2.0  # pixel in cm^-2
    Rparms = np.zeros((Np, 3), dtype=ctypes.c_double)
    Rparms[:, 0] = p_area
    Rparms[:, 1] = -1  # base frequency, read from 'freq' if < 0 #GHz
    Rparms[:, 2] = -1  # frequency step, read from 'freq' if < 0 #GHz
    Rparms = Rparms.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # DEFINE parms for single voxel
    # single-voxel parameters
    ParmLocal = np.zeros((15), dtype=ctypes.c_double)
    depth = 1.e10  # source depth, cm (total depth - the depths for individual voxels will be computed later)
    ParmLocal[0] = depth
    ParmLocal[1] = 1e7  # plasma temperature, K (not used in this example)
    ParmLocal[2] = 1e9  # electron/atomic concentration, cm^{-3} (not used in this example)
    ParmLocal[3] = 1e0  # magnetic field, G (will be changed later)
    ParmLocal[4] = 120e0  # viewing angle, degrees
    ParmLocal[5] = 0e0  # azimuthal angle, degrees
    ParmLocal[6] = 5  # emission mechanism flag (all on)
    ParmLocal[7] = 30  # maximum harmonic number
    ParmLocal[8] = 0  # proton concentration, cm^{-3} (not used in this example)
    ParmLocal[9] = 0  # neutral hydrogen concentration, cm^{-3}
    ParmLocal[10] = 0  # neutral helium concentration, cm^{-3}
    ParmLocal[11] = 0  # local DEM on/off key (on)
    ParmLocal[12] = 1  # local DDM on/off key (off)
    ParmLocal[13] = 0  # element abundance code (coronal, following Feldman 1992)
    ParmLocal[14] = 0  # reserved
    # 3D array of input parameters - for multiple pixels and voxels
    Parms = np.zeros((Np, Nz, 15), dtype=ctypes.c_double,
                     order='F')  # 3D array of input parameters - for multiple pixels and voxels
    for i in range(Nz):
        Parms[:, i, :] = ParmLocal  # most of the parameters are the same in all voxels
        Parms[:, i, 0] = ParmLocal[0] / Nz  # depths of individual voxels
        Parms[:, i, 3] = 0.0  # JUST FREE FREE
        # Parms[3, i] = 1000e0 - 700e0 * i / (Nz - 1)  # magnetic field decreases linearly from 1000 to 300 G
    Parms = Parms.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Adjust inputs
    DEM_arr = dem_res.transpose(1, 2, 0, 3) / depth  # new axis for voxel, =1
    DEM_arr.reshape(Np, Nz, Nt)
    T_arr = 10.0 ** np.asarray(temps)
    T_arr = T_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # dT_arr = np.zeros_like(temps)
    # dT_arr[:-1] = 10 **np.asarray(temps[:-1]) * np.log(10) *(np.asarray(temps[1:]) - np.asarray(temps[:-1]))
    # dT_arr[-1] = 10 ** np.asarray(temps[-1]) * np.log(10) * (np.asarray(temps[-1]) - np.asarray(temps[-2]))
    # convert from column EM (cm^-5) to volume-normalized DEM (cm^-6 K^-1)
    # DEM_arr = DEM_arr / depth
    DDM_arr = np.zeros_like(DEM_arr)
    DEM_arr = DEM_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    DDM_arr = DDM_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    RL = np.zeros((Np, Nf, 7), dtype=ctypes.c_double, order='F')  # input/output array
    RL[:, :, 0] = freq / 1.e9
    RL = RL.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    res = GET_MW(7, (ctypes.c_voidp * 7)(ctypes.cast(Lparms, ctypes.c_voidp),
                                         ctypes.cast(Rparms, ctypes.c_voidp),
                                         ctypes.cast(Parms, ctypes.c_voidp),
                                         ctypes.cast(T_arr, ctypes.c_voidp),
                                         ctypes.cast(DEM_arr, ctypes.c_voidp),
                                         ctypes.cast(DDM_arr, ctypes.c_voidp),
                                         ctypes.cast(RL, ctypes.c_voidp)))
    if res:
        print('U got it')
    output = np.ctypeslib.as_array(RL, shape=(Np, Nf, 7))

    freq = output[:, :, 0]
    left_flux = output[:, :, 5]
    right_flux = output[:, :, 6]

    sfu2cgs = 1e-19
    vc = 2.998e10  # [cm]
    kb = 1.38065e-16
    rad2asec = 180 / np.pi * 3600
    sr = (pixel_size / rad2asec) * (pixel_size / rad2asec)
    Tb = (left_flux + right_flux) * sfu2cgs * vc ** 2 / (2. * kb * (freq * 1e9) ** 2 * sr)
    # return Tb.reshape(dres[0].shape[0], dres[0].shape[1], n_frequency).transpose(2, 0, 1)
    return Tb.reshape(npx, npy, Nf).transpose(2, 0, 1)


def calculate_n_plot(demfile, tbmapfile='', freq=[], showplt=True, overwrite=False,pxy =None):
    # Get the temperature response functions in the correct form for demreg
    if len(freq) == 0:
        output_frequencies = np.logspace(np.log10(1.0e9), np.log10(20e9), 20)
    else:
        output_frequencies = np.array(freq)
    # dem_save_file = '/Volumes/Data/20220511/dem/demreg/test_res.p'
    # dem_save_file = '/Volumes/sandiskSSD/work/research_dataLWS_FST_2022/dem/dem1850_test.p'
    docalc_tbmap = True

    if os.path.exists(tbmapfile):
        if overwrite == False:
            docalc_tbmap = False

    if docalc_tbmap:
        with open(demfile, 'rb') as pf:
            dres = pickle.load(pf, encoding='latin1')
        dem = dres['dem']
        # edem = dres['edem']
        # elogt = dres['elogt']
        # chisq = dres['chisq']
        # dn_reg = dres['dn_reg']
        temps = dres['temps']
        # fov = dres['fov']
        pf.close()
        emission_maps = copy.deepcopy(dem)[np.newaxis, :, :, :]
        temperature_bins = ([np.mean([(np.log10(temps[i])), np.log10((temps[i + 1]))]) \
                             for i in np.arange(0, len(temps) - 1)])
        tbmaps = ff_gyre_func(emission_maps, temperature_bins, output_frequencies)
        res_dict = {'freqghz': output_frequencies / 1e9, 'tbmap': tbmaps}
        with open(tbmapfile, 'wb') as pf:
            pickle.dump(res_dict, pf)
    else:
        with open(tbmapfile, 'rb') as pf:
            res_dict = pickle.load(pf, encoding='latin1')
            tbmaps = res_dict['tbmap']
            output_frequencies = res_dict['freqghz'] * 1e9
    if showplt:
        # # PLOT
        # fig1, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
        # axs = list(chain.from_iterable(axs))
        # for it, ct in enumerate([7, 14, 21, 28, 35, 40]):
        #     cim = axs[it].imshow(emission_maps[0, :, :, ct], cmap="gist_heat", origin="lower")
        #     axs[it].set_title(f"Emission Map for logT={temperature_bins[ct]:.1f}")
        #     # fig1.colorbar(cim, cax=axs[it], orientation='vertical')
        #     cbar = plt.colorbar(cim, ax=[axs[it]])
        #     cbar.ax.set_ylabel('cm^-5 k-1', rotation=270)
        # # plt.show()

        fig2, axs2 = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True) #, figsize=(12, 7))
        axs2 = list(chain.from_iterable(axs2))
        for fi, cf in enumerate(np.linspace(3, 49, 9, dtype=int)):
            vmin = np.quantile(tbmaps[cf, :, :] / 1.e6, 0.9)
            vmax = np.quantile(tbmaps[cf, :, :] / 1.e6, 0.1)
            cim = axs2[fi].imshow(tbmaps[cf, :, :] / 1.e6, cmap="gist_heat", vmin=vmin, vmax=vmax, origin="lower")
            axs2[fi].set_title(f"Coronal model (f={output_frequencies[cf] / 1e9:.1f} GHz)")
            # fig2.colorbar(cim,ax=axs[fi])
            cbar = plt.colorbar(cim, ax=[axs2[fi]])
            cbar.ax.set_ylabel('MK', rotation=270)
        # plot the spectra at selected points

        # ref_map = smap.Map(ref_fits)
        # curmap = make_sub_map(cur_map=ref_map, fov=dres[-1])
        # pxy = get_pixel(curmap, x=940, y=-280)
        # # pxy = get_pixel(cmap,x=933,y=-283)
        # print(pxy)

        if pxy is not None:
            fig3, axs3 = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
            axs3[0].set_xlabel('Freq [GHz]')
            axs3[0].set_ylabel('Tb [K]')
            axs3[1].imshow(tbmaps[30, :, :], origin='lower')
            if isinstance(pxy[0],np.ndarray) or isinstance(pxy[0],list):
                for pxy_ in pxy:
                    axs3[0].loglog(output_frequencies/1e9, tbmaps[:, pxy_[1], pxy_[0]])
                    axs3[1].plot(pxy_[0], pxy_[1], marker='X')
            else:
                axs3[0].loglog(output_frequencies/1e9, tbmaps[:, pxy[1], pxy[0]])
                axs3[1].plot(pxy[0], pxy[1], marker='X', color='r')
            plt.show()
    return res_dict


def main():
    daystr = '20211122'
    daystr = '20220614'
    daystr = '20211124'
    # daystr = '20211125' ### SVG not converge. runtime error in DEM calculation.
    workdir = '/Users/fisher/Desktop/work/research/LWS_FST_2022/'  # main working directory.
    os.chdir(workdir)
    # doaiasmearing = False
    doaiasmearing = True

    # for daystr in ['20211122', '20211123', '20211124']:
    for daystr in ['20211124']:
        # for daystr in ['20220614']:
        # for daystr in ['20210205']:
        # for daystr in ['20211124']:
        dateobj = datetime.strptime(daystr, "%Y%m%d") + timedelta(hours=20)
        # timeobj = Time(dateobj)
        # tbg = dateobj - timedelta(hours=3.5)
        # ted = dateobj + timedelta(hours=3.5)

        demdir = os.path.join(workdir, 'dem', daystr)
        aiafitsdir = os.path.join(workdir, 'aiafiles', daystr)
        aiafitssmdir = os.path.join(workdir, 'aiafiles', daystr, '20UTsm')
        qlookfitsdir = os.path.join(workdir, 'EOVSA/disksynoptic', daystr[:-2], 'final')
        eofitsfiles = glob.glob(os.path.join(qlookfitsdir, 'eovsa_{}.spw??.tb.fits'.format(daystr)))
        eofitsfiles = sorted(eofitsfiles)

        dirs = [demdir, aiafitsdir, aiafitssmdir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
        # Can either get the example data via fido/vso or adapt the example to your own AIA data set

        # tbg = dateobj - timedelta(seconds=6)
        # ted = dateobj + timedelta(seconds=6)
        # # Only want the 6 coronal channels - this might also download 304A
        # result = Fido.search(a.Time(tbg, ted),
        #                      a.Instrument.aia,
        #                      # a.Wavelength(94 * u.angstrom, 335 * u.angstrom),
        #                      a.Sample(60*u.minute))
        # file_list = Fido.fetch(result, path=aiafitsdir)

        if doaiasmearing:
            wv_list = ['Z.131.', 'Z.171.', 'Z.211.', 'Z.94.', 'Z.335.', 'Z.193.']
            for wv in tqdm(wv_list, desc='aia image smearing'):

                aiaoutfits = os.path.join(aiafitsdir, '20UTsm/aia.lev1_euv_12s.{}{}image.fits'.format(
                    dateobj.strftime('%Y-%m-%dT%H%M%S'), wv))
                # if os.path.exists(aiaoutfits):
                #     os.system('rm -rf {}'.format(aiaoutfits))
                if not os.path.exists(aiaoutfits):
                    aiafitswv = sorted(glob.glob(os.path.join(aiafitsdir, 'aia.lev1_euv_12s.{}*{}image.fits'.format(
                        dateobj.strftime('%Y-%m'), wv))))
                    amap_seq = sunpy.map.Map(aiafitswv, sequence=True)
                    amap = amap_seq[int(len(amap_seq) / 2)]
                    hpc_coords = all_coordinates_from_map(amap)
                    maskondisk = coordinate_is_on_solar_disk(hpc_coords)
                    meta = amap.meta
                    amapdata = amap_seq.as_array()
                    exptime = []
                    mamapdata = ma.masked_less_equal(amapdata, 0)
                    mask = np.nansum(mamapdata.mask, axis=-1)
                    mask[mask > 0] = 1
                    mask[maskondisk > 0] = 0
                    for aidx, amp in enumerate(amap_seq):
                        amapdata[:, :, aidx] = amapdata[:, :, aidx] * amp.meta['exptime']
                        exptime.append(amp.meta['exptime'])
                    amapsmdata = np.nansum(amapdata, axis=-1) / np.nansum(exptime)
                    amapsmdata[np.where(mask)] = 0
                    meta['t_obs'] = dateobj.strftime('%Y-%m-%dT%H:%M:%S')
                    meta['exptime'] = np.nanmean(exptime)
                    if os.path.exists(aiaoutfits):
                        os.system('rm -rf {}'.format(aiaoutfits))
                    sio.write_file(aiaoutfits, amapsmdata, meta)
            aiafitsfiles = glob.glob(
                os.path.join(aiafitsdir,
                             '20UTsm/aia.lev1_euv_12s.{}*.image.fits'.format(dateobj.strftime('%Y-%m-%dT2000'))))
        else:
            aiafitsfiles = glob.glob(
                os.path.join(aiafitsdir,
                             '20UT/aia.lev1_euv_12s.{}*.image.fits'.format(dateobj.strftime('%Y-%m-%dT2000'))))

        demfile = os.path.join(demdir, 'demreg.p')
        if not os.path.exists(demfile):
            dres = cal_dem_map(aiafitsfiles, outfile=demfile, scalefactor=0.24, overwrite=False)
        tbmapfile = os.path.join(demdir, 'tbmaps.p')
        freq = np.array([1.2557164, 1.5797771, 2.5499082, 2.8736758, 3.1973457,
                         3.5207715, 3.844002, 4.1670375, 4.4920254, 4.8170137,
                         5.142002, 5.4669905, 5.7919784, 6.1169667, 6.4419556,
                         6.766944, 7.091932, 7.41692, 7.7419086, 8.066896,
                         8.391885, 8.716874, 9.041862, 9.36685, 9.691838,
                         10.016827, 10.341814, 10.666803, 10.991791, 11.316779,
                         11.6417675, 11.966756, 12.291745, 12.616733, 12.941721,
                         13.266709, 13.591698, 13.916685, 14.241674, 14.566662,
                         14.89165, 15.216639, 15.541627, 15.866616, 16.191605,
                         16.516592, 16.841581, 17.166569, 17.491556, 17.816545]) * 1e9
        res_tbmap = calculate_n_plot(demfile, tbmapfile=tbmapfile, freq=freq, showplt=True, overwrite=False)
        tbmap = res_tbmap['tbmap']

        tbmapcovfile = os.path.join(demdir, 'tbmaps_convolved.p')
        # print(not os.path.exists(tbmapcovfile))
        # print(os.path.join(qlookfitsdir, 'eovsa_{}.spw??.tb.fits'.format(daystr)))

        if not os.path.exists(tbmapcovfile):
            tbmap_convolved = convolve_eovsabeam(res_tbmap, eofitsfiles, tbmapcovfile=tbmapcovfile)

        # kernel = Gaussian2DKernel(x_stddev=10, y_stddev=5, theta=np.deg2rad(60))
        # tbmap_conv = convolve(tbmap[3], kernel)
        #
        # fig, axs = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
        # ax = axs[0]
        # ax.imshow(tbmap[3], origin='lower')
        # ax = axs[1]
        # ax.imshow(tbmap_conv, origin='lower')

        # tbmap = res_tbmap['tbmap']
        # output_frequencies = res_tbmap['freqghz'] * 1e9


if __name__ == '__main__':
    main()
