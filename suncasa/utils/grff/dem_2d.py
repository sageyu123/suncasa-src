import numpy as np

import matplotlib
import scipy.io as io


import astropy.time as atime
from astropy.coordinates import SkyCoord
from astropy import units as u
import sunpy.map

from aiapy.calibrate import degradation
from aiapy.calibrate import register, update_pointing
import pickle

import glob

import warnings
import time
import matplotlib.pyplot as plt
from demregpy import dn2dem
from astropy.io import fits

warnings.simplefilter('ignore')
matplotlib.rcParams['font.size'] = 16

start_tim = time.time()

def make_sub_map(cur_map, fov=None, ref_map=None):
    if not ref_map:
        ref_map = cur_map
    bole = SkyCoord(fov[0][0] * u.arcsec, fov[0][1] * u.arcsec, frame=ref_map.coordinate_frame)
    tori = SkyCoord(fov[1][0] * u.arcsec, fov[1][1] * u.arcsec, frame=ref_map.coordinate_frame)
    sub_map = cur_map.submap(bole, tori)
    return sub_map
# dem in 2d
def cal_dem_map():
    # Can either get the example data via fido/vso or adapt the example to your own AIA data set
    from sunpy.net import Fido, attrs as a

    # Only want the 6 coronal channels - this might also download 304A
    # wvsrch=a.Wavelength(94*u.angstrom, 335*u.angstrom)
    #
    # result = Fido.search(a.Time('2011-01-27T00:00:00', '2011-01-27T00:00:09'), a.Instrument("aia"), wvsrch)
    # file_list = Fido.fetch(result,path='/Volumes/Data/20220511/dem_test/20110127')
    file_list = glob.glob('/Volumes/Data/20220511/dem_test/20110127/*aia*.fits')

    aia_tresp_en_file = '/Users/walterwei/software/external_package/demreg/python/aia_tresp_en.dat'
    savefile = '/Volumes/Data/20220511/dem_test/20110127/dem_test_demreg_demo1.p'
    fov = [[-1100,-400],[-900,-200]]
    #time_string = '2011-01-27T00:00:00.000'
    hdulist = fits.open(file_list, mode='readonly')
    time_string = hdulist[1].header['T_OBS'][:-1]
    hdulist.close()
    kw_list = ['Z.131.', 'Z.171.', 'Z.211.', 'Z.94.', 'Z.335.', 'Z.193.']
    iter_st = time.time()
    print('cur_tiem is ', time_string)
    amaps = sunpy.map.Map(file_list)
    # Get the wavelengths of the maps and get index of sort for this list of maps
    wvn0 = [m.meta['wavelnth'] for m in amaps]
    # print(wvn0)
    srt_id = sorted(range(len(wvn0)), key=wvn0.__getitem__)
    print('sorted_list is : ', srt_id)
    amaps = [amaps[i] for i in srt_id]
    # print([m.meta['wavelnth'] for m in amaps])
    channels = [94, 131, 171, 193, 211, 335] * u.angstrom
    #cctime = atime.Time(time_string, scale='utc')
    cctime = atime.Time(time_string, format='isot')
    nc = len(channels)
    degs = np.empty(nc)
    for i in np.arange(nc):
        degs[i] = degradation(channels[i], cctime, calibration_version=10)
    aprep = []
    for m in amaps:
        m_temp = update_pointing(m)
        aprep.append(register(m_temp))
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
    temps = np.logspace(5.7, 7.6, num=42)
    # Temperature bin mid-points for DEM plotting
    mlogt = ([np.mean([(np.log10(temps[i])), np.log10((temps[i + 1]))]) \
              for i in np.arange(0, len(temps) - 1)])

    # --------------------------------------------------------------------------------------------

    tmp_aprep = []
    for api, cap in enumerate(aprep):
        tmp_aprep.append(make_sub_map(cur_map=cap, fov=fov))
    aprep = tmp_aprep
    #aprep[0].peek()

    cmap_shape = aprep[0].data.shape
    data_cube = np.zeros((cmap_shape[0], cmap_shape[1], 6))
    num_pix = 4096.**2/(cmap_shape[0]*cmap_shape[1])
    #num_pix = 1.0
    rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18])*np.sqrt(num_pix)/num_pix
    for mi, m in enumerate(aprep):
        data_cube[:, :, mi] = m.data / degs[mi] / durs[mi]
        dn2ph_cube = np.broadcast_to(dn2ph, (cmap_shape[0], cmap_shape[1], 6))
        degs_cube = np.broadcast_to(degs, (cmap_shape[0], cmap_shape[1], 6))
        rdnse_cube = np.broadcast_to(rdnse, (cmap_shape[0], cmap_shape[1], 6))

    edata_cube = (((dn2ph_cube * data_cube * num_pix) ** 0.5 / dn2ph_cube / num_pix / degs_cube) ** 2 + rdnse_cube ** 2) ** 0.5 / degs_cube
    pe_time = time.time()
    print('it take to {} to prep in iter_time'.format(pe_time - iter_st))
    print('{} * {}, {} pixels to calculate in total'.format(data_cube.shape[0], data_cube.shape[1],
                                                            data_cube.shape[0] * data_cube.shape[1]))

    dem, edem, elogt, chisq, dn_reg = dn2dem(data_cube, edata_cube, trmatrix, tresp_logt, temps)
    fig = plt.figure(figsize=(8, 9))
    for j in range(10):
        fig = plt.subplot(3, 4, j + 1)
        plt.imshow(np.log10(dem[:, :, j * 4] + 1e-20), 'inferno', vmin=19, vmax=23, origin='lower')
        ax = plt.gca()
        ax.set_title('%.1f' % (5.6 + j * 2 * 0.1))
        fig = plt.subplot(3, 4, 12)
        plt.imshow(data_cube[:,:,0], origin='lower')
    res_tuple = (dem, edem, elogt, chisq, dn_reg, temps,fov)
    #pickle.dump(res_tuple, open('/Volumes/Data/20220511/dem/demreg/test_res.p', 'wb'))
    pickle.dump(res_tuple, open(savefile, 'wb'))
    return (dem, edem, elogt, chisq, dn_reg, temps,fov)


def main():
    cal_dem_map()

if __name__ == '__main__':
    main()
