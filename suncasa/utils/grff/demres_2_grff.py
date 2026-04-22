import pickle
import copy
import matplotlib.pyplot as plt
import ctypes
import numpy as np
import platform
import os
from numpy.ctypeslib import ndpointer
from itertools import chain
import sunpy.map as smap

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
    #mwfunc.argtypes = [_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
    mwfunc.restype = ctypes.c_int

    return mwfunc


def ff_gyre_func(dem_res, temps,freq):
    # Get funcs depends on the os
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        #libname = '/Users/walter/Desktop/grff_test/GRFF_DEM_Transfer.so'
        #libname = '/Volumes/Data/20220511/dem/GRFF_DEM_Transfer.so'
        libname = '/Users/walterwei/Downloads/GRFF_DEM_Transfer.so'
        # libname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                        '../binaries/MWTransferArr.so')
    if platform.system() == 'Windows':
        libname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               '../binaries/MWTransferArr64.dll')
    GET_MW = initGET_MW(libname)  # load the library

    #DEFINE Lparms: array of dimensions etc.
    (npx, npy) = (dem_res.shape[1], dem_res.shape[2])
    Np = npx*npy # number of pixels
    Nz = 1 # number of Voxel
    Nf = len(freq)
    Nt = len(temps)
    use_dem = 0  # whether to use DEM -- 0 (on) or 1 (off)
    use_ddm = 1  # whether to use DEM -- 0 (on) or 1 (off)
    Lparms = np.zeros(6, dtype=ctypes.c_int)  # array of dimensions etc.
    Lparms[0] = Np
    Lparms[1] = Nz
    Lparms[2]=Nf
    Lparms[3]=Nt
    Lparms[4]=0 #global DEM on (0)/off(1) key
    Lparms[5]=1 #global DDM on (0)/off(1) key
    Lparms = Lparms.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    #DEFINE global floating-point parameters
    pixel_size = 0.600698 #pixel size in arcsec
    arcsec2cm = 7.3e7
    p_area = (pixel_size * arcsec2cm) ** 2.0 # pixel in cm^-2
    Rparms = np.zeros((Np,3), dtype=ctypes.c_double)
    Rparms[:,0] = p_area
    Rparms[:,1] = -1 # base frequency, read from 'freq' if < 0 #GHz
    Rparms[:,2] = -1 # frequency step, read from 'freq' if < 0 #GHz
    Rparms = Rparms.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    #DEFINE parms for single voxel
    #single-voxel parameters
    ParmLocal = np.zeros((15),dtype=ctypes.c_double)
    depth = 1.e10 # source depth, cm (total depth - the depths for individual voxels will be computed later)
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
    Parms = np.zeros((Np,Nz,15), dtype=ctypes.c_double, order='F')  # 3D array of input parameters - for multiple pixels and voxels
    for i in range(Nz):
        Parms[:, i, :] = ParmLocal  # most of the parameters are the same in all voxels
        Parms[:, i, 0] = ParmLocal[0]/Nz #depths of individual voxels
        Parms[:, i, 3] = 0.0 #JUST FREE FREE
        #Parms[3, i] = 1000e0 - 700e0 * i / (Nz - 1)  # magnetic field decreases linearly from 1000 to 300 G
    Parms = Parms.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    #Adjust inputs
    DEM_arr = dem_res.transpose(1,2,0,3) / depth  # new axis for voxel, =1
    DEM_arr.reshape(Np, Nz, Nt)
    T_arr = 10.0 ** np.asarray(temps)
    T_arr = T_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # dT_arr = np.zeros_like(temps)
    # dT_arr[:-1] = 10 **np.asarray(temps[:-1]) * np.log(10) *(np.asarray(temps[1:]) - np.asarray(temps[:-1]))
    # dT_arr[-1] = 10 ** np.asarray(temps[-1]) * np.log(10) * (np.asarray(temps[-1]) - np.asarray(temps[-2]))
    # convert from column EM (cm^-5) to volume-normalized DEM (cm^-6 K^-1)
    #DEM_arr = DEM_arr / depth
    DDM_arr = np.zeros_like(DEM_arr)
    DEM_arr = DEM_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    DDM_arr = DDM_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


    RL = np.zeros((Np, Nf, 7), dtype=ctypes.c_double, order='F')  # input/output array
    RL[:, :, 0] = freq/1.e9
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
    vc = 2.998e10 #[cm]
    kb = 1.38065e-16
    rad2asec = 180 / np.pi * 3600
    sr = (pixel_size / rad2asec) * (pixel_size / rad2asec)
    Tb = (left_flux + right_flux) * sfu2cgs * vc ** 2 / (2. * kb * (freq * 1e9) ** 2 * sr)
    #return Tb.reshape(dres[0].shape[0], dres[0].shape[1], n_frequency).transpose(2, 0, 1)
    return Tb.reshape(npx, npy, Nf).transpose(2, 0, 1)

def calculate_n_plot():
    # Get the temperature response functions in the correct form for demreg
    output_frequencies = np.logspace(np.log10(1.0e9), np.log10(20e9), 20)
    #dem_save_file = '/Volumes/Data/20220511/dem/demreg/test_res.p'
    dem_save_file = '/Volumes/Data/20220511/dem_test/dem1850_test.p'
    with open(dem_save_file, 'rb') as pf:
        dres = pickle.load(pf, encoding='latin1')
    pf.close()
    emission_maps = copy.deepcopy(dres[0])[np.newaxis,:,:,:]
    temps = dres[-2]
    temperature_bins = ([np.mean([(np.log10(temps[i])), np.log10((temps[i + 1]))]) \
              for i in np.arange(0, len(temps) - 1)])
    tb_maps = ff_gyre_func(emission_maps, temperature_bins, output_frequencies)
    #PLOT
    fig1, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
    axs = list(chain.from_iterable(axs))
    for it,ct in enumerate([7,14,21,28,35,40]):
        cim = axs[it].imshow(emission_maps[0, :, :, ct], cmap="gist_heat", origin="lower")
        axs[it].set_title(f"Emission Map for logT={temperature_bins[ct]:.1f}")
        #fig1.colorbar(cim, cax=axs[it], orientation='vertical')
        cbar = plt.colorbar(cim,ax=[axs[it]])
        cbar.ax.set_ylabel('cm^-5 k-1', rotation=270)
    #plt.show()

    fig2, axs2 = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
    axs2 = list(chain.from_iterable(axs2))
    for fi, cf in enumerate(np.linspace(0,19,9,dtype=int)):
        vmin = np.quantile(tb_maps[cf, :, :]/1.e6, 0.99)
        vmax = np.quantile(tb_maps[cf, :, :]/1.e6, 0.01)
        cim = axs2[fi].imshow(tb_maps[cf, :, :]/1.e6, cmap="gist_heat", vmin=vmin, vmax=vmax, origin="lower")
        axs2[fi].set_title(f"Coronal model (f={output_frequencies[cf] / 1e9:.1f} GHz)")
        #fig2.colorbar(cim,ax=axs[fi])
        cbar = plt.colorbar(cim, ax=[axs2[fi]])
        cbar.ax.set_ylabel('MK', rotation=270)
    #plot the spectra at selected points
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    def make_sub_map(cur_map, fov=None, ref_map=None):
        if not ref_map:
            ref_map = cur_map
        bole = SkyCoord(fov[0][0] * u.arcsec, fov[0][1] * u.arcsec, frame=ref_map.coordinate_frame)
        tori = SkyCoord(fov[1][0] * u.arcsec, fov[1][1] * u.arcsec, frame=ref_map.coordinate_frame)
        sub_map = cur_map.submap(bole, tori)
        return sub_map
    def get_pixel(cur_map, x, y):
        coor = SkyCoord(x * u.arcsec, y * u.arcsec, frame=cur_map.coordinate_frame)
        pix_coor = cur_map.world_to_pixel(coor)
        x = int(pix_coor.x.value)
        y = int(pix_coor.y.value)
        return [x, y]
    ref_map =  smap.Map('/Volumes/Data/20220511/dem_test/aia.lev1_euv_12s.2022-05-11T184959Z.94.image_lev1.fits')
    cmap = make_sub_map(cur_map=ref_map, fov=dres[-1])
    pxy = get_pixel(cmap,x=940,y=-280)
    #pxy = get_pixel(cmap,x=933,y=-283)
    print(pxy)
    fig3, axs3 = plt.subplots(nrows=1,ncols=2, figsize=(6, 4))
    axs3[0].loglog(output_frequencies, tb_maps[:, pxy[1], pxy[0]])
    axs3[1].imshow(tb_maps[0, :, :],origin='lower')
    axs3[1].plot(pxy[0],pxy[1],marker='X',color='r')
    axs3[0].set_xlabel('Freq [GHz]')
    axs3[0].set_ylabel('Tb [K]')
    plt.show()

def main():
    calculate_n_plot()
if __name__ == '__main__':
    main()