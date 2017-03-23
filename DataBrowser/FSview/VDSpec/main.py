import json
import os, sys
import pickle
import time
from collections import OrderedDict
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from sys import platform
import scipy.ndimage as sn
from math import radians, cos, sin
from bokeh.layouts import row, column, widgetbox, gridplot
from bokeh.models import (ColumnDataSource, CustomJS, Slider, Button, TextInput, RadioButtonGroup, CheckboxGroup,
                          BoxSelectTool, LassoSelectTool, HoverTool, Spacer, LabelSet, Div)
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.widgets import Select, RangeSlider
from bokeh.palettes import Spectral11
from bokeh.plotting import figure, curdoc
import glob
from astropy.time import Time
from suncasa.utils.puffin import PuffinMap
from suncasa.utils import DButil
from suncasa.utils import ctplot

__author__ = ["Sijie Yu"]
__email__ = "sijie.yu@njit.edu"

if platform == "linux" or platform == "linux2":
    print 'Runing QLook in Linux platform'
    for ll in xrange(5100, 5100 + 10):
        os.system('fuser -n tcp -k {}'.format(ll))
elif platform == "darwin":
    print 'Runing QLook in OS X platform'
    for ll in xrange(5100, 5100 + 10):
        os.system(
            'port=($(lsof -i tcp:{}|grep python2.7 |cut -f2 -d" ")); [[ -n "$port" ]] && kill -9 $port'.format(ll))
        os.system('port=($(lsof -i tcp:{}|grep Google |cut -f2 -d" ")); [[ -n "$port" ]] && kill -9 $port'.format(ll))
elif platform == "win32":
    print 'Runing QLook in Windows platform'

'''load config file'''
suncasa_dir = os.path.expandvars("${SUNCASA}") + '/'
DButil.initconfig(suncasa_dir)
'''load config file'''
config_main = DButil.loadjsonfile(suncasa_dir + 'DataBrowser/config.json')
database_dir = config_main['datadir']['database']
database_dir = os.path.expandvars(database_dir) + '/'
config_EvtID = DButil.loadjsonfile('{}config_EvtID_curr.json'.format(database_dir))
SDOdir = DButil.getSDOdir(config_main, database_dir + '/aiaBrowserData/', suncasa_dir)
spec_square_rs_tmax = config_main['plot_config']['tab_FSview_base']['spec_square_rs_tmax']
spec_square_rs_fmax = config_main['plot_config']['tab_FSview_base']['spec_square_rs_fmax']
spec_image_rs_ratio = config_main['plot_config']['tab_FSview_base']['spec_image_rs_ratio']
tidx_prev = None

# do_spec_regrid = False

'''define the colormaps'''
colormap_jet = cm.get_cmap("jet")  # choose any matplotlib colormap here
bokehpalette_jet = [colors.rgb2hex(m) for m in colormap_jet(np.arange(colormap_jet.N))]
colormap = cm.get_cmap("cubehelix")  # choose any matplotlib colormap here
bokehpalette_SynthesisImg = [colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
colormap_viridis = cm.get_cmap("viridis")  # choose any matplotlib colormap here
bokehpalette_viridis = [colors.rgb2hex(m) for m in colormap_viridis(np.arange(colormap_viridis.N))]
'''
-------------------------- panel 2,3   --------------------------
'''


def read_fits(fname):
    hdulist = fits.open(fname)
    hdu = hdulist[0]
    return hdu


def goodchan(hdu):
    ndx = hdu.header["NAXIS1"]
    ndy = hdu.header["NAXIS2"]
    xc = ndx / 2
    yc = ndy / 2
    hdu_goodchan = \
        np.where(np.nanmean(hdu.data[0, :, yc - ndy / 16:yc + ndy / 16, xc - ndx / 16:xc + ndx / 16], axis=(-1, -2)))[0]
    return hdu_goodchan


# initial the source of maxfit centroid
def tab2_SRC_maxfit_centroid_init(dspecDFsel):
    start_timestamp = time.time()
    global SRC_maxfit_centroid
    SRC_maxfit_centroid = {}
    for ll in np.unique(dspecDFsel['time']):
        df_tmp = pd.DataFrame(
            {'freq': [], 'shape_longitude': [], 'shape_latitude': [], 'peak': []})
        SRC_maxfit_centroid[np.where(abs(tab2_dtim - ll) < 0.02)[0].tolist()[0]] = ColumnDataSource(df_tmp)
    print("---tab2_SRC_maxfit_centroid_init -- %s seconds ---" % (time.time() - start_timestamp))


def aia_submap_wavelength_selection(attrname, old, new):
    global tab3_r_aia_submap
    select_wave = tab2_Select_aia_wave.value
    aiamap = DButil.readsdofile(datadir=SDOdir, wavelength=select_wave, jdtime=xx[0] / 3600. / 24.,
                                timtol=tab2_dur / 3600. / 24.)
    print 'wavelength {} selected'.format(select_wave)
    lengthx = vla_local_pfmap.dw[0] * u.arcsec / 3.0
    lengthy = vla_local_pfmap.dh[0] * u.arcsec / 3.0
    x0 = vla_local_pfmap.smap.center.x
    y0 = vla_local_pfmap.smap.center.y
    aiamap_submap = aiamap.submap(u.Quantity([x0 - lengthx / 2, x0 + lengthx / 2]),
                                  u.Quantity([y0 - lengthy / 2, y0 + lengthy / 2]))
    aia_submap_pfmap = PuffinMap(smap=aiamap_submap,
                                 plot_height=config_main['plot_config']['tab_FSview_FitANLYS'][
                                     'aia_submap_hght'],
                                 plot_width=config_main['plot_config']['tab_FSview_FitANLYS'][
                                     'aia_submap_wdth'],
                                 webgl=config_main['plot_config']['WebGL'])
    tab3_r_aia_submap.data_source.data['image'] = aia_submap_pfmap.ImageSource()['data']


def tab2_Select_vla_pol_update(attrname, old, new):
    global hdu, select_vla_pol, dspecDF0POL, tidx_prev
    select_vla_pol = tab2_Select_vla_pol.value
    dspecDF0POL = DButil.dspecDFfilter(dspecDF0, select_vla_pol)
    thresholdrange = (np.floor(dspecDF0POL['peak'].min()), np.ceil(dspecDF0POL['peak'].max()))
    tab3_rSlider_threshold.start = thresholdrange[0]
    tab3_rSlider_threshold.end = thresholdrange[1]
    tab3_rSlider_threshold.range = thresholdrange
    if tab2_dspec_vector_selected:
        VdspecDF_update(selected=tab2_dspec_vector_selected)
    else:
        VdspecDF_update()
    for ll in range(len(tab3_dspec_small_CTRLs_OPT['labels_dspec_small'])):
        RBG_dspec_small_update(ll)


def tab2_SRC_maxfit_centroid_update(dspecDFsel):
    start_timestamp = time.time()
    global SRC_maxfit_centroid, timebin
    subset_label = ['freq', 'shape_longitude', 'shape_latitude', 'peak']
    if tab3_BUT_animate_ONOFF.label == 'Animate ON & Go':
        SRC_maxfit_centroid = {}
        for ll in np.unique(dspecDFsel['time']):
            dftmp = dspecDFsel[dspecDFsel.time == ll]
            dftmp = dftmp.dropna(how='any', subset=subset_label)
            df_tmp = pd.concat(
                [dftmp.loc[:, 'freq'], dftmp.loc[:, 'shape_longitude'], dftmp.loc[:, 'shape_latitude'],
                 dftmp.loc[:, 'peak']], axis=1)
            SRC_maxfit_centroid[np.where(abs(tab2_dtim - ll) < 0.02)[0].tolist()[0]] = ColumnDataSource(df_tmp)
    else:
        time_dspec = np.unique(dspecDFsel['time'])
        ntime_dspec = len(time_dspec)
        if timebin != 1:
            tidx = np.arange(0, ntime_dspec + 1, timebin)
            time_seq = time_dspec[0:0 + timebin]
            dftmp = dspecDFsel[dspecDFsel['time'].isin(time_seq)]
            dftmp = dftmp.dropna(how='any', subset=subset_label)
            dftmp_concat = pd.DataFrame(dict(dftmp.mean()), index=[0, ])
            for ll in tidx[1:]:
                time_seq = time_dspec[ll:ll + timebin]
                dftmp = dspecDFsel[dspecDFsel['time'].isin(time_seq)]
                dftmp = dftmp.dropna(how='any', subset=subset_label)
                dftmp_concat = dftmp_concat.append(pd.DataFrame(dict(dftmp.mean()), index=[0, ]),
                                                   ignore_index=True)
            SRC_maxfit_centroid = ColumnDataSource(
                dftmp_concat[subset_label].dropna(
                    how='any'))
        else:
            dftmp = dspecDFsel.copy()
            dftmp = dftmp.dropna(how='any', subset=subset_label)
            df_tmp = pd.concat(
                [dftmp.loc[:, 'freq'], dftmp.loc[:, 'shape_longitude'], dftmp.loc[:, 'shape_latitude'],
                 dftmp.loc[:, 'peak']], axis=1)
            SRC_maxfit_centroid = ColumnDataSource(df_tmp)
    print("--- tab2_SRC_maxfit_centroid_update -- %s seconds ---" % (time.time() - start_timestamp))


def tab3_aia_submap_cross_selection_change(attrname, old, new):
    global tab3_dspec_vectorx_img, tab3_dspec_vectory_img
    global vmax_vx, vmax_vy, vmin_vx, vmin_vy, mean_vx, mean_vy
    global VdspecDF
    tab3_aia_submap_cross_selected = tab3_r_aia_submap_cross.data_source.selected['1d']['indices']
    if tab3_aia_submap_cross_selected:
        tmpDF = tab3_r_aia_submap_cross.data_source.to_df().iloc[tab3_aia_submap_cross_selected, :]
        xa0, xa1 = tmpDF['shape_longitude'].min(), tmpDF['shape_longitude'].max()
        ya0, ya1 = tmpDF['shape_latitude'].min(), tmpDF['shape_latitude'].max()
        print xa0, xa1, ya0, ya1
        mean_vx = (xa0 + xa1) / 2
        mean_vy = (ya0 + ya1) / 2
        tab3_r_aia_submap_rect.data_source.data['x'] = [mean_vx]
        tab3_r_aia_submap_rect.data_source.data['y'] = [mean_vy]
        tab3_r_aia_submap_rect.data_source.data['width'] = [(xa1 - xa0)]
        tab3_r_aia_submap_rect.data_source.data['height'] = [(ya1 - ya0)]
        vx = (VdspecDF['shape_longitude'].copy()).values.reshape(tab2_nfreq, tab2_ntim)
        vmax_vx, vmin_vx = xa1, xa0
        vx[vx > vmax_vx] = vmax_vx
        vx[vx < vmin_vx] = vmin_vx
        tab3_r_dspec_vectorx.data_source.data['image'] = [vx]
        vy = (VdspecDF['shape_latitude'].copy()).values.reshape(tab2_nfreq, tab2_ntim)
        vmax_vy, vmin_vy = ya1, ya0
        vy[vy > vmax_vy] = vmax_vy
        vy[vy < vmin_vy] = vmin_vy
        tab3_r_dspec_vectory.data_source.data['image'] = [vy]
        tab3_dspec_small_CTRLs_OPT['vmax_values_last'][1] = xa1
        tab3_dspec_small_CTRLs_OPT['vmax_values_last'][2] = ya1
        tab3_dspec_small_CTRLs_OPT['vmin_values_last'][1] = xa0
        tab3_dspec_small_CTRLs_OPT['vmin_values_last'][2] = ya0
    else:
        tab3_r_aia_submap_rect.data_source.data['x'] = [(vmax_vx + vmin_vx) / 2]
        tab3_r_aia_submap_rect.data_source.data['y'] = [(vmax_vy + vmin_vy) / 2]
        tab3_r_aia_submap_rect.data_source.data['width'] = [(vmax_vx - vmin_vx)]
        tab3_r_aia_submap_rect.data_source.data['height'] = [(vmax_vy - vmin_vy)]


def VdspecDF_init():
    global VdspecDF, dspecDF0, dspecDF0POL
    VdspecDF = pd.DataFrame()
    nrows_dspecDF = len(dspecDF0POL.index)
    VdspecDF['peak'] = pd.Series([np.nan] * nrows_dspecDF, index=dspecDF0POL.index)
    VdspecDF['shape_longitude'] = pd.Series([np.nan] * nrows_dspecDF, index=dspecDF0POL.index)
    VdspecDF['shape_latitude'] = pd.Series([np.nan] * nrows_dspecDF, index=dspecDF0POL.index)


def VdspecDF_update(selected=None):
    global VdspecDF
    if selected:
        VdspecDF.loc[selected, 'shape_longitude'] = dspecDF0POL.loc[selected, 'shape_longitude']
        VdspecDF.loc[selected, 'shape_latitude'] = dspecDF0POL.loc[selected, 'shape_latitude']
        VdspecDF.loc[selected, 'peak'] = dspecDF0POL.loc[selected, 'peak']
    else:
        VdspecDF.loc[:, 'shape_longitude'] = dspecDF0POL.loc[:, 'shape_longitude']
        VdspecDF.loc[:, 'shape_latitude'] = dspecDF0POL.loc[:, 'shape_latitude']
        VdspecDF.loc[:, 'peak'] = dspecDF0POL.loc[:, 'peak']


def tab3_SRC_dspec_vector_init():
    global tab3_dspec_vector_img, tab3_dspec_vectorx_img, tab3_dspec_vectory_img
    global mean_amp_g, mean_vx, mean_vy, drange_amp_g, drange_vx, drange_vy
    global vmax_amp_g, vmax_vx, vmax_vy, vmin_amp_g, vmin_vx, vmin_vy
    start_timestamp = time.time()
    amp_g = (dspecDF0POL['peak'].copy()).values.reshape(tab2_nfreq, tab2_ntim)
    mean_amp_g = np.nanmean(amp_g)
    drange_amp_g = 40.
    vmax_amp_g, vmin_amp_g = mean_amp_g + drange_amp_g * np.asarray([1., -1.])
    amp_g[amp_g > vmax_amp_g] = vmax_amp_g
    amp_g[amp_g < vmin_amp_g] = vmin_amp_g
    tab3_dspec_vector_img = [amp_g]
    vx = (dspecDF0POL['shape_longitude'].copy()).values.reshape(tab2_nfreq, tab2_ntim)
    mean_vx = np.nanmean(vx)
    drange_vx = 40.
    vmax_vx, vmin_vx = mean_vx + drange_vx * np.asarray([1., -1.])
    vx[vx > vmax_vx] = vmax_vx
    vx[vx < vmin_vx] = vmin_vx
    tab3_dspec_vectorx_img = [vx]
    vy = (dspecDF0POL['shape_latitude'].copy()).values.reshape(tab2_nfreq, tab2_ntim)
    mean_vy = np.nanmean(vy)
    drange_vy = 40.
    vmax_vy, vmin_vy = mean_vy + drange_vy * np.asarray([1., -1.])
    vy[vy > vmax_vy] = vmax_vy
    vy[vy < vmin_vy] = vmin_vy
    tab3_dspec_vectory_img = [vy]
    tab3_r_aia_submap_rect.data_source.data['x'] = [(vmax_vx + vmin_vx) / 2]
    tab3_r_aia_submap_rect.data_source.data['y'] = [(vmax_vy + vmin_vy) / 2]
    tab3_r_aia_submap_rect.data_source.data['width'] = [(vmax_vx - vmin_vx)]
    tab3_r_aia_submap_rect.data_source.data['height'] = [(vmax_vy - vmin_vy)]
    print("--- tab3_SRC_dspec_small_init -- %s seconds ---" % (time.time() - start_timestamp))


def tab3_SRC_dspec_vector_update():
    global tab3_r_dspec_vector, tab3_r_dspec_vectorx, tab3_r_dspec_vectory
    global mean_amp_g, mean_vx, mean_vy, drange_amp_g, drange_vx, drange_vy
    global vmax_amp_g, vmax_vx, vmax_vy, vmin_amp_g, vmin_vx, vmin_vy
    global VdspecDF
    start_timestamp = time.time()
    amp_g = (VdspecDF['peak'].copy()).values.reshape(tab2_nfreq, tab2_ntim)
    mean_amp_g = np.nanmean(amp_g)
    drange_amp_g = 40.
    vmax_amp_g, vmin_amp_g = mean_amp_g + drange_amp_g * np.asarray([1., -1.])
    amp_g[amp_g > vmax_amp_g] = vmax_amp_g
    amp_g[amp_g < vmin_amp_g] = vmin_amp_g
    tab3_r_dspec_vector.data_source.data['image'] = [amp_g]
    # todo add threshold selection to the vector dynamic spectrum
    vx = (VdspecDF['shape_longitude'].copy()).values.reshape(tab2_nfreq, tab2_ntim)
    mean_vx = np.nanmean(vx)
    drange_vx = 40.
    vmax_vx, vmin_vx = mean_vx + drange_vx * np.asarray([1., -1.])
    vx[vx > vmax_vx] = vmax_vx
    vx[vx < vmin_vx] = vmin_vx
    tab3_r_dspec_vectorx.data_source.data['image'] = [vx]
    vy = (VdspecDF['shape_latitude'].copy()).values.reshape(tab2_nfreq, tab2_ntim)
    mean_vy = np.nanmean(vy)
    drange_vy = 40.
    vmax_vy, vmin_vy = mean_vy + drange_vy * np.asarray([1., -1.])
    vy[vy > vmax_vy] = vmax_vy
    vy[vy < vmin_vy] = vmin_vy
    tab3_r_dspec_vectory.data_source.data['image'] = [vy]
    print("--- tab3_SRC_dspec_small_update -- %s seconds ---" % (time.time() - start_timestamp))


def rSlider_threshold_handler(attrname, old, new):
    global thresholdrange
    print tab3_p_dspec_vector.x_range.start, tab3_p_dspec_vector.x_range.end
    thresholdrange = tab3_rSlider_threshold.range
    tab2_SRC_dspec_vector_square.selected = {'2d': {}, '1d': {'indices': list(
        dspecDF0POL[dspecDF0POL['peak'] <= thresholdrange[1]][dspecDF0POL['peak'] >= thresholdrange[0]][
            dspecDF0POL['time'] >= tab3_p_dspec_vector.x_range.start][
            dspecDF0POL['time'] <= tab3_p_dspec_vector.x_range.end][
            dspecDF0POL['freq'] >= tab3_p_dspec_vector.y_range.start][
            dspecDF0POL['freq'] <= tab3_p_dspec_vector.y_range.end].index)},
                                             '0d': {'indices': [], 'get_view': {}, 'glyph': None}}
    for ll in range(len(tab3_dspec_small_CTRLs_OPT['labels_dspec_small'])):
        RBG_dspec_small_update(ll)


def dspec_vector_selection_change(selected):
    global dspecDF_select
    dspecDF_select = dspecDF0POL.iloc[selected, :]
    VdspecDF_init()
    VdspecDF_update(selected=selected)
    # tab3_SRC_dspec_vector_update(VdspecDF)
    tab2_SRC_maxfit_centroid_update(dspecDF_select)
    if tab3_BUT_animate_ONOFF.label == 'Animate OFF & Go':
        tab3_r_aia_submap_cross.visible = True
        tab3_r_dspec_vector_line.visible = False
        tab3_r_dspec_vectorx_line.visible = False
        tab3_r_dspec_vectory_line.visible = False
        tab3_r_aia_submap_cross.data_source.data = SRC_maxfit_centroid.data


def tab2_dspec_vector_selection_change(attrname, old, new):
    global tab2_dspec_vector_selected
    tab2_dspec_vector_selected = tab2_SRC_dspec_vector_square.selected['1d']['indices']
    if tab2_dspec_vector_selected:
        dspec_vector_selection_change(tab2_dspec_vector_selected)


def RBG_dspec_small_update(idx):
    global tab3_dspec_small_CTRLs_OPT
    tab3_dspec_small_CTRLs_OPT['idx_p_dspec_small'] = idx
    tab3_dspec_small_CTRLs_OPT['radio_button_group_dspec_small_update_flag'] = True
    mean_values = tab3_dspec_small_CTRLs_OPT['mean_values']
    drange_values = tab3_dspec_small_CTRLs_OPT['drange_values']
    vmax_values_last = tab3_dspec_small_CTRLs_OPT['vmax_values_last']
    vmin_values_last = tab3_dspec_small_CTRLs_OPT['vmin_values_last']
    tab3_Slider_dspec_small_dmax.start = mean_values[idx] - drange_values[idx]
    tab3_Slider_dspec_small_dmax.end = mean_values[idx] + 2 * drange_values[idx]
    tab3_Slider_dspec_small_dmax.value = vmax_values_last[idx]
    tab3_Slider_dspec_small_dmin.start = mean_values[idx] - 2 * drange_values[
        idx]
    tab3_Slider_dspec_small_dmin.end = mean_values[idx] + drange_values[idx]
    tab3_Slider_dspec_small_dmin.value = vmin_values_last[idx]
    tab3_dspec_small_CTRLs_OPT['radio_button_group_dspec_small_update_flag'] = False


def tab3_RBG_dspec_small_handler(attrname, old, new):
    idx_p_dspec_small = tab3_RBG_dspec_small.active
    RBG_dspec_small_update(idx_p_dspec_small)


def tab3_BUT_dspec_small_reset_update():
    global VdspecDF, tab2_nfreq, tab2_ntim, tab3_dspec_small_CTRLs_OPT
    items_dspec_small = tab3_dspec_small_CTRLs_OPT['items_dspec_small']
    mean_values = tab3_dspec_small_CTRLs_OPT['mean_values']
    drange_values = tab3_dspec_small_CTRLs_OPT['drange_values']
    vmax_values = tab3_dspec_small_CTRLs_OPT['vmax_values']
    vmin_values = tab3_dspec_small_CTRLs_OPT['vmin_values']
    source_list = [tab3_r_dspec_vector, tab3_r_dspec_vectorx, tab3_r_dspec_vectory]
    for ll, item in enumerate(items_dspec_small):
        TmpData = (VdspecDF[item].copy()).values.reshape(tab2_nfreq, tab2_ntim)
        TmpData[TmpData > vmax_values[ll]] = vmax_values[ll]
        TmpData[TmpData < vmin_values[ll]] = vmin_values[ll]
        source_list[ll].data_source.data['image'] = [TmpData]
    idx_p_dspec_small = 0
    tab3_dspec_small_CTRLs_OPT['idx_p_dspec_small'] = idx_p_dspec_small
    tab3_RBG_dspec_small.active = idx_p_dspec_small
    tab3_Slider_dspec_small_dmax.start = mean_values[idx_p_dspec_small] - drange_values[idx_p_dspec_small]
    tab3_Slider_dspec_small_dmax.end = mean_values[idx_p_dspec_small] + 2 * drange_values[idx_p_dspec_small]
    tab3_Slider_dspec_small_dmax.value = vmax_values[idx_p_dspec_small]
    tab3_Slider_dspec_small_dmin.start = mean_values[idx_p_dspec_small] - 2 * drange_values[
        idx_p_dspec_small]
    tab3_Slider_dspec_small_dmin.end = mean_values[idx_p_dspec_small] + drange_values[idx_p_dspec_small]
    tab3_Slider_dspec_small_dmin.value = vmin_values[idx_p_dspec_small]
    tab3_dspec_small_CTRLs_OPT['vmax_values_last'] = [ll for ll in vmax_values]
    tab3_dspec_small_CTRLs_OPT['vmin_values_last'] = [ll for ll in vmin_values]
    vmax_vx, vmax_vy = tab3_dspec_small_CTRLs_OPT['vmax_values_last'][1:]
    vmin_vx, vmin_vy = tab3_dspec_small_CTRLs_OPT['vmin_values_last'][1:]
    tab3_r_aia_submap_rect.data_source.data['x'] = [(vmax_vx + vmin_vx) / 2]
    tab3_r_aia_submap_rect.data_source.data['y'] = [(vmax_vy + vmin_vy) / 2]
    tab3_r_aia_submap_rect.data_source.data['width'] = [(vmax_vx - vmin_vx)]
    tab3_r_aia_submap_rect.data_source.data['height'] = [(vmax_vy - vmin_vy)]


def tab3_BUT_dspec_small_resetall_update():
    VdspecDF_update()
    tab3_BUT_dspec_small_reset_update()
    print 'reset all'


def tab3_slider_dspec_small_update(attrname, old, new):
    global VdspecDF, tab2_nfreq, tab2_ntim, tab3_dspec_small_CTRLs_OPT
    items_dspec_small = tab3_dspec_small_CTRLs_OPT['items_dspec_small']
    idx_p_dspec_small = tab3_dspec_small_CTRLs_OPT['idx_p_dspec_small']
    dmax = tab3_Slider_dspec_small_dmax.value
    dmin = tab3_Slider_dspec_small_dmin.value
    if not tab3_dspec_small_CTRLs_OPT['radio_button_group_dspec_small_update_flag']:
        tab3_dspec_small_CTRLs_OPT['vmax_values_last'][idx_p_dspec_small] = dmax
        tab3_dspec_small_CTRLs_OPT['vmin_values_last'][idx_p_dspec_small] = dmin
    TmpData = (VdspecDF[items_dspec_small[idx_p_dspec_small]].copy()).values.reshape(tab2_nfreq, tab2_ntim)
    TmpData[TmpData > dmax] = dmax
    TmpData[TmpData < dmin] = dmin
    if idx_p_dspec_small == 0:
        tab3_r_dspec_vector.data_source.data['image'] = [TmpData]
    elif idx_p_dspec_small == 1:
        tab3_r_dspec_vectorx.data_source.data['image'] = [TmpData]
    elif idx_p_dspec_small == 2:
        tab3_r_dspec_vectory.data_source.data['image'] = [TmpData]
    vmax_vx, vmax_vy = tab3_dspec_small_CTRLs_OPT['vmax_values_last'][1:]
    vmin_vx, vmin_vy = tab3_dspec_small_CTRLs_OPT['vmin_values_last'][1:]
    tab3_r_aia_submap_rect.data_source.data['x'] = [(vmax_vx + vmin_vx) / 2]
    tab3_r_aia_submap_rect.data_source.data['y'] = [(vmax_vy + vmin_vy) / 2]
    tab3_r_aia_submap_rect.data_source.data['width'] = [(vmax_vx - vmin_vx)]
    tab3_r_aia_submap_rect.data_source.data['height'] = [(vmax_vy - vmin_vy)]


def tab3_slider_ANLYS_idx_update(attrname, old, new):
    global tab2_dtim, tab2_freq, tab2_ntim, SRC_maxfit_centroid
    if tab3_BUT_animate_ONOFF.label == 'Animate ON & Go':
        tab3_Slider_ANLYS_idx.start = next(
            i for i in xrange(tab2_ntim) if tab2_dtim[i] >= tab3_p_dspec_vector.x_range.start)
        tab3_Slider_ANLYS_idx.end = next(
            i for i in xrange(tab2_ntim - 1, -1, -1) if tab2_dtim[i] <= tab3_p_dspec_vector.x_range.end) + 1
        indices_time = tab3_Slider_ANLYS_idx.value
        tab3_r_dspec_vector_line.visible = True
        tab3_r_dspec_vector_line.data_source.data = ColumnDataSource(
            pd.DataFrame({'time': [tab2_dtim[indices_time], tab2_dtim[indices_time]],
                          'freq': [tab2_freq[0], tab2_freq[-1]]})).data
        try:
            tab3_r_aia_submap_cross.visible = True
            tab3_r_aia_submap_cross.data_source.data = SRC_maxfit_centroid[indices_time].data
        except:
            tab3_r_aia_submap_cross.visible = False
    else:
        tab3_Div_Tb.text = """<p><b>Warning: Animate is OFF!!!</b></p>"""


def tab2_panel3_savimgs_handler():
    dspecDF0POLsub = dspecDF0POL[dspecDF0POL['time'] >= tab3_p_dspec_vector.x_range.start][
        dspecDF0POL['time'] <= tab3_p_dspec_vector.x_range.end][
        dspecDF0POL['freq'] >= tab3_p_dspec_vector.y_range.start][
        dspecDF0POL['freq'] <= tab3_p_dspec_vector.y_range.end]
    timselseq = np.unique(dspecDF_select['time'])
    timseq = np.unique(dspecDF0POLsub['time'])
    subset_label = ['freq', 'shape_longitude', 'shape_latitude', 'timestr']
    nfiles = len(timseq)
    for sidx, ll in enumerate(timseq):
        timstr = dspecDF0POLsub[dspecDF0POLsub['time'] == ll]['timestr'].iloc[0]
        maponly = True
        centroids = {}
        centroids['freqran'] = [tab3_p_dspec_vector.y_range.start, tab3_p_dspec_vector.y_range.end]
        if ll in timselseq:
            dftmp = dspecDF_select[dspecDF_select.time == ll][subset_label]
            dftmp = dftmp.dropna(how='any', subset=subset_label)
            centroids['freq'] = dftmp['freq'].as_matrix()
            centroids['shape_longitude'] = dftmp['shape_longitude'].as_matrix()
            centroids['shape_latitude'] = dftmp['shape_latitude'].as_matrix()
            maponly = False
        ctplot.plotmap(centroids, aiamap_submap, outfile=outimgdir + timstr.replace(':', '') + '.png',
                       label='VLA ' + timstr,
                       x_range=[tab3_p_aia_submap.x_range.start, tab3_p_aia_submap.x_range.end],
                       y_range=[tab3_p_aia_submap.y_range.start, tab3_p_aia_submap.y_range.end], maponly=maponly)
        tab3_Div_Tb.text = """<p>{}</p>""".format(
            DButil.ProgressBar(sidx + 1, nfiles, suffix='Output', decimals=0, length=30, empfill='=', fill='#'))
    tab3_Div_Tb.text = '<p>images saved to <b>{}</b>.</p>'.format(outimgdir)


def tab2_panel3_dumpdata_handler():
    import Tkinter
    import tkFileDialog
    dspecDF0POLsub = dspecDF0POL[dspecDF0POL['time'] >= tab3_p_dspec_vector.x_range.start][
        dspecDF0POL['time'] <= tab3_p_dspec_vector.x_range.end][
        dspecDF0POL['freq'] >= tab3_p_dspec_vector.y_range.start][
        dspecDF0POL['freq'] <= tab3_p_dspec_vector.y_range.end]
    tarr = np.unique(dspecDF0POLsub['time'].as_matrix() + timestart)
    farr = np.unique(dspecDF0POLsub['freq'].as_matrix())
    nt = len(tarr)
    nf = len(farr)
    parr = dspecDF0POLsub['peak'].as_matrix().reshape(nf, nt)
    xarr = dspecDF0POLsub['shape_longitude'].as_matrix().reshape(nf, nt)
    yarr = dspecDF0POLsub['shape_latitude'].as_matrix().reshape(nf, nt)
    centroidsdict = {'time': tarr, 'freq': farr, 'peak': parr, 'x': xarr, 'y': yarr}
    centroids_save = 'centroids{}.npy'.format(tab2_Select_vla_pol.value)
    tkRoot = Tkinter.Tk()
    tkRoot.withdraw()  # Close the root window
    out_path = tkFileDialog.asksaveasfilename(initialdir=outimgdir, initialfile=centroids_save)
    tkRoot.destroy()
    if not out_path:
        out_path = outimgdir + centroids_save
    np.save(out_path, centroidsdict)
    tab3_Div_Tb.text = '<p>centroids info saved to <b>{}</b>.</p>'.format(out_path)


def tab3_BUT_plot_xargs_default():
    global tab3_plot_xargs_dict
    tab3_plot_xargs_dict = OrderedDict()
    tab3_plot_xargs_dict['timebin'] = "1"
    tab3_plot_xargs_dict['timeline'] = "False"
    tab3_Div_plot_xargs_text = '<p>' + ';'.join(
        "<b>{}</b> = {}".format(key, val) for (key, val) in tab3_plot_xargs_dict.items()) + '</p>'
    tab3_Div_plot_xargs.text = tab3_Div_plot_xargs_text
    tab3_Div_Tb.text = '<p><b>Default xargs Restored.</b></p>'


def tab3_animate_update():
    global tab3_animate_step, tab2_dspec_vector_selected
    if tab3_BUT_animate_ONOFF.label == 'Animate ON & Go':
        if tab2_dspec_vector_selected:
            indices_time = tab3_Slider_ANLYS_idx.value + tab3_animate_step
            if (tab3_animate_step == timebin) and (indices_time > tab3_Slider_ANLYS_idx.end):
                indices_time = tab3_Slider_ANLYS_idx.start
            if (tab3_animate_step == -timebin) and (indices_time < tab3_Slider_ANLYS_idx.start):
                indices_time = tab3_Slider_ANLYS_idx.end
            tab3_Slider_ANLYS_idx.value = indices_time
            tab3_Div_Tb.text = """ """
        else:
            tab3_Div_Tb.text = """<p><b>Warning: Select time and frequency from the Dynamic Spectrum first!!!</b></p>"""
    else:
        tab3_Div_Tb.text = """<p><b>Warning: Animate is OFF!!!</b></p>"""


def tab3_animate():
    global tab2_dspec_vector_selected
    if tab3_BUT_animate_ONOFF.label == 'Animate ON & Go':
        if tab3_BUT_PlayCTRL.label == 'Play':
            if tab2_dspec_vector_selected:
                tab3_BUT_PlayCTRL.label = 'Pause'
                tab3_BUT_PlayCTRL.button_type = 'danger'
                curdoc().add_periodic_callback(tab3_animate_update, 150)
                tab3_Div_Tb.text = """ """
            else:
                tab3_Div_Tb.text = """<p><b>Warning: Select time and frequency from the Dynamic Spectrum first!!!</b></p>"""
        else:
            tab3_BUT_PlayCTRL.label = 'Play'
            tab3_BUT_PlayCTRL.button_type = 'success'
            curdoc().remove_periodic_callback(tab3_animate_update)
    else:
        tab3_Div_Tb.text = """<p><b>Warning: Animate is OFF!!!</b></p>"""


def tab3_animate_step_CTRL():
    global tab3_animate_step, tab2_dspec_vector_selected
    if tab3_BUT_animate_ONOFF.label == 'Animate ON & Go':
        if tab2_dspec_vector_selected:
            if tab3_BUT_PlayCTRL.label == 'Pause':
                tab3_BUT_PlayCTRL.label = 'Play'
                tab3_BUT_PlayCTRL.button_type = 'success'
                curdoc().remove_periodic_callback(tab3_animate_update)
            idx = tab3_Slider_ANLYS_idx.value + tab3_animate_step
            if (tab3_animate_step == timebin) and (idx > tab3_Slider_ANLYS_idx.end):
                idx = tab3_Slider_ANLYS_idx.start
            elif (tab3_animate_step == -timebin) and (idx < tab3_Slider_ANLYS_idx.start):
                idx = tab3_Slider_ANLYS_idx.end
            tab3_Slider_ANLYS_idx.value = idx
            tab3_Div_Tb.text = """ """
        else:
            tab3_Div_Tb.text = """<p><b>Warning: Select time and frequency from the Dynamic Spectrum first!!!</b></p>"""
    else:
        tab3_Div_Tb.text = """<p><b>Warning: Animate is OFF!!!</b></p>"""


def tab3_animate_FRWD_REVS():
    global tab3_animate_step
    if tab3_BUT_animate_ONOFF.label == 'Animate ON & Go':
        if tab2_dspec_vector_selected:
            if tab3_animate_step == timebin:
                tab3_BUT_FRWD_REVS_CTRL.label = 'Reverse'
                tab3_animate_step = -timebin
            else:
                tab3_BUT_FRWD_REVS_CTRL.label = 'Forward'
                tab3_animate_step = timebin
            tab3_Div_Tb.text = """ """
        else:
            tab3_Div_Tb.text = """<p><b>Warning: Select time and frequency from the Dynamic Spectrum first!!!</b></p>"""
    else:
        tab3_Div_Tb.text = """<p><b>Warning: Animate is OFF!!!</b></p>"""


def tab3_animate_onoff():
    if tab2_dspec_vector_selected:
        global tab3_plot_xargs_dict, timebin, timeline
        if not 'timebin' in tab3_plot_xargs_dict.keys():
            tab3_plot_xargs_dict['timebin'] = '1'
        if not 'timeline' in tab3_plot_xargs_dict.keys():
            tab3_plot_xargs_dict['timeline'] = 'False'
        txts = tab3_input_plot_xargs.value.strip()
        txts = txts.split(';')
        for txt in txts:
            txt = txt.strip()
            txt = txt.split('=')
            if len(txt) == 2:
                key, val = txt
                key, val = key.strip(), val.strip()
                if key == 'timebin':
                    if not (0 <= int(val) <= tab2_ntim - 1):
                        val = '1'
                    timebin = int(val)
                if key == 'timeline':
                    if val not in ['True', 'False']:
                        val = 'False'
                    timeline = json.loads(val.lower())
                tab3_plot_xargs_dict[key.strip()] = val.strip()
                if key not in ['timebin', 'timeline']:
                    tab3_plot_xargs_dict.pop(key, None)
            else:
                tab3_Div_plot_xargs.text = '<p>Input syntax: <b>timebin</b>=1; <b>linesytle</b>=False;' \
                                           'Any spaces will be ignored.</p>'

        tab3_Div_plot_xargs_text = '<p>' + ';'.join(
            "<b>{}</b> = {}".format(key, val) for (key, val) in tab3_plot_xargs_dict.items()) + '</p>'
        tab3_Div_plot_xargs.text = tab3_Div_plot_xargs_text
        tab3_animate_step = timebin
        tab3_Slider_ANLYS_idx.step = timebin
        if tab3_BUT_animate_ONOFF.label == 'Animate ON & Go':
            tab3_BUT_animate_ONOFF.label = 'Animate OFF & Go'
            tab3_r_aia_submap_cross.visible = True
            tab3_r_aia_submap_line.visible = timeline
            tab3_r_dspec_vector_line.visible = False
            tab3_r_dspec_vectorx_line.visible = False
            tab3_r_dspec_vectory_line.visible = False
            tab2_SRC_maxfit_centroid_update(dspecDF_select)
            tab3_r_aia_submap_cross.data_source.data = SRC_maxfit_centroid.data
        else:
            tab3_BUT_animate_ONOFF.label = 'Animate ON & Go'
            tab3_r_aia_submap_cross.visible = True
            tab3_r_aia_submap_line.visible = False
            tab3_r_dspec_vector_line.visible = True
            tab3_r_dspec_vectorx_line.visible = True
            tab3_r_dspec_vectory_line.visible = True
            tab2_SRC_maxfit_centroid_update(dspecDF_select)
            indices_time = tab3_Slider_ANLYS_idx.value
            tab3_r_aia_submap_cross.data_source.data = SRC_maxfit_centroid[indices_time].data
            tab3_Div_Tb.text = """ """
    else:
        tab3_Div_Tb.text = """<p><b>Warning: Select time and frequency from the Dynamic Spectrum first!!!</b></p>"""


def tab2_panel_exit():
    tab2_panel2_Div_exit.text = """<p><b>You may close the tab anytime you like.</b></p>"""
    raise SystemExit


event_id = config_EvtID['datadir']['event_id']
event_dir = database_dir + event_id
try:
    infile = event_dir + 'CurrFS.json'
    FS_config = DButil.loadjsonfile(infile)
except:
    print 'Error: No CurrFS.json found!!!'
    raise SystemExit
struct_id = FS_config['datadir']['struct_id']
struct_dir = database_dir + event_id + struct_id
CleanID = FS_config['datadir']['clean_id']
CleanID_dir = struct_dir + CleanID
ImgfitID = FS_config['datadir']['imfit_id']
ImgfitID_dir = CleanID_dir + ImgfitID
outimgdir = ImgfitID_dir + '/img_centroids/'
if not os.path.exists(outimgdir):
    os.makedirs(outimgdir)
FS_dspecDF = ImgfitID_dir + 'dspecDF-save'
FS_specfile = FS_config['datadir']['FS_specfile']
tab2_specdata = np.load(FS_specfile)
tab2_spec = tab2_specdata['spec']
tab2_npol = tab2_specdata['npol']
tab2_nbl = tab2_specdata['nbl']
tab2_tim = tab2_specdata['tim']
tab2_dt = np.median(np.diff(tab2_tim))
tab2_freq = tab2_specdata['freq'] / 1e9
tab2_freq = [float('{:.03f}'.format(ll)) for ll in tab2_freq]
tab2_df = np.median(np.diff(tab2_freq))
tab2_ntim = len(tab2_tim)
tab2_nfreq = len(tab2_freq)

if isinstance(tab2_specdata['bl'].tolist(), str):
    tab2_bl = tab2_specdata['bl'].item().split(';')
elif isinstance(tab2_specdata['bl'].tolist(), list):
    tab2_bl = ['&'.join(ll) for ll in tab2_specdata['bl'].tolist()]
else:
    raise ValueError('Please check the data of {}'.format(FS_specfile))

tab2_dtim = tab2_tim - tab2_tim[0]
tab2_dur = tab2_dtim[-1] - tab2_dtim[0]
tim_map = ((np.tile(tab2_tim, tab2_nfreq).reshape(tab2_nfreq, tab2_ntim) / 3600. / 24. + 2400000.5)) * 86400.
freq_map = np.tile(tab2_freq, tab2_ntim).reshape(tab2_ntim, tab2_nfreq).swapaxes(0, 1)
xx = tim_map.flatten()
yy = freq_map.flatten()
timestart = xx[0]
fits_LOCL = config_EvtID['datadir']['fits_LOCL']
fits_GLOB = config_EvtID['datadir']['fits_GLOB']
fits_LOCL_dir = CleanID_dir + fits_LOCL
fits_GLOB_dir = CleanID_dir + fits_GLOB

if os.path.exists(FS_dspecDF):
    with open(FS_dspecDF, 'rb') as f:
        dspecDF0 = pickle.load(f)
    if DButil.getcolctinDF(dspecDF0, 'peak')[0] > 0:
        vlafile = glob.glob(fits_LOCL_dir + '*.fits')
        tab2_panel2_Div_exit = Div(text="""<p><b>Warning</b>: Click the <b>Exit FSview</b>
                                first before closing the tab</p></b>""",
                                   width=config_main['plot_config']['tab_FSview_base']['widgetbox_wdth'])
        # import the vla image
        hdu = read_fits(vlafile[0])
        pols = DButil.polsfromfitsheader(hdu.header)
        # initial dspecDF_select and dspecDF0POL
        dspecDF_select = DButil.dspecDFfilter(dspecDF0, pols[0])
        dspecDF0POL = dspecDF_select.copy()  # DButil.dspecDFfilter(dspecDF0, pols[0])

        tab2_SRC_maxfit_centroid_init(dspecDF_select)
        hdu_goodchan = goodchan(hdu)
        vla_local_pfmap = PuffinMap(hdu.data[0, hdu_goodchan[0], :, :], hdu.header,
                                    plot_height=config_main['plot_config']['tab_FSview_base']['vla_hght'],
                                    plot_width=config_main['plot_config']['tab_FSview_base']['vla_wdth'],
                                    webgl=config_main['plot_config']['WebGL'])
        # plot the contour of vla image
        mapx, mapy = vla_local_pfmap.meshgrid()
        mapx, mapy = mapx.value, mapy.value
        mapvlasize = mapy.shape
        tab2_SRC_vlamap_contour = DButil.get_contour_data(mapx, mapy, vla_local_pfmap.smap.data)
        # mapx2, mapy2 = vla_local_pfmap.meshgrid(rescale=0.5)
        # mapx2, mapy2 = mapx2.value, mapy2.value
        ImgDF0 = pd.DataFrame({'xx': mapx.ravel(), 'yy': mapy.ravel()})
        tab2_SRC_vla_square = ColumnDataSource(ImgDF0)
        colormap = cm.get_cmap("cubehelix")  # choose any matplotlib colormap here
        bokehpalette_SynthesisImg = [colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
        tab2_SRC_ImgRgn_Patch = ColumnDataSource(pd.DataFrame({'xx': [], 'yy': []}))

        # try:
        aiamap = DButil.readsdofile(datadir=SDOdir, wavelength='171', jdtime=xx[0] / 3600. / 24.,
                                    timtol=tab2_dur / 3600. / 24.)
        # except:
        # raise SystemExit('No SDO fits found under {}. '.format(SDOdir))

        lengthx = vla_local_pfmap.dw[0] * u.arcsec / 3.0
        lengthy = vla_local_pfmap.dh[0] * u.arcsec / 3.0
        x0 = vla_local_pfmap.smap.center.x
        y0 = vla_local_pfmap.smap.center.y
        aiamap_submap = aiamap.submap(u.Quantity([x0 - lengthx / 2, x0 + lengthx / 2]),
                                      u.Quantity([y0 - lengthy / 2, y0 + lengthy / 2]))

        # plot the detail AIA image
        aia_submap_pfmap = PuffinMap(smap=aiamap_submap,
                                     plot_height=config_main['plot_config']['tab_FSview_FitANLYS']['aia_submap_hght'],
                                     plot_width=config_main['plot_config']['tab_FSview_FitANLYS']['aia_submap_wdth'],
                                     webgl=config_main['plot_config']['WebGL'])

        # tab2_SRC_aia_submap_square = ColumnDataSource(ImgDF0)
        tab3_p_aia_submap, tab3_r_aia_submap = aia_submap_pfmap.PlotMap(DrawLimb=True, DrawGrid=True,
                                                                        grid_spacing=20 * u.deg,
                                                                        title='EM sources centroid map')

        tab3_p_aia_submap.border_fill_alpha = 0.4
        tab3_p_aia_submap.axis.major_tick_out = 0
        tab3_p_aia_submap.axis.major_tick_in = 5
        tab3_p_aia_submap.axis.minor_tick_out = 0
        tab3_p_aia_submap.axis.minor_tick_in = 3
        tab3_p_aia_submap.axis.major_tick_line_color = "white"
        tab3_p_aia_submap.axis.minor_tick_line_color = "white"
        color_mapper = LinearColorMapper(Spectral11)

        tab3_r_aia_submap_cross = tab3_p_aia_submap.cross(x='shape_longitude', y='shape_latitude', size=15,
                                                          color={'field': 'freq', 'transform': color_mapper},
                                                          line_width=3,
                                                          source=SRC_maxfit_centroid[tab2_dtim[0]], line_alpha=0.8)
        tab3_r_aia_submap_line = tab3_p_aia_submap.line(x='shape_longitude', y='shape_latitude', line_width=3,
                                                        line_color='black',
                                                        line_alpha=0.5,
                                                        source=SRC_maxfit_centroid[tab2_dtim[0]])
        tab3_p_aia_submap.add_tools(BoxSelectTool(renderers=[tab3_r_aia_submap_cross]))
        tab3_r_aia_submap_line.visible = False
        tab3_SRC_aia_submap_rect = ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})
        tab3_r_aia_submap_rect = tab3_p_aia_submap.rect(x='x', y='y', width='width', height='height', fill_alpha=0.1,
                                                        line_color='black', fill_color='black',
                                                        source=tab3_SRC_aia_submap_rect)

        tab2_Select_aia_wave = Select(title="Wavelenght:", value='171', options=['94', '131', '171'],
                                      width=config_main['plot_config']['tab_FSview_base']['widgetbox_wdth'])

        tab2_Select_aia_wave.on_change('value', aia_submap_wavelength_selection)

        # pols = ['RR', 'LL', 'I', 'V']
        SRL = set(['RR', 'LL'])
        SXY = set(['XX', 'YY', 'XY', 'YX'])
        Spol = set(pols)
        if hdu.header['NAXIS4'] == 2 and len(SRL.intersection(Spol)) == 2:
            pols = pols + ['I', 'V']
        if hdu.header['NAXIS4'] == 4 and len(SXY.intersection(Spol)) == 4:
            pols = pols + ['I', 'V']

        tab2_Select_vla_pol = Select(title="Polarization:", value=pols[0], options=pols,
                                     width=config_main['plot_config']['tab_FSview_base']['widgetbox_wdth'])
        select_vla_pol = tab2_Select_vla_pol.value

        tab2_Select_vla_pol.on_change('value', tab2_Select_vla_pol_update)

        tab2_LinkImg_HGHT = config_main['plot_config']['tab_FSview_base']['vla_hght']
        tab2_LinkImg_WDTH = config_main['plot_config']['tab_FSview_base']['vla_wdth']

        tab2_panel3_BUT_exit = Button(label='Exit FSview',
                                      width=config_main['plot_config']['tab_FSview_base']['widgetbox_wdth'],
                                      button_type='danger')
        tab2_panel3_BUT_exit.on_click(tab2_panel_exit)

        tab2_panel3_BUT_savimgs = Button(label='Save images',
                                         width=config_main['plot_config']['tab_FSview_base']['widgetbox_wdth'],
                                         button_type='primary')
        tab2_panel3_BUT_savimgs.on_click(tab2_panel3_savimgs_handler)

        tab2_panel3_BUT_dumpdata = Button(label='Dump data',
                                          width=config_main['plot_config']['tab_FSview_base']['widgetbox_wdth'],
                                          button_type='success')
        tab2_panel3_BUT_dumpdata.on_click(tab2_panel3_dumpdata_handler)

        tab3_p_dspec_vector = figure(tools='pan,wheel_zoom,box_zoom,save,reset',
                                     plot_width=config_main['plot_config']['tab_FSview_FitANLYS']['dspec_small_wdth'],
                                     plot_height=config_main['plot_config']['tab_FSview_FitANLYS']['dspec_small_hght'],
                                     x_range=(tab2_dtim[0] - tab2_dt / 2.0, tab2_dtim[-1] + tab2_dt / 2.0),
                                     y_range=(tab2_freq[0] - tab2_df / 2.0, tab2_freq[-1] + tab2_df / 2.0),
                                     toolbar_location='above')
        tab3_p_dspec_vectorx = figure(tools='pan,wheel_zoom,box_zoom,save,reset',
                                      plot_width=config_main['plot_config']['tab_FSview_FitANLYS']['dspec_small_wdth'],
                                      plot_height=config_main['plot_config']['tab_FSview_FitANLYS']['dspec_small_hght'],
                                      x_range=tab3_p_dspec_vector.x_range,
                                      y_range=tab3_p_dspec_vector.y_range, toolbar_location='above')
        tab3_p_dspec_vectory = figure(tools='pan,wheel_zoom,box_zoom,save,reset',
                                      plot_width=config_main['plot_config']['tab_FSview_FitANLYS']['dspec_small_wdth'],
                                      plot_height=config_main['plot_config']['tab_FSview_FitANLYS'][
                                                      'dspec_small_hght'] + 40,
                                      x_range=tab3_p_dspec_vector.x_range,
                                      y_range=tab3_p_dspec_vector.y_range, toolbar_location='above')
        tim0_char = Time(xx[0] / 3600. / 24., format='jd', scale='utc', precision=3, out_subfmt='date_hms').iso
        tab3_p_dspec_vector.xaxis.visible = False
        tab3_p_dspec_vectorx.xaxis.visible = False
        tab3_p_dspec_vector.title.text = "Vector Dynamic spectrum (Intensity)"
        tab3_p_dspec_vectorx.title.text = "Vector Dynamic spectrum (Vx)"
        tab3_p_dspec_vectory.title.text = "Vector Dynamic spectrum (Vy)"
        tab3_p_dspec_vectory.xaxis.axis_label = 'Seconds since ' + tim0_char
        tab3_p_dspec_vector.yaxis.axis_label = 'Frequency [GHz]'
        tab3_p_dspec_vectorx.yaxis.axis_label = 'Frequency [GHz]'
        tab3_p_dspec_vectory.yaxis.axis_label = 'Frequency [GHz]'
        # tab3_p_dspec_vector.border_fill_color = "silver"
        tab3_p_dspec_vector.border_fill_alpha = 0.4
        tab3_p_dspec_vector.axis.major_tick_out = 0
        tab3_p_dspec_vector.axis.major_tick_in = 5
        tab3_p_dspec_vector.axis.minor_tick_out = 0
        tab3_p_dspec_vector.axis.minor_tick_in = 3
        tab3_p_dspec_vector.axis.major_tick_line_color = "white"
        tab3_p_dspec_vector.axis.minor_tick_line_color = "white"
        # tab3_p_dspec_vectorx.border_fill_color = "silver"
        tab3_p_dspec_vectorx.border_fill_alpha = 0.4
        tab3_p_dspec_vectorx.axis.major_tick_out = 0
        tab3_p_dspec_vectorx.axis.major_tick_in = 5
        tab3_p_dspec_vectorx.axis.minor_tick_out = 0
        tab3_p_dspec_vectorx.axis.minor_tick_in = 3
        tab3_p_dspec_vectorx.axis.major_tick_line_color = "white"
        tab3_p_dspec_vectorx.axis.minor_tick_line_color = "white"
        # tab3_p_dspec_vectory.border_fill_color = "silver"
        tab3_p_dspec_vectory.border_fill_alpha = 0.4
        tab3_p_dspec_vectory.axis.major_tick_out = 0
        tab3_p_dspec_vectory.axis.major_tick_in = 5
        tab3_p_dspec_vectory.axis.minor_tick_out = 0
        tab3_p_dspec_vectory.axis.minor_tick_in = 3
        tab3_p_dspec_vectory.axis.major_tick_line_color = "white"
        tab3_p_dspec_vectory.axis.minor_tick_line_color = "white"
        tab3_p_dspec_vector.add_tools(BoxSelectTool())
        tab3_p_dspec_vector.add_tools(LassoSelectTool())
        tab3_p_dspec_vector.select(BoxSelectTool).select_every_mousemove = False
        tab3_p_dspec_vector.select(LassoSelectTool).select_every_mousemove = False
        tab3_r_aia_submap_cross.data_source.on_change('selected', tab3_aia_submap_cross_selection_change)

        VdspecDF_init()
        VdspecDF_update()
        tab3_SRC_dspec_vector_init()
        tab3_r_dspec_vector = tab3_p_dspec_vector.image(image=tab3_dspec_vector_img, x=tab2_dtim[0] - tab2_dt / 2.0,
                                                        y=tab2_freq[0] - tab2_df / 2.0,
                                                        dw=tab2_dur + tab2_dt,
                                                        dh=tab2_freq[-1] - tab2_freq[0] + tab2_df,
                                                        palette=bokehpalette_jet)
        tab3_r_dspec_vectorx = tab3_p_dspec_vectorx.image(image=tab3_dspec_vectorx_img, x=tab2_dtim[0] - tab2_dt / 2.0,
                                                          y=tab2_freq[0] - tab2_df / 2.0,
                                                          dw=tab2_dur + tab2_dt,
                                                          dh=tab2_freq[-1] - tab2_freq[0] + tab2_df,
                                                          palette=bokehpalette_jet)
        tab3_r_dspec_vectory = tab3_p_dspec_vectory.image(image=tab3_dspec_vectory_img, x=tab2_dtim[0] - tab2_dt / 2.0,
                                                          y=tab2_freq[0] - tab2_df / 2.0,
                                                          dw=tab2_dur + tab2_dt,
                                                          dh=tab2_freq[-1] - tab2_freq[0] + tab2_df,
                                                          palette=bokehpalette_jet)
        tab3_source_idx_line = ColumnDataSource(pd.DataFrame({'time': [], 'freq': []}))
        tab3_r_dspec_vector_line = tab3_p_dspec_vector.line(x='time', y='freq', line_width=1.5, line_alpha=0.8,
                                                            line_color='white', source=tab3_source_idx_line)
        tab3_r_dspec_vectorx_line = tab3_p_dspec_vectorx.line(x='time', y='freq', line_width=1.5, line_alpha=0.8,
                                                              line_color='white',
                                                              source=tab3_source_idx_line)
        tab3_r_dspec_vectory_line = tab3_p_dspec_vectory.line(x='time', y='freq', line_width=1.5, line_alpha=0.8,
                                                              line_color='white',
                                                              source=tab3_source_idx_line)
        tab2_SRC_dspec_vector_square = ColumnDataSource(dspecDF0POL)
        tab2_r_dspec_vector_square = tab3_p_dspec_vector.square('time', 'freq', source=tab2_SRC_dspec_vector_square,
                                                                fill_color=None,
                                                                fill_alpha=0.0,
                                                                line_color=None, line_alpha=0.0,
                                                                selection_fill_alpha=0.2,
                                                                selection_fill_color='black',
                                                                nonselection_fill_alpha=0.0,
                                                                selection_line_alpha=0.0, selection_line_color='white',
                                                                nonselection_line_alpha=0.0,
                                                                size=min(
                                                                    config_main['plot_config']['tab_FSview_FitANLYS'][
                                                                        'dspec_small_wdth'] / tab2_ntim,
                                                                    config_main['plot_config']['tab_FSview_FitANLYS'][
                                                                        'dspec_small_hght'] / tab2_nfreq))

        tab2_dspec_vector_selected = None
        tab2_SRC_dspec_vector_square.on_change('selected', tab2_dspec_vector_selection_change)
        tab3_dspec_small_CTRLs_OPT = dict(mean_values=[mean_amp_g, mean_vx, mean_vy],
                                          drange_values=[drange_amp_g, drange_vx, drange_vy],
                                          vmax_values=[vmax_amp_g, vmax_vx, vmax_vy],
                                          vmin_values=[vmin_amp_g, vmin_vx, vmin_vy],
                                          vmax_values_last=[vmax_amp_g, vmax_vx, vmax_vy],
                                          vmin_values_last=[vmin_amp_g, vmin_vx, vmin_vy],
                                          items_dspec_small=['peak', 'shape_longitude', 'shape_latitude'],
                                          labels_dspec_small=["Flux", "X-pos", "Y-pos"], idx_p_dspec_small=0,
                                          radio_button_group_dspec_small_update_flag=False)

        tab3_RBG_dspec_small = RadioButtonGroup(labels=tab3_dspec_small_CTRLs_OPT['labels_dspec_small'], active=0)
        tab3_BUT_dspec_small_reset = Button(label='Reset DRange',
                                            width=config_main['plot_config']['tab_FSview_base']['widgetbox_wdth'])
        tab3_Slider_dspec_small_dmax = Slider(start=mean_amp_g, end=mean_amp_g + 2 * drange_amp_g, value=vmax_amp_g,
                                              step=1, title='dmax', callback_throttle=250)
        tab3_Slider_dspec_small_dmin = Slider(start=mean_amp_g - 2 * drange_amp_g, end=mean_amp_g, value=vmin_amp_g,
                                              step=1, title='dmin', callback_throttle=250)
        thresholdrange = (np.floor(dspecDF0POL['peak'].min()), np.ceil(dspecDF0POL['peak'].max()))
        tab3_rSlider_threshold = RangeSlider(start=thresholdrange[0], end=thresholdrange[1], range=thresholdrange,
                                             step=1, title='flux threshold selection', callback_throttle=250, width=400)
        tab3_rSlider_threshold.on_change('range', rSlider_threshold_handler)
        tab3_RBG_dspec_small.on_change('active', tab3_RBG_dspec_small_handler)
        tab3_BUT_dspec_small_reset.on_click(tab3_BUT_dspec_small_reset_update)
        tab3_BUT_dspec_small_resetall = Button(label='Reset All',
                                               width=config_main['plot_config']['tab_FSview_base']['widgetbox_wdth'])
        tab3_BUT_dspec_small_resetall.on_click(tab3_BUT_dspec_small_resetall_update)
        tab3_CTRLs_dspec_small = [tab3_Slider_dspec_small_dmax, tab3_Slider_dspec_small_dmin]
        for ctrl in tab3_CTRLs_dspec_small:
            ctrl.on_change('value', tab3_slider_dspec_small_update)

        tab3_RBG_TimeFreq = RadioButtonGroup(labels=["time", "freq"], active=0)
        tab3_Slider_ANLYS_idx = Slider(start=0, end=tab2_ntim - 1, value=0, step=1, title="time idx", width=450)
        tab3_Slider_ANLYS_idx.on_change('value', tab3_slider_ANLYS_idx_update)
        tab3_Div_Tb = Div(text=""" """, width=400)
        timebin = 1
        timeline = False
        tab3_animate_step = timebin
        tab3_BUT_PlayCTRL = Button(label='Play', width=60, button_type='success')
        tab3_BUT_PlayCTRL.on_click(tab3_animate)
        tab3_BUT_StepCTRL = Button(label='Step', width=60, button_type='primary')
        tab3_BUT_StepCTRL.on_click(tab3_animate_step_CTRL)
        tab3_BUT_FRWD_REVS_CTRL = Button(label='Forward', width=60, button_type='warning')
        tab3_BUT_FRWD_REVS_CTRL.on_click(tab3_animate_FRWD_REVS)
        tab3_BUT_animate_ONOFF = Button(label='Animate ON & Go', width=80)
        tab3_BUT_animate_ONOFF.on_click(tab3_animate_onoff)
        tab3_Div_plot_xargs = Div(text='', width=300)
        tab3_BUT_plot_xargs_default()
        tab3_SPCR_LFT_BUT_Step = Spacer(width=10, height=10)
        tab3_SPCR_LFT_BUT_REVS_CTRL = Spacer(width=10, height=10)
        tab3_SPCR_LFT_BUT_animate_ONOFF = Spacer(width=20, height=10)
        tab3_input_plot_xargs = TextInput(value='Input the param here', title="Plot parameters:", width=300)
        # todo add RCP LCP check box
        tab3_CheckboxGroup_pol = CheckboxGroup(labels=["RCP", "LCP"], active=[0, 1])


        def Buttonaskdir_handler():
            import Tkinter
            import tkFileDialog
            global outimgdir
            tkRoot = Tkinter.Tk()
            tkRoot.withdraw()  # Close the root window
            outdir = tkFileDialog.askdirectory(initialdir=outimgdir, parent=tkRoot) + '/'
            if outdir:
                outimgdir = outdir
            tkRoot.destroy()


        But_outdir = Button(label='outpath', width=config_main['plot_config']['tab_FSview_base']['widgetbox_wdth'])
        But_outdir.on_click(Buttonaskdir_handler)

        lout3_1 = column(tab3_p_aia_submap, tab3_Slider_ANLYS_idx,
                         row(tab3_BUT_PlayCTRL, tab3_SPCR_LFT_BUT_Step, tab3_BUT_StepCTRL,
                             tab3_SPCR_LFT_BUT_REVS_CTRL,
                             tab3_BUT_FRWD_REVS_CTRL, tab3_SPCR_LFT_BUT_animate_ONOFF,
                             tab3_BUT_animate_ONOFF), tab3_input_plot_xargs, tab3_Div_plot_xargs)
        lout3_2 = column(gridplot([tab3_p_dspec_vector], [tab3_p_dspec_vectorx], [tab3_p_dspec_vectory],
                                  toolbar_location='right'), tab3_Div_Tb)
        lout3_3 = widgetbox(tab3_RBG_dspec_small, tab3_Slider_dspec_small_dmax, tab3_Slider_dspec_small_dmin,
                            tab3_BUT_dspec_small_reset, tab3_BUT_dspec_small_resetall, tab3_rSlider_threshold,
                            tab2_Select_vla_pol, tab2_Select_aia_wave, But_outdir, tab2_panel3_BUT_savimgs,
                            tab2_panel3_BUT_dumpdata, tab2_panel3_BUT_exit, tab2_panel2_Div_exit,
                            width=200)
        # todo dump vdspec data
        # todo add dspec and contour to saveimage

        lout = row(lout3_1, lout3_2, lout3_3)

        curdoc().add_root(lout)
        curdoc().title = "VDSpec"
    else:
        raise SystemExit
else:
    raise SystemExit
