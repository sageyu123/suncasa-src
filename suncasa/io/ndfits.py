import copy
import os
import numpy as np
from astropy.io import fits
from sunpy import map as smap
import warnings
import sys

warnings.simplefilter("ignore")

stokesval = {'1': 'I', '2': 'Q', '3': 'U', '4': 'V', '-1': 'RR', '-2': 'LL', '-3': 'RL', '-4': 'LR', '-5': 'XX',
             '-6': 'YY', '-7': 'XY', '-8': 'YX'}


def is_compressed_fits(fitsfile):
    '''
    Function to check if the FITS file contains compressed data
    '''
    with fits.open(fitsfile) as hdul:
        for hdu in hdul:
            if isinstance(hdu, fits.hdu.CompImageHDU):
                return True
        return False


def headerfix(header, PC_coor=True):
    '''
	this code fixes the header problem of fits out from CASA 5.4+ which leads to a streched solar image.
    Setting PC_coor equal to True will reset the rotation matrix.
    '''

    keys2remove = []
    for k in header:
        if k.upper().startswith('PC'):
            if not k.upper().startswith('PC0'):
                pcidxs = k.upper().replace('PC', '')
                hd_ = 'PC0' + pcidxs
                keys2remove.append(k)
                if PC_coor:
                    pcidx0, pcidx1 = pcidxs.split('_')
                    if pcidx0 == pcidx1:
                        header[hd_] = 1.0
                    else:
                        header[hd_] = 0.0
                else:
                    header[hd_] = header[k]
    for k in keys2remove:
        header.remove(k)
    return header


def headerparse(header):
    '''
        get axis index of polarization
    '''

    ndim = header['NAXIS']
    stokesinfo = {'axis': None, 'headernew': {}}
    keys2analy = ['NAXIS', 'CTYPE', 'CRVAL', 'CDELT', 'CRPIX', 'CUNIT']
    for dim in range(1, ndim + 1):
        k = 'CTYPE{}'.format(dim)
        if header[k].startswith('STOKES'):
            stokesinfo['axis'] = dim
    if stokesinfo['axis'] is not None:
        dim = stokesinfo['axis']
        for k in keys2analy:
            k_ = '{}{}'.format(k, dim)
            stokesinfo['headernew']['{}{}'.format(k, 4)] = header[k_]
    return stokesinfo


def headersqueeze(header, data):
    """
    Squeezes single-dimensional entries from an n-dimensional FITS image data array and updates the FITS header accordingly.

    This function is useful for preparing image data for astropy fits compression, which only supports 1D, 2D, or 3D images. It removes
    any single-dimensional entries from the shape of the data array and updates the corresponding FITS header keys to reflect the
    new dimensions.

    :param header: FITS header object containing the metadata of the image.
    :type header: astropy.io.fits.Header
    :param data: n-dimensional image data array.
    :type data: numpy.ndarray

    :return: A tuple of the updated header object and the squeezed data array.
    :rtype: (astropy.io.fits.Header, numpy.ndarray)

    .. note::
        This function only updates the header keys related to dimensions, coordinate types, values, increments, reference pixels,
        and units. Any specific header keys related to coordinate transformations (e.g., PC matrix) for dimensions higher than
        the third are also updated if necessary. The function does not handle higher-order WCS transformations beyond simple axis
        permutations and squeezes.
    """
    dshape = data.shape
    ndim = data.ndim
    # Count single-dimensional entries in the data shape
    nsdim = np.count_nonzero(np.array(dshape) == 1)
    # Calculate the number of non-single-dimensional entries
    nonsdim = ndim - nsdim

    # If there are no single-dimensional entries, return the original header and data
    if nsdim == 0:
        return header, data
    else:
        # Define header keys that might need to be changed due to squeezing
        keys2chng = ['NAXIS', 'CTYPE', 'CRVAL', 'CDELT', 'CRPIX', 'CUNIT']  # ,'PC01_', 'PC02_', 'PC03_', 'PC04_']

        idx_nonsdim = 0
        for idx, dim in enumerate(dshape[::-1]):
            # if dim>1: continue
            if dim > 1:
                idx_nonsdim = idx_nonsdim + 1
            for k in keys2chng:
                k_ = f'{k}{idx + 1}'
                v = header[k_]
                header.remove(k_)
                if dim > 1:
                    k_new = f'{k}{idx_nonsdim}'
                    header[k_new] = v
                else:
                    if k == 'CTYPE' and v.startswith('STOKES'):
                        header['STOKES'] = header[f'CRVAL{idx + 1}']

        idx_nonsdim1 = 0
        # Update PC matrix keys for non-single-dimensional entries
        for idx1, dim1 in enumerate(dshape[::-1]):
            if dim1 > 1:
                idx_nonsdim1 = idx_nonsdim1 + 1
            idx_nonsdim2 = 0
            for idx2, dim2 in enumerate(dshape[::-1]):
                if dim2 > 1:
                    idx_nonsdim2 = idx_nonsdim2 + 1
                k_ = f'PC{idx1 + 1:02d}_{idx2 + 1}'
                if k_ in header.keys():
                    v = header[k_]
                    header.remove(k_)
                    if dim1 > 1 and dim2 > 1:
                        k_new = f'PC{idx_nonsdim1:02d}_{idx_nonsdim2}'
                        header[k_new] = v

        header['NAXIS'] = nonsdim
        data = np.squeeze(data)
        return header, data


def get_bdinfo(freq, bw):
    """
    get band information from center frequencies and band widths.

    Parameters
    ----------
    freq : array_like
        an array of the center frequencies of all frequency bands in Hz
    bw: array_like
        an array of the band widths of all frequency bands in Hz

    Returns
    -------
    fbounds : `dict`
        A dict of band information
    """
    fghz = freq / 1.e9
    bwghz = bw / 1.e9
    bounds_lo = fghz - bwghz / 2.0
    bounds_hi = fghz + bwghz / 2.0
    bounds_all = np.hstack([bounds_lo, bounds_hi[-1]])
    fbounds = {'cfreqs': fghz, 'cfreqs_all': fghz, 'bounds_lo': bounds_lo,
               'bounds_hi': bounds_hi, 'bounds_all': bounds_all}
    return fbounds


def read(filepath, hdus=None, verbose=False, **kwargs):
    """
    Read a fits file.

    Parameters
    ----------
    filepath : `str`
        The fits file to be read.
    hdus: `int` or iterable
        The HDU indexes to read from the file.
    verbose: `bool`
        if verbose

    Returns
    -------
    pairs : `list`
        A list of (data, header) tuples

    Notes
    -----
    This routine reads all the HDU's in a fits file and returns a list of the
    data and a FileHeader instance for each one.

    Also all comments in the original file are concatenated into a single
    "comment" key in the returned FileHeader.
    """
    import collections

    with fits.open(filepath, ignore_blank=True) as hdulist:
        if hdus is not None:
            if isinstance(hdus, int):
                hdulist = hdulist[hdus]
            elif isinstance(hdus, collections.Iterable):
                hdulist = [hdulist[i] for i in hdus]

        hdulist = fits.hdu.HDUList(hdulist)
        for h in hdulist:
            h.verify('silentfix+warn')

        meta = {}
        data = None
        # Process primary HDUs to extract image/map metadata and data
        for i, hdu in enumerate(hdulist):
            try:
                header = hdu.header
                arr = hdu.data
                if arr is None:
                    continue

                ndim = arr.ndim
                slc = [slice(None)] * ndim
                freq_axis = None
                pol_axis = None
                npol = 1
                nfreq = 1

                # Identify frequency and polarization axes
                for idx in range(ndim):
                    ctype = header.get(f'CTYPE{idx+1}', '')
                    if ctype.startswith('FREQ'):
                        freq_axis = ndim - (idx + 1)
                        nfreq = header.get(f'NAXIS{idx+1}', 1)
                    if ctype.startswith('STOKES'):
                        pol_axis = ndim - (idx + 1)
                        npol = header.get(f'NAXIS{idx+1}', 1)

                # Build frequency metadata
                if freq_axis is not None:
                    slc[freq_axis] = slice(0, 1)
                    vals = header.get(f'NAXIS{ndim-freq_axis}', 1)
                    crval = header.get(f'CRVAL{ndim-freq_axis}', 0.0)
                    cdelt = header.get(f'CDELT{ndim-freq_axis}', 1.0)
                    meta['ref_cfreqs'] = crval + cdelt * np.arange(vals)
                    meta['ref_freqdelts'] = np.ones(vals) * cdelt
                else:
                    restfrq = header.get('RESTFRQ', 0.0)
                    meta['ref_cfreqs'] = np.array([restfrq])
                    meta['ref_freqdelts'] = np.array([0.0])

                # Build polarization metadata
                if pol_axis is not None:
                    slc[pol_axis] = slice(0, 1)
                    vals = header.get(f'NAXIS{ndim-pol_axis}', 1)
                    crval = header.get(f'CRVAL{ndim-pol_axis}', 0.0)
                    cdelt = header.get(f'CDELT{ndim-pol_axis}', 1.0)
                    meta['pol_idxs'] = crval + cdelt * np.arange(vals)
                    # Assume a global stokesval dict exists or adapt as needed
                    meta['pol_names'] = [stokesval.get(str(int(p)), '') for p in meta['pol_idxs']]

                # Clean NaNs and build a SunPy map for the first slice
                clean_arr = np.nan_to_num(arr)
                rmap = smap.Map(np.squeeze(clean_arr[tuple(slc)]), header)

                data = clean_arr.copy()
                # Populate basic metadata
                meta.update({
                    'header': header.copy(),
                    'refmap': rmap,
                    'naxis': ndim,
                    'nx': header.get('NAXIS1', 0),
                    'ny': header.get('NAXIS2', 0),
                    'hgln_axis': ndim - 1,
                    'hglt_axis': ndim - 2,
                    'freq_axis': freq_axis,
                    'nfreq': nfreq,
                    'pol_axis': pol_axis,
                    'npol': npol,
                })
                break
            except Exception as e:
                if verbose:
                    print(e)
                    print(f'skipped HDU {i}')
                meta, data = {}, None

        # Check for additional table columns in the last HDU and map keys as needed
        last_data = hdulist[-1].data
        if hasattr(last_data, 'dtype') and last_data.dtype.names:
            key_map = {
                'cfreqs': 'ref_cfreqs',
                'cdelts': 'ref_freqdelts',
                'cbmaj': 'bmaj',
                'cbmin': 'bmin',
                'cbpa': 'bpa',
            }
            for key in last_data.dtype.names:
                arr = np.array(last_data[key])
                out_key = key_map.get(key, key)
                meta[out_key] = arr
        elif verbose:
            print('No table fields found in the last HDU; no additional columns added.')

        hdulist.close()
    return meta, data



def write(fname, data, header, mask=None, fix_invalid=True, filled_value=0.0, overwrite=True, **kwargs):
    """
    Take a data header pair and write a compressed FITS file.
    Caveat: only 1D, 2D, or 3D images are currently supported by Astropy fits compression.
    To be compressed, the image data array (n-dimensional) must have
    at least n-3 single-dimensional entries.

    Parameters
    ----------
    fname : `str`
        File name, with extension.
    data : `numpy.ndarray`
        n-dimensional data array.
    header : `dict`
        A header dictionary.
    compression_type: `str`, optional
        Compression algorithm: one of 'RICE_1', 'RICE_ONE', 'PLIO_1', 'GZIP_1', 'GZIP_2', 'HCOMPRESS_1'
    hcomp_scale: `float`, optional
        HCOMPRESS scale parameter
    """

    dshape = data.shape
    dim = data.ndim
    if dim - np.count_nonzero(np.array(dshape) == 1) > 3:
        return 0
    else:
        if fix_invalid:
            data[np.isnan(data)] = filled_value
        if kwargs is {}:
            kwargs.update({'compression_type': 'RICE_1', 'quantize_level': 4.0})
        if isinstance(fname, str):
            fname = os.path.expanduser(fname)

        if os.path.exists(fname):
            if overwrite:
                os.system('rm -rf {}'.format(fname))
            else:
                print('File exists. Set overwrite=True to overwrite the file.')
                return 0
        header, data = headersqueeze(header, data)
        hdunew = fits.CompImageHDU(data=data, header=header, **kwargs)
        if mask is None:
            hdulnew = fits.HDUList([fits.PrimaryHDU(), hdunew])
        else:
            hdumask = fits.CompImageHDU(data=mask.astype(np.uint8), **kwargs)
            hdulnew = fits.HDUList([fits.PrimaryHDU(), hdunew, hdumask])
        hdulnew.writeto(fname, output_verify='fix')
        return 1


def header_to_xml(header):
    import xml.etree.ElementTree as ET

    from datetime import datetime
    dt = datetime.now()
    formatted_dt = dt.strftime("%a %b %d %H:%M:%S %Y")

    # create the file structure
    tree = ET.ElementTree()
    root = ET.Element('meta')

    # Add fits section
    elem = ET.Element('fits')
    for k, v in header.items():
        child = ET.Element(k)
        if isinstance(v, bool):
            v = int(v)
        child.text = str(v)
        elem.append(child)
    root.append(elem)

    # Add helioviewer section
    hv_comment = f"""
    JP2 file generated by {__file__} on {formatted_dt} at EOVSA (NJIT).
    For inquiries or feedback regarding this file, please contact Sijie Yu at sijie.yu@njit.edu.
    The conversion relies on the Glymur library. Source code and documentation for suncasa.io.ndfits are available at https://github.com/suncasa/suncasa-src.
    Report any code-related issues to the repository maintainers.
    """
    print(hv_comment)
    helioviewer_elem = ET.Element('helioviewer')
    hv_comment_elem = ET.Element('HV_COMMENT')
    hv_comment_elem.text = hv_comment
    helioviewer_elem.append(hv_comment_elem)
    root.append(helioviewer_elem)

    tree._setroot(root)
    return tree
    # from lxml import etree
    # tree = etree.Element("meta")
    # elem = etree.SubElement(tree,"fits")
    # for k,v in header.items():
    #     child = etree.SubElement(elem, k)
    #     if isinstance(v,bool):
    #         v = int(v)
    #     child.text = str(v)
    # return tree


def write_j2000_image(fname, data, header):
    ## todo: write scaled data to jp2. so hv doesn't have to to scale.
    ## todo: add date and time to filename following 2024_03_06__14_18_45_343__EOVSA_1.5GHz.jp2
    ## todo: flare iamges: do similar thing as the CME tag for ccmc data, eOVSa provide movie links to hv. hv will show flare tag
    ## on the solar image and include movie as a external link
    import glymur
    datamax = np.max(data)
    datamin = np.min(data)

    jp2 = glymur.Jp2k(fname + '.tmp.jp2',
                      ((data - datamin) * 256 / (datamax - datamin)).astype(np.uint8), cratios=[20, 10])
    boxes = jp2.box
    header['wavelnth'] = header['crval3']
    header['waveunit'] = header['cunit3']
    header['datamax'] = datamax
    header['datamin'] = datamin
    xmlobj = header_to_xml(header)
    xmlfile = 'image.xml'
    if os.path.exists(xmlfile):
        os.remove(xmlfile)

    xmlobj.write(xmlfile)
    xmlbox = glymur.jp2box.XMLBox(filename='image.xml')
    boxes.insert(3, xmlbox)
    jp2_xml = jp2.wrap(fname, boxes=boxes)

    os.remove(fname + '.tmp.jp2')
    os.remove(xmlfile)
    return True


def wrap(fitsfiles, outfitsfile=None, docompress=False, mask=None, fix_invalid=True, filled_value=0.0, observatory=None,
         imres=None, verbose=False, **kwargs):
    '''
    wrap single frequency fits files into a multiple frequencies fits file
    '''
    from astropy.time import Time
    if len(fitsfiles) <= 1:
        print('There is only one files in the fits file list. wrap is aborted!')
        return ''
    else:
        try:
            fitsfiles = np.array(fitsfiles)
            num_files = len(fitsfiles)
            freqs = np.zeros(num_files)
            for i in range(num_files):
                head = fits.getheader(fitsfiles[i])
                freqs[i] = head['CRVAL3']
                del head
            pos = np.argsort(freqs)
            fitsfiles = fitsfiles[pos]
        except:
            fitsfiles = sorted(fitsfiles)
        nband = len(fitsfiles)
        fits_exist = []
        idx_fits_exist = []
        for sidx, fitsf in enumerate(fitsfiles):
            if os.path.exists(fitsf):
                fits_exist.append(fitsf)
                idx_fits_exist.append(sidx)
        if len(fits_exist) == 0: raise ValueError('None of the input fitsfiles exists!')
        if outfitsfile is None:
            hdu = fits.open(fits_exist[0])
            if observatory is None:
                try:
                    observatory = hdu[-1].header['TELESCOP']
                except:
                    observatory = 'RADIO'
                    print('Failed to acquire telescope information. set as RADIO')
            outfitsfile = Time(hdu[-1].header['DATE-OBS']).strftime(
                '{}.%Y%m%dT%H%M%S.%f.allbd.fits'.format(observatory))
            hdu.close()
        os.system('cp {} {}'.format(fits_exist[0], outfitsfile))
        hdu0 = fits.open(outfitsfile, mode='update')
        header = hdu0[-1].header

        if header['NAXIS'] != 4:
            if verbose:
                print('s1')
            if imres is None:
                cdelts = [325.e8] * nband
                print(
                    'Failed to read bandwidth information. Set as 325 MHz assuming this is EOVSA data. Use the value of keyword CDELT3 with caution.')
            else:
                cdelts = np.squeeze(np.diff(np.array(imres['freq']), axis=1)) * 1e9
            stokesinfo = headerparse(header)

            if stokesinfo['axis'] is None:
                npol = 1
            else:
                npol = int(header['NAXIS{}'.format(stokesinfo['axis'])])
                if docompress:
                    print(f'Warning: only 1D, 2D, or 3D images are currently supported for compression.  {npol} polarization(s) found in the input fits files. Only the first polarization will be written to the output file.')
                    npol=1
            nbd = nband
            ny = int(header['NAXIS2'])
            nx = int(header['NAXIS1'])

            data = np.zeros((npol, nbd, ny, nx))
            cfreqs = []
            cbmaj = []
            cbmin = []
            cbpa = []
            for sidx, fitsf in enumerate(fits_exist):
                hdu = fits.open(fitsf)
                cfreqs.append(hdu[-1].header['RESTFRQ'])
                cbmaj.append(hdu[-1].header['BMAJ'])
                cbmin.append(hdu[-1].header['BMIN'])
                cbpa.append(hdu[-1].header['BPA'])
                for pidx in range(npol):
                    if hdu[-1].data.ndim == 2:
                        data[pidx, idx_fits_exist[sidx], :, :] = hdu[-1].data
                    elif hdu[-1].data.ndim == 3:
                        data[pidx, idx_fits_exist[sidx], :, :] = hdu[-1].data[pidx, :, :]
                    else:
                        data[pidx, idx_fits_exist[sidx], :, :] = hdu[-1].data[pidx, 0, :, :]
            cfreqs = np.array(cfreqs)
            cdelts = np.array(cdelts)
            cbmaj = np.array(cbmaj)
            cbmin = np.array(cbmin)
            cbpa = np.array(cbpa)
            indfreq = np.argsort(cfreqs)
            cfreqs = cfreqs[indfreq]
            cdelts = cdelts[indfreq]
            cbmaj = cbmaj[indfreq]
            cbmin = cbmin[indfreq]
            cbpa = cbpa[indfreq]
            for pidx in range(npol):
                data[pidx, ...] = data[pidx, indfreq]

            df = np.nanmean(np.diff(cfreqs) / np.diff(idx_fits_exist))  ## in case some of the band is missing
            header['NAXIS'] = 4
            header['NAXIS3'] = nband
            header['CTYPE3'] = 'FREQ'
            header['CRVAL3'] = cfreqs[0]
            header['CDELT3'] = df
            header['CRPIX3'] = 1.0
            header['CUNIT3'] = 'Hz      '
            if stokesinfo['axis'] is None:
                header['NAXIS4'] = 1
                header['CTYPE4'] = 'STOKES'
                header['CRVAL4'] = -5  ## assume XX
                header['CDELT4'] = 1.0
                header['CRPIX4'] = 1.0
                header['CUNIT4'] = '        '
            else:
                for k, v in stokesinfo['headernew'].items():
                    print(k, v)
                    header[k] = v
        else:
            if verbose:
                print('s2')
            npol = int(header['NAXIS4'])
            nbd = nband
            ny = int(header['NAXIS2'])
            nx = int(header['NAXIS1'])

            data = np.zeros((npol, nbd, ny, nx))
            cdelts = []
            cfreqs = []
            cbmaj = []
            cbmin = []
            cbpa = []
            for sidx, fitsf in enumerate(fits_exist):
                hdu = fits.open(fitsf)

                cdelts.append(hdu[-1].header['CDELT3'])
                cfreqs.append(hdu[-1].header['CRVAL3'])
                cbmaj.append(hdu[-1].header['BMAJ'])
                cbmin.append(hdu[-1].header['BMIN'])
                cbpa.append(hdu[-1].header['BPA'])

                for pidx in range(npol):
                    if hdu[-1].data.ndim == 2:
                        data[pidx, idx_fits_exist[sidx], :, :] = hdu[-1].data
                    else:
                        data[pidx, idx_fits_exist[sidx], :, :] = hdu[-1].data[pidx, 0, :, :]

            cfreqs = np.array(cfreqs)
            cdelts = np.array(cdelts)
            cbmaj = np.array(cbmaj)
            cbmin = np.array(cbmin)
            cbpa = np.array(cbpa)

            df = np.nanmean(np.diff(cfreqs) / np.diff(idx_fits_exist))  ## in case some of the band is missing
            header['cdelt3'] = df
            header['NAXIS3'] = nband
            header['NAXIS'] = 4
            header['CRVAL3'] = header['CRVAL3'] - df * idx_fits_exist[0]

        for dim1 in range(1, header['NAXIS'] + 1):
            for dim2 in range(1, header['NAXIS'] + 1):
                k = 'PC{:02d}_{:d}'.format(dim1, dim2)
                if dim1 == dim2:
                    header[k] = 1.0
                else:
                    header[k] = 0.0

        if os.path.exists(outfitsfile):
            os.system('rm -rf {}'.format(outfitsfile))

        col1 = fits.Column(name='cfreqs', format='E', array=cfreqs)
        col2 = fits.Column(name='cdelts', format='E', array=cdelts)
        col3 = fits.Column(name='bmaj', format='E', array=cbmaj)
        col4 = fits.Column(name='bmin', format='E', array=cbmin)
        col5 = fits.Column(name='bpa', format='E', array=cbpa)
        tbhdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5])

        if docompress:
            if fix_invalid:
                data[np.isnan(data)] = filled_value
            if kwargs is {}:
                kwargs.update({'compression_type': 'RICE_1', 'quantize_level': 4.0})
            if isinstance(outfitsfile, str):
                outfitsfile = os.path.expanduser(outfitsfile)

            header, data = headersqueeze(header, data)
            if data.ndim == 4:
                print('only 1D, 2D, or 3D images are currently supported for compression. Aborting compression...')
            else:
                hdunew = fits.CompImageHDU(data=data, header=header, **kwargs)

                if mask is None:
                    hdulnew = fits.HDUList([fits.PrimaryHDU(), hdunew, tbhdu])
                else:
                    hdumask = fits.CompImageHDU(data=mask.astype(np.uint8), **kwargs)
                    hdulnew = fits.HDUList([fits.PrimaryHDU(), hdunew, tbhdu, hdumask])
                hdulnew.writeto(outfitsfile, output_verify='fix')
                return outfitsfile

        hdulnew = fits.HDUList([fits.PrimaryHDU(data=data, header=header), tbhdu])
        hdulnew.writeto(outfitsfile)
        print('wrapped fits written as ' + outfitsfile)
        return outfitsfile


def update(fitsfile, new_data=None, new_columns=None, new_header_entries=None):
    """
    Updates a FITS file by optionally replacing its primary or compressed image data, adding new columns to the
    first binary table (BinTableHDU), and/or updating header entries in the first image HDU (PrimaryHDU for
    uncompressed or CompImageHDU for compressed FITS files).

    Parameters:
    - fitsfile (str): Path to the FITS file to be updated.
    - new_data (np.ndarray, optional): New data array to replace the existing data in the first image HDU.
      Defaults to None, which means the data will not be updated.
    - new_columns (list of astropy.io.fits.Column, optional): New columns to be added to the first BinTableHDU.
      Defaults to None, which means no columns will be added.
    - new_header_entries (dict, optional): Header entries to update or add in the first image HDU. Each key-value
      pair represents a header keyword and its new value. Defaults to None, which means no header updates will be made.

    Returns:
    - bool: True if any of the specified updates were successfully applied, False otherwise.

    The function determines whether the FITS file is compressed to properly handle the image HDU type. It attempts
    to update the image HDU's data, the BinTableHDU's columns, and the image HDU's header based on the provided
    arguments. If all input parameters are None, indicating no updates are specified, the function will print a message
    and return False.
    """
    if new_data is None and new_columns is None and new_header_entries is None:
        print("No updates to perform: all new_data new_columns and new_header_entries are None.")
        return False

    if is_compressed_fits(fitsfile):
        imagehdutype = fits.hdu.CompImageHDU
    else:
        imagehdutype = fits.hdu.PrimaryHDU

    try:
        with fits.open(fitsfile, mode='update') as hdul:
            if new_header_entries is None:
                header_updated = True
                ## skip updating header
            else:
                header_updated = False
                for hdu in hdul:
                    if isinstance(hdu, imagehdutype):
                        for key, value in new_header_entries.items():
                            if key == "HISTORY":
                                hdu.header.add_history(value)
                            else:
                                hdu.header[key] = value
                        header_updated = True
                        break  # Updating only the first relevant image HDU

            if new_data is None:
                data_updated = True
            else:
                data_updated = False
                for hdu in hdul:
                    if isinstance(hdu, imagehdutype):
                        hdu.data = new_data
                        data_updated = True
                        break
            if new_columns is None:
                bintable_updated = True
                ## skip updating bintable
            else:
                # Add new columns to the first BinTableHDU
                bintable_updated = False
                for idx, hdu in enumerate(hdul):
                    if isinstance(hdu, imagehdutype):
                        continue  # Skip compressed image HDUs
                    if isinstance(hdu, fits.hdu.BinTableHDU):
                        combined_columns = fits.ColDefs(hdu.columns) + fits.ColDefs(new_columns)
                        new_tbhdu = fits.BinTableHDU.from_columns(combined_columns)
                        hdul[idx] = new_tbhdu
                        bintable_updated = True
                        break  # Updating only the first BinTableHDU
            if not header_updated:
                print("Header update failed.")
            if not bintable_updated:
                print("BinTable update failed.")
            if not data_updated:
                print("Data update failed.")
            if not header_updated and not bintable_updated and not data_updated:
                return False
            return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
