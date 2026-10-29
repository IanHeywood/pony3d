#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk


import os
import re
import glob
import numpy as np
import logging
from astropy.io import fits


def create_directories(opdir, masktag, noisetag, averagetag, filtertag, cubetag, savenoise, saveaverage, catalogue, subcubes):
    """
    Create necessary directories for output files.

    Args:
    opdir (str): Output directory.
    masktag (str): Tag for mask files.
    noisetag (str): Tag for noise files.
    averagetag (str): Tag for averaged files.
    filtertag (str): Tag for filtered files.
    savenoise (bool): Whether to save noise files.
    saveaverage (bool): Whether to save averaged files.
    nofilter (bool): Whether to skip filtering.
    """
    if not os.path.isdir(opdir):
        os.mkdir(opdir)
    if not os.path.isdir(f'{opdir}/{masktag}'):
        os.mkdir(f'{opdir}/{masktag}')
    if savenoise and not os.path.isdir(f'{opdir}/{noisetag}'):
        os.mkdir(f'{opdir}/{noisetag}')
    if saveaverage and not os.path.isdir(f'{opdir}/{averagetag}'):
        os.mkdir(f'{opdir}/{averagetag}')
    if not os.path.isdir(f'{opdir}/{filtertag}'):
        os.mkdir(f'{opdir}/{filtertag}')
    if catalogue and not os.path.isdir(f'{opdir}/cat_temp'):
        os.mkdir(f'{opdir}/cat_temp')
    if subcubes and not os.path.isdir(f'{opdir}/{cubetag}'):
        os.mkdir(f'{opdir}/{cubetag}')


def get_image(fits_file):
    """
    Extract the 2D image data from a FITS file.

    Args:
    fits_file (str): Path to the FITS file.

    Returns:
    np.array: 2D image data
    image header
    """
    logging.info(f'Reading {fits_file}')
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data
        if len(image_data.shape) == 2: image_data = image_data[:,:]
        elif len(image_data.shape) == 3: image_data = image_data[0,:,:]
        else: image_data = image_data[0,0,:,:]
        header = hdul[0].header
    logging.info(f'Done with {fits_file}')
    return image_data, header


def get_header(fits_file):
    """Read only the primary header of a FITS file."""
    return fits.getheader(fits_file)


def channel_frequency(header):
    """
    Frequency (Hz) of the single channel in a per-channel image header, allowing for CRPIX3 != 1.
    Returns None if the header has no third axis.
    """
    if 'CRVAL3' not in header:
        return None
    return header['CRVAL3'] + (1 - header.get('CRPIX3', 1)) * header.get('CDELT3', 0.0)


def check_channel_spacing(freqs, rtol=1e-3):
    """
    Check that a sequence of channel frequencies is evenly spaced.

    Returns (ok, spacing, gaps) where spacing is the median channel separation and gaps
    lists the indices i where freqs[i+1] - freqs[i] differs from it by more than rtol.
    """
    freqs = np.asarray(freqs, dtype=float)
    if len(freqs) < 2:
        return True, None, []
    diffs = np.diff(freqs)
    spacing = np.median(diffs)
    gaps = list(np.where(np.abs(diffs - spacing) > rtol * abs(spacing))[0])
    return len(gaps) == 0, spacing, gaps


def flush_image(image_data, header, fits_file, dtype='float32'):
    """
    Write the 2D image data array to a FITS file.

    Args:
    fits_file (str): Path to the FITS file to write to.
    image_data (np.array): 2D array of image data.
    header: FITS file header (generally copied from the corresponding input image)
    """
    logging.info(f'Writing {fits_file} with data length {len(image_data)}; duplicate file? {os.path.exists(fits_file)}')
    try:
        hdu = fits.PrimaryHDU(image_data.astype(dtype), header)
        hdu.verify('silentfix')
        hdu.writeto(fits_file, overwrite=True)
    except Exception:
        logging.exception(f'Error: FITS file NOT written {fits_file}')
        return False
    if os.path.exists(fits_file):
        logging.info(f'Written {fits_file}')
        return True
    logging.error(f'Error: FITS file NOT written {fits_file}')
    return False


def load_cube(fits_list, dtype=None):
    """
    Load a sequence of FITS images into a 3D numpy array.

    The cube is allocated once and filled plane by plane, so peak memory is
    one cube plus one image (rather than a list of images plus a stacked copy).

    Args:
    fits_list (list): List of FITS file paths.
    dtype: Data type of the returned cube. Default is the dtype of the first image.
           Use bool for masks, where only non-zero values matter.

    Returns:
    np.array: 3D cube of image data, shape (y, x, channel).
    """
    logging.info(f'Forming cube from {fits_list[0]} -- {fits_list[-1]} ({len(fits_list)} images)')
    first_image = get_image(fits_list[0])[0]
    if dtype is None:
        dtype = first_image.dtype.newbyteorder('=')
    cube = np.empty(first_image.shape + (len(fits_list),), dtype=dtype)
    for i, fits_file in enumerate(fits_list):
        image = first_image if i == 0 else get_image(fits_file)[0]
        if image.shape != first_image.shape:
            raise ValueError(f'{fits_file} has shape {image.shape}, expected {first_image.shape}')
        cube[:, :, i] = ((image != 0) & ~np.isnan(image)) if dtype == bool else image
    logging.info(f'Cube formed from {fits_list[0]} -- {fits_list[-1]}')
    return cube


# def load_cube_with_concat(fits_list):
#     """
#     Load a sequence of FITS images with RA and Dec axes and a Freq axis of length 1 into a 3D numpy array.

#     Args:
#     fits_list (list): List of FITS file paths.

#     Returns:
#     np.ndarray: 3D cube of image data (Freq, Dec, RA).
#     """
#     # List to hold each 3D array (1, Dec, RA) from each FITS file
#     image_list = []

#     # Load each FITS file and append the data to the list
#     for fits_file in fits_list:
#         # Load the FITS data; assuming data shape is (Dec, RA) or (1, Dec, RA)
#         data = fits.getdata(fits_file)

#         # If data is 2D (Dec, RA), expand to 3D (1, Dec, RA)
#         if len(data.shape) == 2:
#             data = data[np.newaxis, :, :]  # Add a new axis for the frequency dimension

#         # Ensure data is now 3D with shape (1, Dec, RA)
#         if len(data.shape) != 3 or data.shape[0] != 1:
#             raise ValueError(f"Unexpected data shape {data.shape} in file {fits_file}. Expected (1, Dec, RA).")

#         # Append the data to the list
#         image_list.append(data)

#     # Concatenate all 3D arrays along the frequency axis (axis=0)
#     cube = np.concatenate(image_list, axis=0)

#     return cube


def natural_sort(l):
    """
    Sort the given iterable in the way that humans expect.

    Args:
    l (list): List of strings to sort.

    Returns:
    list: Naturally sorted list.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
