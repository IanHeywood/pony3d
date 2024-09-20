import os
import re
import glob
import numpy as np
from astropy.io import fits

def create_directories(opdir, masktag, noisetag, averagetag, filtertag, savenoise, saveaverage, minchans):
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

def get_image(fits_file):
    """
    Extract the 2D image data from a FITS file.

    Args:
    fits_file (str): Path to the FITS file.

    Returns:
    np.array: 2D image data.
    """
    input_hdu = fits.open(fits_file, ignore_missing_simple=True)[0]
    if len(input_hdu.data.shape) == 2:
        return np.array(input_hdu.data[:, :])
    elif len(input_hdu.data.shape) == 3:
        return np.array(input_hdu.data[0, :, :])
    else:
        return np.array(input_hdu.data[0, 0, :, :])

def flush_image(image_data, fits_file):
    """
    Write the 2D image data array to a FITS file.

    Args:
    image_data (np.array): 2D array of image data.
    fits_file (str): Path to the FITS file to write to.
    """
    with fits.open(fits_file, mode='update') as f:
        input_hdu = f[0]
        if len(input_hdu.data.shape) == 2:
            input_hdu.data[:, :] = image_data
        elif len(input_hdu.data.shape) == 3:
            input_hdu.data[0, :, :] = image_data
        elif len(input_hdu.data.shape) == 4:
            input_hdu.data[0, 0, :, :] = image_data
        f.flush()

def load_cube(fits_list):
    """
    Load a sequence of FITS images into a 3D numpy array.

    Args:
    fits_list (list): List of FITS file paths.

    Returns:
    np.array: 3D cube of image data.
    """
    temp = [get_image(fits_file) for fits_file in fits_list]
    return np.dstack(temp)

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

