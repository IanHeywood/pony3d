import logging
import numpy as np
import os
import shutil
from scipy.ndimage import binary_dilation, minimum_filter
import scipy.special
from pony3d.file_operations import get_image, flush_image

def get_mask_and_noise(input_image, threshold, boxsize, dilate):
    """
    Generate a mask and noise image from the input image.

    Args:
    input_image (np.array): 2D input image array.
    threshold (float): Sigma threshold for masking.
    boxsize (int): Box size for background noise estimation.
    dilate (int): Number of dilation iterations.

    Returns:
    tuple: Boolean mask array and noise estimation array.
    """
    box = (boxsize, boxsize)
    n = boxsize ** 2.0
    x = np.linspace(-10, 10, 1000)
    f = 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))
    F = 1.0 - (1.0 - f) ** n
    ratio = np.abs(np.interp(0.5, F, x))
    noise_image = -minimum_filter(input_image, box) / ratio
    noise_image[noise_image < 0.0] = 1.0e-10
    median_noise = np.median(noise_image)
    noise_image[noise_image < median_noise] = median_noise
    mask_image = input_image > threshold * noise_image
    if dilate > 0:
        mask_image = binary_dilation(mask_image, iterations=dilate)
    return mask_image, noise_image

def make_mask(input_fits, threshold, boxsize, dilate, invert, opdir, masktag, noisetag, savenoise, overwrite, idx):
    """
    Create a mask for a single FITS image.

    Args:
    input_fits (str): Path to the input FITS file.
    threshold (float): Sigma threshold for masking.
    boxsize (int): Box size for background noise estimation.
    dilate (int): Number of dilation iterations.
    invert (bool): Whether to invert the image.
    opdir (str): Output directory.
    masktag (str): Tag for mask files.
    noisetag (str): Tag for noise files.
    savenoise (bool): Whether to save noise files.
    overwrite (bool): Whether to overwrite existing files.
    idx (int): Index for logging.
    """
    idx = str(idx).zfill(5)
    mask_fits = os.path.join(opdir, masktag, input_fits.replace('.fits', f'.{masktag}.fits'))
    if os.path.isfile(mask_fits) and not overwrite:
        logging.info(f'[M{idx}] Skipping {input_fits} (overwrite disabled)')
        return

    logging.info(f'[M{idx}] Reading {input_fits}')
    input_image = get_image(input_fits)
    if invert:
        logging.info(f'[M{idx}] Inverting {input_fits}')
        input_image *= -1.0
    logging.info(f'[M{idx}] Finding islands')
    mask_image, noise_image = get_mask_and_noise(input_image, threshold, boxsize, dilate)
    logging.info(f'[M{idx}] Writing {mask_fits}')
    shutil.copyfile(input_fits, mask_fits)
    flush_image(mask_image, mask_fits)
    if savenoise:
        noise_fits = os.path.join(opdir, noisetag, input_fits.replace('.fits', f'.{noisetag}.fits'))
        shutil.copyfile(input_fits, noise_fits)
        logging.info(f'[M{idx}] Writing {noise_fits}')
        flush_image(noise_image, noise_fits)

def make_averaged_mask(input_fits_subset, threshold, boxsize, dilate, invert, opdir, masktag, noisetag, savenoise, averagetag, saveaverage, overwrite, idx):
    """
    Create a mask for a sequence of averaged FITS images.

    Args:
    input_fits_subset (list): List of input FITS files.
    threshold (float): Sigma threshold for masking.
    boxsize (int): Box size for background noise estimation.
    dilate (int): Number of dilation iterations.
    invert (bool): Whether to invert the image.
    opdir (str): Output directory.
    masktag (str): Tag for mask files.
    noisetag (str): Tag for noise files.
    savenoise (bool): Whether to save noise files.
    averagetag (str): Tag for averaged files.
    saveaverage (bool): Whether to save averaged files.
    overwrite (bool): Whether to overwrite existing files.
    idx (int): Index for logging.
    """
    idx = str(idx).zfill(5)
    nfits = len(input_fits_subset)
    input_fits = input_fits_subset[nfits // 2]
    mask_fits = os.path.join(opdir, masktag, input_fits.replace('.fits', f'.{masktag}.fits'))
    if os.path.isfile(mask_fits) and not overwrite:
        logging.info(f'[A{idx}] Skipping {mask_fits} (overwrite disabled)')
        return

    logging.info(f'[A{idx}] Reading subset')
    cube = load_cube(input_fits_subset)
    mean_image = np.nanmean(cube, axis=2)
    if invert:
        logging.info(f'[M{idx}] Inverting subset')
        mean_image *= -1.0
    logging.info(f'[A{idx}] Finding islands')
    mask_image, noise_image = get_mask_and_noise(mean_image, threshold, boxsize, dilate)
    logging.info(f'[A{idx}] Writing {mask_fits}')
    shutil.copyfile(input_fits, mask_fits)
    flush_image(mask_image, mask_fits)
    if saveaverage:
        mean_fits = os.path.join(opdir, averagetag, input_fits.replace('.fits', f'.{averagetag}.fits'))
        logging.info(f'[A{idx}] Writing {mean_fits}')
        shutil.copyfile(input_fits, mean_fits)
        flush_image(mean_image, mean_fits)
    if savenoise:
        noise_fits = os.path.join(opdir, noisetag, input_fits.replace('.fits', f'.{noisetag}.fits'))
        shutil.copyfile(input_fits, noise_fits)
        logging.info(f'[A{idx}] Writing {noise_fits}')
        flush_image(noise_image, noise_fits)

import numpy as np
import shutil
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from pony3d.file_operations import get_image, flush_image, load_cube

def filter_mask(mask_subset, specdilate, masktag, filtertag, overwrite, idx):
    """
    Filter mask images for single channel islands.

    Args:
    mask_subset (list): List of mask files.
    specdilate (int): Number of iterations of binary dilation in the spectral dimension.
    masktag (str): Tag for mask files.
    filtertag (str): Tag for filtered files.
    overwrite (bool): Whether to overwrite existing files.
    idx (int): Index for logging.
    """
    idx = str(idx).zfill(5)
    template_fits = []
    output_fits = []
    exists = []

    for input_fits in mask_subset[1:-1]:
        filtered_fits = input_fits.replace(masktag, filtertag)
        template_fits.append(input_fits)
        output_fits.append(filtered_fits)
        exists.append(os.path.isfile(filtered_fits))

    if all(exists) and not overwrite:
        logging.info(f'[F{idx}] Subset is complete, skipping (overwrite disabled)')
        return

    logging.info(f'[F{idx}] Reading subset')
    cube = load_cube(mask_subset) != 0
    recon_struct = np.ones((3, 3))

    logging.info(f'[F{idx}] Filtering image')
    for i in range(cube.shape[0]):
        cut = binary_fill_holes(cube[i, :, :])
        eroded = binary_erosion(cut)
        dilated = binary_dilation(eroded, structure=recon_struct, iterations=specdilate)
        cube[i, :, :] = dilated

    for i, filtered_fits in enumerate(output_fits):
        if exists[i] and not overwrite:
            logging.info(f'[F{idx}] Skipping {filtered_fits} (overwrite disabled)')
        else:
            logging.info(f'[F{idx}] Writing {filtered_fits}')
            shutil.copyfile(template_fits[i], filtered_fits)
            flush_image(cube[:, :, i + 1], filtered_fits)

def count_islands(input_fits, orig_fits, idx):
    """
    Count the number of islands in a mask to inform cleaning.

    Args:
    input_fits (str): Path to the input mask FITS file.
    orig_fits (str): Path to the original FITS file.
    idx (int): Index for logging.
    """
    idx = str(idx).zfill(5)
    input_image = get_image(input_fits)
    input_image = input_image.byteswap().newbyteorder()  # Fix for potential endianness issue
    labeled_mask_image, n_islands = label(input_image)
    orig_image = get_image(orig_fits)
    rms = np.std(orig_image)
    logging.info(f'[C{idx}] Clean parameters: {input_fits} {n_islands} {rms}')
