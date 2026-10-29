#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk

import argparse
import itertools
import logging
import numpy as np
import os
import sys
import time
import warnings
import glob
import re

from astropy.io import fits
from astropy.stats import sigma_clip
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from scipy.signal import savgol_filter
from scipy.ndimage import binary_dilation, median_filter


def initialize_logging():
    date_time = datetime.now()
    timestamp = date_time.strftime('%d%m%Y_%H%M%S')
    logfile = f'imcontsub_{timestamp}.log'
    logging.basicConfig(
        filename=logfile, level=logging.DEBUG,
        format='%(asctime)s:: %(levelname)-5s :: %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S ', force=True
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    return logger


def natural_sort(l):
    """Sort strings the way humans expect (channel 10 after channel 9)."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    return sorted(l, key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])


def load_cube(input_fits):
    """
    Loads a FITS cube as a 3D (chan, y, x) float32 array.
    Degenerate axes (e.g. a length-1 Stokes axis) are dropped; the original
    shape is returned so the output can be written in the same layout.
    """
    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        shape = data.shape
        cube = np.squeeze(data).astype(np.float32)
    if cube.ndim != 3:
        raise ValueError(f'{input_fits} does not contain a 3D cube (shape {shape})')
    return cube, shape


def image_shape(fits_file):
    """Shape of the 2D image in fits_file, ignoring degenerate axes."""
    with fits.open(fits_file, memmap=True) as hdul:
        return np.squeeze(hdul[0].data).shape


def load_image_chunk(image_files, chunk_idx, chunk_size):
    """
    Loads one spatial chunk of a sequence of 2D FITS images into a (chan, y, x) cube.

    Spectra are processed independently, so chunks do not need to overlap.

    Parameters:
    - image_files: List of input image file paths, in frequency order.
    - chunk_idx: Index of the current chunk.
    - chunk_size: Number of divisions along each spatial axis.

    Returns:
    - cube_chunk: 3D NumPy array of the selected chunk.
    - chunk_slices: (y_slice, x_slice) of the chunk within the full image.
    """
    ny, nx = image_shape(image_files[0])
    y_chunks = np.array_split(np.arange(ny), chunk_size)
    x_chunks = np.array_split(np.arange(nx), chunk_size)
    y_idx = chunk_idx // chunk_size
    x_idx = chunk_idx % chunk_size
    y_slice = slice(y_chunks[y_idx][0], y_chunks[y_idx][-1] + 1)
    x_slice = slice(x_chunks[x_idx][0], x_chunks[x_idx][-1] + 1)

    cube_chunk = np.zeros((len(image_files), y_slice.stop - y_slice.start, x_slice.stop - x_slice.start), dtype=np.float32)
    for i, file in enumerate(image_files):
        with fits.open(file, memmap=True) as hdul:
            image = np.squeeze(hdul[0].data)
            if image.shape != (ny, nx):
                raise ValueError(f'{file} has shape {image.shape}, expected {(ny, nx)}')
            cube_chunk[i, :, :] = image[y_slice, x_slice]

    return cube_chunk, (y_slice, x_slice)


def write_cube(data_array, shape, output_fits, input_fits, overwrite):
    """Writes a 3D FITS cube with the header (and axis layout) of the input cube."""
    header = fits.getheader(input_fits)
    hdu = fits.PrimaryHDU(data_array.reshape(shape), header=header)
    hdu.writeto(output_fits, overwrite=overwrite)


def output_name(filename, outdir):
    """Output name for an input image: .fits -> .contsub.fits, optionally in outdir."""
    base = os.path.basename(filename)
    base = base[:-5] + '.contsub.fits' if base.endswith('.fits') else base + '.contsub.fits'
    return os.path.join(outdir if outdir else os.path.dirname(filename), base)


def create_output_images(template_files, outdir, overwrite):
    """Creates NaN-filled output images with the same header and shape as each input."""
    for filename in template_files:
        output_filename = output_name(filename, outdir)
        if os.path.exists(output_filename) and not overwrite:
            raise FileExistsError(f'{output_filename} exists, use --overwrite')
        with fits.open(filename) as hdul:
            header = hdul[0].header
            data = np.full(hdul[0].data.shape, np.nan, dtype=np.float32)
        fits.PrimaryHDU(data, header=header).writeto(output_filename, overwrite=True)


def write_chunked_images(data_array, template_files, chunk_slices, outdir):
    """Writes one spatial chunk of the processed cube into the existing output images."""
    y_slice, x_slice = chunk_slices
    for i, filename in enumerate(template_files):
        with fits.open(output_name(filename, outdir), mode='update', memmap=True) as hdul:
            image = np.squeeze(hdul[0].data) # view onto the file data
            image[y_slice, x_slice] = data_array[i, :, :]
            hdul.flush()


def fit_baseline(x, y, poly_order=2, sigma_clip_threshold=3):
    """Fits and subtracts a polynomial baseline."""
    x = np.asarray(x)
    y = np.asarray(y)

    # Initial polynomial fit
    p = np.polyfit(x, y, poly_order)
    baseline = np.polyval(p, x)

    # Compute residuals & mask spectral features
    residuals = y - baseline
    std_dev = np.std(residuals)
    mask = np.abs(residuals) < sigma_clip_threshold * std_dev  # Boolean mask

    if np.sum(mask) == 0:
        raise ValueError("No valid data points remain after sigma clipping.")

    # Refit only to the "continuum" (non-line) regions
    p_final = np.polyfit(x[mask], y[mask], poly_order)
    baseline = np.polyval(p_final, x)
    
    return baseline, y - baseline


def ignore_edges(mask):
    """
    Ignore leading and trailing masked (True) values in a boolean mask.
    
    Parameters:
    - mask: 1D NumPy boolean array.

    Returns:
    - Updated mask with edges set to False.
    """
    # Find first and last False value (valid data)
    idx_start = next((i for i, j in enumerate(mask) if not j), None)
    idx_end = next((i for i, j in enumerate(mask[::-1]) if not j), None)
    
    if idx_start is not None:
        mask[:idx_start] = False  # Set all leading True values to False
    if idx_end is not None:
        idx_end = len(mask) - idx_end
        mask[idx_end:] = False  # Set all trailing True values to False

    return mask


def apply_savgol_filter(y, sigma_clip_threshold=2.3, median_filter_size=31, savgol_filter_size=31, poly_order=1):
    """
    Applies Savitzky-Golay filtering for continuum subtraction, incorporating masking and median filtering.

    Parameters:
    - y: 1D NumPy array (spectrum).
    - sigma_clip_threshold: Sigma clipping threshold for masking spectral features.
    - median_filter_size: Window size for median filtering.
    - savgol_filter_size: Window length for Savitzky-Golay filtering (must be odd).
    - poly_order: Polynomial order for Savitzky-Golay filter.

    Returns:
    - y_corrected: Spectrum with baseline removed.
    """

    # Ensure window sizes are valid (odd, no longer than the spectrum, longer than poly_order)
    largest_odd = len(y) if len(y) % 2 else len(y) - 1
    savgol_filter_size = savgol_filter_size if savgol_filter_size % 2 else savgol_filter_size + 1
    savgol_filter_size = min(savgol_filter_size, largest_odd)
    if savgol_filter_size <= poly_order:
        return y - np.median(y)
    median_filter_size = min(median_filter_size, largest_odd)

    # Step 1: Apply sigma clipping to identify spectral features
    y_median = median_filter(y, size=median_filter_size)  # Median filter first
    residuals = y - y_median
    std_dev = np.std(residuals)
    mask = np.abs(residuals) > sigma_clip_threshold * std_dev  # Mask strong deviations

    # Step 2: Apply binary dilation to expand the masked regions
    mask = binary_dilation(mask, iterations=2)  # Expand mask slightly

    # Step 3: Ignore edges (mask leading/trailing `True` values)
    mask = ignore_edges(mask)

    # Step 4: Interpolate over masked regions
    if mask.all():
        return y - np.median(y)
    x = np.arange(len(y))
    y_masked = np.interp(x, x[~mask], y[~mask])  # Linear interpolation over masked values

    # Step 5: Apply Savitzky-Golay filter to smooth the baseline
    smoothed_baseline = savgol_filter(y_masked, window_length=savgol_filter_size, polyorder=poly_order)

    # Step 6: Subtract the baseline to obtain the continuum-subtracted spectrum
    return y - smoothed_baseline


def process_spectrum(spec, mode, sigma_clip_threshold, median_filter_size, savgol_filter_size, polyorder):
    """
    Continuum-subtract one spectrum. NaN channels are interpolated over for the fit
    and are NaN in the output; spectra with too few finite channels are returned as NaN.
    """
    finite = np.isfinite(spec)
    if finite.sum() < polyorder + 2:
        return np.full_like(spec, np.nan)
    x = np.arange(len(spec))
    y = spec if finite.all() else np.interp(x, x[finite], spec[finite])

    if mode == "poly":
        _, S_contsub = fit_baseline(x, y, poly_order=polyorder, sigma_clip_threshold=sigma_clip_threshold)
    elif mode == "savgol":
        S_contsub = apply_savgol_filter(y, sigma_clip_threshold=sigma_clip_threshold,
                                        median_filter_size=median_filter_size,
                                        savgol_filter_size=savgol_filter_size,
                                        poly_order=polyorder)
    else:
        raise ValueError(f"Unknown mode '{mode}', choose 'poly' or 'savgol'.")

    S_contsub[~finite] = np.nan
    return S_contsub


def worker(block, mode, sigma_clip_threshold, median_filter_size, savgol_filter_size, polyorder):
    """Worker function: continuum-subtract every spectrum in a (chan, rows, x) block."""
    out = np.empty_like(block)
    for i in range(block.shape[1]):
        for j in range(block.shape[2]):
            out[:, i, j] = process_spectrum(block[:, i, j], mode, sigma_clip_threshold,
                                            median_filter_size, savgol_filter_size, polyorder)
    return out


def parallel_process_spectra(cube, mode, sigma_clip_threshold, median_filter_size, savgol_filter_size, polyorder, ncores):
    """
    Parallel spectral processing for a (chan, y, x) cube.
    The cube is split into blocks of rows, so each task carries many spectra
    (one task per spectrum made pickling overheads dominate).
    """
    ny = cube.shape[1]
    nblocks = min(ny, max(1, ncores * 4))
    row_blocks = [r for r in np.array_split(np.arange(ny), nblocks) if len(r) > 0]
    blocks = [cube[:, r[0]:r[-1] + 1, :] for r in row_blocks]

    worker_with_args = partial(worker, mode=mode, sigma_clip_threshold=sigma_clip_threshold,
                               median_filter_size=median_filter_size,
                               savgol_filter_size=savgol_filter_size, polyorder=polyorder)
    with Pool(processes=ncores) as pool:
        results = pool.map(worker_with_args, blocks)

    return np.concatenate(results, axis=1)


#------------------------------------------------


def main():
    logger = initialize_logging()
    warnings.filterwarnings('ignore', category=UserWarning, append=True)

    parser = argparse.ArgumentParser(description="Continuum subtraction for FITS cubes or large image sequences with chunking.")
    parser.add_argument("--cube", type=str, help="Input FITS cube file")
    parser.add_argument("--images", type=str, help="Pattern to load a sequence of 2D FITS images (e.g., '*.fits')")
    parser.add_argument("--mode", type=str, choices=['savgol', 'poly'], default="savgol", help="Continuum subtraction mode")
    parser.add_argument("--sigma", type=float, default=2.3, help="Sigma clipping threshold")
    parser.add_argument("--savgol", type=int, default=31, help="Savitzky-Golay filter window size")
    parser.add_argument("--median", type=int, default=31, help="Median filter window size used to find spectral features (savgol mode)")
    parser.add_argument("--polyorder", type=int, default=1, help="Polynomial order")
    parser.add_argument("--out", type=str, help="Output filename (for cubes) or output directory (for images); default replaces .fits with .contsub.fits alongside the input")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--ncores", type=int, default=16, help="Number of CPU cores for processing")
    parser.add_argument("--chunk", type=int, default=1, help="Number of spatial chunks for images (default: 1, no chunking)")

    args = parser.parse_args()

    # **Case 1: Processing a 3D Cube**
    if args.cube:
        logger.info(f'Loading cube: {args.cube}')
        data_array, shape = load_cube(args.cube)

        logger.info(f'Processing cube with {args.ncores} cores.')
        t0 = time.time()
        contsub_array = parallel_process_spectra(data_array, args.mode, args.sigma, args.median, args.savgol, args.polyorder, args.ncores)
        elapsed = time.time() - t0
        logger.info(f'Cube processing completed in {round(elapsed / 60, 2)} minutes')

        # Determine output filename
        output_fits = args.out or args.cube.replace('.fits', '.contsub.fits')
        logger.info(f'Writing {output_fits}')
        write_cube(contsub_array, shape, output_fits, args.cube, args.overwrite)

    # **Case 2: Processing a 2D Image Sequence with Chunking**
    elif args.images:
        image_files = natural_sort(glob.glob(args.images))
        if not image_files:
            logger.error("No input images found matching the pattern.")
            sys.exit(1)
        if args.out and not os.path.isdir(args.out):
            os.makedirs(args.out)

        logger.info(f'Processing {len(image_files)} images with {args.chunk}x{args.chunk} spatial chunks.')
        create_output_images(image_files, args.out, args.overwrite)

        total_chunks = args.chunk ** 2
        for chunk_idx in range(total_chunks):
            logger.info(f'Processing chunk {chunk_idx + 1}/{total_chunks}')
            data_array, chunk_slices = load_image_chunk(image_files, chunk_idx, args.chunk)
            contsub_array = parallel_process_spectra(data_array, args.mode, args.sigma, args.median, args.savgol, args.polyorder, args.ncores)
            write_chunked_images(contsub_array, image_files, chunk_slices, args.out)

        logger.info("Image processing complete.")

    else:
        logger.error("You must specify either --cube (3D FITS cube) or --images (2D FITS sequence).")
        sys.exit(1)

    logger.info("Done.")



if __name__ == '__main__':
    main()
