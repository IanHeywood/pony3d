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


def load_cube(input_fits):
    """Loads a 3D FITS cube."""
    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        return np.squeeze(data)


def load_images(image_pattern):
    """Loads a sequence of 2D FITS images into a 3D cube."""
    image_files = sorted(glob.glob(image_pattern))  # Sort to ensure correct order
    if not image_files:
        raise FileNotFoundError(f"No FITS files found matching pattern: {image_pattern}")
    
    # Read first image to get dimensions
    with fits.open(image_files[0]) as hdul:
        shape = hdul[0].data.shape
    
    # Create 3D cube
    cube = np.zeros((len(image_files), *shape), dtype=np.float32)

    for i, file in enumerate(image_files):
        with fits.open(file) as hdul:
            cube[i, :, :] = hdul[0].data

    return cube, image_files  # Return file list to use as template for writing


def load_image_chunk(image_files, chunk_idx, chunk_size, total_chunks):
    """
    Loads a chunk of a sequence of 2D FITS images into a 3D cube for processing.

    - image_files: List of input image file paths.
    - chunk_idx: Index of the current chunk.
    - chunk_size: Number of divisions along RA/Dec.
    - total_chunks: Total number of chunks.

    Returns:
    - cube: 3D NumPy array of the selected chunk.
    - template_files: List of image filenames for writing.
    - chunk_slices: The spatial slice indices for updating the output files.
    """
    image_files = sorted(image_files)  # Ensure images are in order

    # Read first image to get dimensions
    with fits.open(image_files[0]) as hdul:
        full_shape = hdul[0].data.shape  # (RA, Dec)

    ra_chunks = np.array_split(range(full_shape[0]), chunk_size)
    dec_chunks = np.array_split(range(full_shape[1]), chunk_size)

    # Determine which RA/Dec slices correspond to this chunk
    ra_idx = chunk_idx % chunk_size
    dec_idx = chunk_idx // chunk_size

    ra_slice = slice(ra_chunks[ra_idx][0], ra_chunks[ra_idx][-1] + 1)
    dec_slice = slice(dec_chunks[dec_idx][0], dec_chunks[dec_idx][-1] + 1)

    # Load only the relevant chunk into memory
    cube_chunk = np.zeros((len(image_files), ra_slice.stop - ra_slice.start, dec_slice.stop - dec_slice.start), dtype=np.float32)

    for i, file in enumerate(image_files):
        with fits.open(file) as hdul:
            cube_chunk[i, :, :] = hdul[0].data[ra_slice, dec_slice]

    return cube_chunk, image_files, (ra_slice, dec_slice)


def write_cube(data_array, output_fits, input_fits, overwrite):
    """Writes a 3D FITS cube."""
    with fits.open(input_fits) as hdul:
        header = hdul[0].header
    hdu = fits.PrimaryHDU(data_array, header=header)
    hdu.writeto(output_fits, overwrite=overwrite)


def write_images(data_array, template_files, overwrite):
    """Writes out a sequence of 2D FITS images, replacing .fits with .contsub.fits if no output prefix is provided."""
    for i, filename in enumerate(template_files):
        output_filename = filename.replace(".fits", ".contsub.fits")
        with fits.open(filename) as hdul:
            header = hdul[0].header
        hdu = fits.PrimaryHDU(data_array[i, :, :], header=header)
        hdu.writeto(output_filename, overwrite=overwrite)


def write_chunked_images(data_array, template_files, chunk_idx, total_chunks, overwrite):
    """
    Writes out a sequence of 2D FITS images in chunks, replacing .fits with .contsub.fits.

    - If `chunk_idx == 0`, creates new output files.
    - Otherwise, updates existing files with new chunk data.
    """
    for i, filename in enumerate(template_files):
        output_filename = filename.replace(".fits", ".contsub.fits")

        with fits.open(filename) as hdul:
            header = hdul[0].header

        # Load or create output file
        if chunk_idx == 0 or not os.path.exists(output_filename):
            output_data = np.zeros_like(data_array[i])  # Create new empty output file
        else:
            with fits.open(output_filename, mode='update') as hdul_out:
                output_data = hdul_out[0].data  # Load existing output data

        # Update chunk in the output file
        output_data[...] = data_array[i]  # Overwrite only the processed chunk

        # Write back to file
        hdu = fits.PrimaryHDU(output_data, header=header)
        hdu.writeto(output_filename, overwrite=overwrite)


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

    # Ensure window sizes are valid (must be odd and smaller than the spectrum length)
    savgol_filter_size = savgol_filter_size if savgol_filter_size % 2 else savgol_filter_size + 1  # Ensure odd length
    median_filter_size = min(median_filter_size, len(y) - (1 - len(y) % 2))

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
    x = np.arange(len(y))
    y_masked = np.interp(x, x[~mask], y[~mask])  # Linear interpolation over masked values

    # Step 5: Apply Savitzky-Golay filter to smooth the baseline
    smoothed_baseline = savgol_filter(y_masked, window_length=savgol_filter_size, polyorder=poly_order)

    # Step 6: Subtract the baseline to obtain the continuum-subtracted spectrum
    return y - smoothed_baseline


def worker(args, mode, sigma_clip_threshold, median_filter_size, savgol_filter_size, polyorder):
    """Worker function for parallel spectral processing."""
    index_pair, spec = args
    i, j = index_pair
    x = np.arange(len(spec))

    if mode == "poly":
        _, S_contsub = fit_baseline(x, spec, poly_order=polyorder, sigma_clip_threshold=sigma_clip_threshold)
    elif mode == "savgol":
        S_contsub = apply_savgol_filter(spec, sigma_clip_threshold=sigma_clip_threshold, 
                                        median_filter_size=median_filter_size, 
                                        savgol_filter_size=savgol_filter_size, 
                                        poly_order=polyorder)
    else:
        raise ValueError(f"Unknown mode '{mode}', choose 'poly' or 'savgol'.")

    return (i, j, S_contsub)


def parallel_process_spectra(cube, mode, sigma_clip_threshold, median_filter_size, savgol_filter_size, polyorder, ncores):
    """Parallel spectral processing for a 3D FITS cube."""
    nx, ny = cube.shape[1], cube.shape[2]
    tasks = [((i, j), cube[:, i, j]) for i, j in itertools.product(range(nx), range(ny))]

    with Pool(processes=ncores) as pool:
        worker_with_args = partial(worker, mode=mode, sigma_clip_threshold=sigma_clip_threshold, 
                                   savgol_filter_size=savgol_filter_size, polyorder=polyorder)
        results = pool.map(worker_with_args, tasks)
    
    processed_cube = np.zeros_like(cube)
    for i, j, spectrum in results:
        processed_cube[:, i, j] = spectrum
    return processed_cube


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
    parser.add_argument("--polyorder", type=int, default=1, help="Polynomial order")
    parser.add_argument("--out", type=str, help="Output filename (for cubes) or image prefix (default: replaces .fits with .contsub.fits)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--ncores", type=int, default=16, help="Number of CPU cores for processing")
    parser.add_argument("--chunk", type=int, default=1, help="Number of spatial chunks for images (default: 1, no chunking)")

    args = parser.parse_args()

    # **Case 1: Processing a 3D Cube**
    if args.cube:
        logger.info(f'Loading cube: {args.cube}')
        data_array = load_cube(args.cube)

        logger.info(f'Processing cube with {args.ncores} cores.')
        t0 = time.time()
        contsub_array = parallel_process_spectra(data_array, args.mode, args.sigma, args.savgol, args.polyorder, args.ncores)
        elapsed = time.time() - t0
        logger.info(f'Cube processing completed in {round(elapsed / 60, 2)} minutes')

        # Determine output filename
        output_fits = args.out or args.cube.replace('.fits', '.contsub.fits')
        logger.info(f'Writing {output_fits}')
        write_cube(contsub_array, output_fits, args.cube, args.overwrite)

    # **Case 2: Processing a 2D Image Sequence with Chunking**
    elif args.images:
        image_files = sorted(glob.glob(args.images))
        if not image_files:
            logger.error("No input images found matching the pattern.")
            sys.exit(1)

        logger.info(f'Processing {len(image_files)} images with {args.chunk}x{args.chunk} spatial chunks.')

        total_chunks = args.chunk ** 2
        for chunk_idx in range(total_chunks):
            logger.info(f'Processing chunk {chunk_idx + 1}/{total_chunks}')
            
            # Load chunk into memory
            data_array, template_files, chunk_slices = load_image_chunk(image_files, chunk_idx, args.chunk, total_chunks)

            # Process the chunk
            contsub_array = parallel_process_spectra(data_array, args.mode, args.sigma, args.savgol, args.polyorder, args.ncores)

            # Write chunk back to images
            write_chunked_images(contsub_array, template_files, chunk_idx, total_chunks, args.overwrite)

        logger.info("Image processing complete.")

    else:
        logger.error("You must specify either --cube (3D FITS cube) or --images (2D FITS sequence).")
        sys.exit(1)

    logger.info("Done.")



if __name__ == '__main__':
    main()
