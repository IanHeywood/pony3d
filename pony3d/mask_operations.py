#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk


import gc
import glob
import logging
import numpy as np
import os
import scipy.special
import time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, binary_erosion, convolve, minimum_filter
from scipy.ndimage import find_objects, center_of_mass, label, binary_fill_holes
from pony3d.file_operations import get_image, get_header, flush_image, load_cube
from pony3d.file_operations import channel_frequency, check_channel_spacing
from pony3d import __version__


def get_tdl():
    tdl = '_-pony-_' # weird delimiter string to avoid split errors from the catalogue temp files
    return tdl


def tagged_name(fits_file, tag):
    """
    Output filename for fits_file with .tag inserted before the .fits suffix.
    Only the basename is used, so directory names never get altered.
    """
    base = os.path.basename(fits_file)
    if base.endswith('.fits'):
        return f'{base[:-5]}.{tag}.fits'
    return f'{base}.{tag}.fits'


def format_ra_dec(ra,dec,sep=':'):
    """
    Converts ra and dec in decimal degrees to hms/dms format

    Args:
    ra (float): right ascension in decimal degres
    dec (float): declination in decimal degrees
    sep (string): delimiter character for output

    Returns:
    tuple: RA and Dec strings in hms/dms format, and an ID for the catalogue
    """

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    ra_hms = str(coord.ra.to_string(unit=u.hour, sep=':', precision=2, pad= True))
    dec_dms = str(coord.dec.to_string(unit=u.deg, sep=':', precision=1, alwayssign=True, pad=True))
#    src_id = f'J{ra_hms.split(".")[0].replace(":","")}{dec_dms.split(".")[0].replace(":","")}'
    src_id = f'J{ra_hms.replace(":","")}{dec_dms.replace(":","")}'
    return ra_hms,dec_dms,src_id


def apply_region_mask(image_data, wcs, region_file):
    """
    Apply a DS9/CARTA region mask to a FITS image.
    
    Parameters:
    image_data (str): 2D numpy array containing the image data.
    WCS (???): WCS from header corresponding to the image data.
    region_file (str): Path to the DS9/CARTA region file.

    Returns:
    image_data (ndarray): Image data with the region applied as a mask (masked areas are set to np.nan).
    """

    from regions import Regions # only needed when --regionmask is used
    regs = Regions.read(region_file) #, format='ds9')
    mask = np.zeros_like(image_data) #, dtype='int')

    if wcs.naxis > 2:
        wcs = wcs.sub([1,2])

    for reg in regs:
        if hasattr(reg, 'to_pixel'):
            reg = reg.to_pixel(wcs)
            reg_image = reg.to_mask().to_image(mask.shape)
            if reg_image is not None: # None if the region lies entirely off the image
                mask += reg_image

    mask = np.array(mask,dtype='bool')
    image_data[mask] = np.nan
    return image_data


def get_mask_and_noise(input_image, threshold, boxsize, trim):
    """
    Generate a mask and noise image from the input image.
    Kentoc'h mervel eget bezan saotret

    Args:
    input_image (np.array): 2D input image array.
    threshold (float): Sigma threshold for masking.
    boxsize (int): Box size for background noise estimation.
    trim (int): Number of erosion iterations to deal with wacky mosaick edges.

    Returns:
    tuple: Boolean mask array and noise estimation array.
    """
    box = (boxsize, boxsize)
    n = boxsize ** 2.0
    x = np.linspace(-10, 10, 1000)
    f = 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))
    F = 1.0 - (1.0 - f) ** n
    ratio = np.abs(np.interp(0.5, F, x))
    # NaNs (blanked edges, region masks) are replaced with +inf so they never
    # become the box minimum, and are restored afterwards
    nan_mask = np.isnan(input_image)
    filled_image = np.where(nan_mask, np.inf, input_image)
    noise_image = -minimum_filter(filled_image, box) / ratio
    noise_image[~np.isfinite(noise_image)] = 1.0e-10 # boxes that are entirely NaN
    noise_image[noise_image < 0.0] = 1.0e-10
    noise_image[nan_mask] = np.nan
    median_noise = np.nanmedian(noise_image)
    noise_image[noise_image < median_noise] = median_noise
    mask_image = np.zeros(input_image.shape, dtype=bool)
    mask_image[~nan_mask] = input_image[~nan_mask] > threshold * noise_image[~nan_mask]

    # Erosion iterations trim method 
    if trim > 0:
        trim_mask = ~np.isnan(input_image)
        trim_mask = binary_erosion(trim_mask, iterations = trim)
        mask_image[~trim_mask] = 0.0
        noise_image[~trim_mask] = np.nan

    # Big structure erosion method (too slow)
    # if trim > 0:
    #     t0 = time.time()
    #     struct_element = disk(trim)
    #     trim_mask = ~np.isnan(input_image)
    #     trim_mask = binary_erosion(trim_mask, structure=struct_element)
    #     mask_image[~trim_mask] = 0.0
    #     noise_image[~trim_mask] = np.nan
    #     elapsed = time.time() - t0
    #     print(f'Trim operation took {round(elapsed,2)} seconds')

    # Convolution method (even slower)
    # if trim > 0:
    #     t0 = time.time()
    #     border_mask = ~np.isnan(input_image)
    #     kernel_size = 2 * trim + 1
    #     kernel = np.ones((kernel_size, kernel_size), dtype=int)
    #     convolved_mask = convolve(border_mask.astype(int), kernel)
    #     trim_mask = convolved_mask >= np.sum(kernel)
    #     mask_image[~trim_mask] = 0.0
    #     noise_image[~trim_mask] = np.nan
    #     elapsed = time.time() - t0
    #     print(f'Trim operation took {round(elapsed,2)} seconds')

    return mask_image, noise_image


def make_mask(input_fits, threshold, boxsize, dilate, trim, regionmask, invert, opdir, masktag, noisetag, savenoise, overwrite, idx):
    """
    Create a mask for a single FITS image.

    Args:
    input_fits (str): Path to the input FITS file.
    threshold (float): Sigma threshold for masking.
    boxsize (int): Box size for background noise estimation.
    dilate (int): Binary dilation iterations in the spatial dimensions
    trim (int): Binary erosion iterations to deal with noisy borders
    regionmask: Region file defining exclusion zones
    invert (bool): Whether to invert the image.
    opdir (str): Output directory.
    masktag (str): Tag for mask files.
    noisetag (str): Tag for noise files.
    savenoise (bool): Whether to save noise files.
    overwrite (bool): Whether to overwrite existing files.
    idx (int): Index for logging.
    """
    log_prefix = 'Mask'
    idx = str(idx).zfill(5)
    logging.info(f'[{log_prefix}_{idx}] Worker process {idx} operating on image {input_fits}')
    mask_fits = os.path.join(opdir, masktag, tagged_name(input_fits, masktag))
    if os.path.isfile(mask_fits) and not overwrite:
        logging.info(f'[{log_prefix}_{idx}] Skipping {input_fits} (overwrite disabled)')
        return

    logging.info(f'[{log_prefix}_{idx}] Reading {input_fits}')
    input_image,header = get_image(input_fits)
 
    if invert:
        logging.info(f'[{log_prefix}_{idx}] Inverting {input_fits}')
        input_image *= -1.0
    if regionmask != '':
        wcs = WCS(header)
        logging.info(f'[{log_prefix}_{idx}] Applying region file mask {regionmask}')
        input_image = apply_region_mask(input_image, wcs, regionmask)

    logging.info(f'[{log_prefix}_{idx}] Finding islands')
    mask_image, noise_image = get_mask_and_noise(input_image, threshold, boxsize, trim)
 
    if dilate > 0:
        logging.info(f'[{log_prefix}_{idx}] Dilating islands with {dilate} iterations along spatial axes')
        mask_image = binary_dilation(mask_image, iterations=dilate)
    
    logging.info(f'[{log_prefix}_{idx}] Writing mask image {mask_fits}')
    f_written = flush_image(mask_image, header, mask_fits)
    if not f_written:
        logging.error(f'Failed to write FITS file: [{log_prefix}_{idx}] Writing mask image {mask_fits}')

    if savenoise:
        noise_fits = os.path.join(opdir, noisetag, tagged_name(input_fits, noisetag))
        logging.info(f'[{log_prefix}_{idx}] Writing noise image {noise_fits}')
        f_written = flush_image(noise_image, header, noise_fits)
        if not f_written:
            logging.error(f'Failed to write FITS file: [{log_prefix}_{idx}] Writing noise image {noise_fits}')


def make_averaged_mask(input_fits_subset, threshold, boxsize, dilate, trim, regionmask, invert, opdir, masktag, noisetag, savenoise, averagetag, saveaverage, overwrite, idx):
    """
    Create a mask for a sequence of averaged FITS images.

    Args:
    input_fits_subset (list): List of input FITS files.
    threshold (float): Sigma threshold for masking.
    boxsize (int): Box size for background noise estimation.
    dilate (int): Number of dilation iterations.
    trim (int): Binary erosion iterations to deal with noisy borders
    regionmask: Region file defining exclusion zones
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
    log_prefix = 'BoxcarMask'
    idx = str(idx).zfill(5)
    logging.info(f'[{log_prefix}_{idx}] Worker process {idx} operating on image subset {input_fits_subset[0]} -- {input_fits_subset[-1]}')
    nfits = len(input_fits_subset)
    input_fits = input_fits_subset[nfits // 2]
    input_image, header = get_image(input_fits) # Need a WCS for image output and any region masking
    mask_fits = os.path.join(opdir, masktag, tagged_name(input_fits, masktag))
    if os.path.isfile(mask_fits) and not overwrite:
        logging.info(f'[{log_prefix}_{idx}] Skipping {mask_fits} (overwrite disabled)')
        return
    logging.info(f'[{log_prefix}_{idx}] Reading input image subset')
    cube = load_cube(input_fits_subset)
    mean_image = np.nanmean(cube, axis=2)
    if invert:
        logging.info(f'[{log_prefix}_{idx}] Inverting averaged image')
        mean_image *= -1.0
    if regionmask != '':
        logging.info(f'[{log_prefix}_{idx}] Applying region file mask {regionmask}')
        wcs = WCS(header)
        mean_image = apply_region_mask(mean_image, wcs, regionmask)
    logging.info(f'[{log_prefix}_{idx}] Finding islands')
    mask_image, noise_image = get_mask_and_noise(mean_image, threshold, boxsize, trim)
    if dilate > 0:
        mask_image = binary_dilation(mask_image, iterations=dilate)
    logging.info(f'[{log_prefix}_{idx}] Writing mask image {mask_fits}')
    f_written = flush_image(mask_image, header, mask_fits)
    if not f_written:
        logging.error(f'Failed to write FITS file: [{log_prefix}_{idx}] Writing mask image {mask_fits}')

    if saveaverage:
        mean_fits = os.path.join(opdir, averagetag, tagged_name(input_fits, averagetag))
        logging.info(f'[{log_prefix}_{idx}] Writing averaged mask image {mean_fits}')
        f_written = flush_image(mean_image, header, mean_fits)
        if not f_written:
            logging.error(f'Failed to write FITS file: [{log_prefix}_{idx}] Writing averaged mask image {mean_fits}')

    if savenoise:
        noise_fits = os.path.join(opdir, noisetag, tagged_name(input_fits, noisetag))
        logging.info(f'[{log_prefix}_{idx}] Writing noise image {noise_fits}')
        f_written = flush_image(noise_image, header, noise_fits)
        if not f_written:
            logging.error(f'Failed to write FITS file: [{log_prefix}_{idx}] Writing noise image {noise_fits}')


def filter_mask(mask_subset, core_lo, core_hi, minchans, specdilate, masktag, filtertag, overwrite, idx):
    """
    Filter mask images for islands spanning fewer than minchans along the spectral axis,
    then dilate along the spectral axis.

    The subset is a window of channels: a core (written out by this worker) plus a halo
    of channels either side (read for context only, written by neighbouring workers).
    With a halo of at least minchans + specdilate channels, every island touching the
    core is judged on its full spectral extent, so results do not depend on chunking
    and no two workers write the same file.

    Args:
    mask_subset (list): List of mask files (halo + core + halo).
    core_lo, core_hi (int): Core channel range within mask_subset (core_hi exclusive).
    minchans (int): Minimum number of channels below which to reject an island
    specdilate (int): Number of iterations of binary dilation in the spectral dimension.
    masktag (str): Tag for mask files.
    filtertag (str): Tag for filtered files.
    overwrite (bool): Whether to overwrite existing files.
    idx (int): Index for logging.
    """

    log_prefix = 'Filter'
    idx = str(idx).zfill(5)

    try:

        logging.info(f'[{log_prefix}_{idx}] Worker process {idx} operating on image subset {mask_subset[core_lo]} -- {mask_subset[core_hi-1]} (+ halo)')
        output_fits = []
        exists = []

        for input_fits in mask_subset[core_lo:core_hi]:
            filtered_name = tagged_name(os.path.basename(input_fits).replace(f'.{masktag}.fits', '.fits'), filtertag)
            filtered_fits = os.path.join(os.path.dirname(os.path.dirname(input_fits)), filtertag, filtered_name)
            output_fits.append(filtered_fits)
            exists.append(os.path.isfile(filtered_fits))

        if all(exists) and not overwrite:
            logging.info(f'[{log_prefix}_{idx}] Subset is complete, skipping (overwrite disabled)')
            return

        logging.info(f'[{log_prefix}_{idx}] Reading mask subset')
        cube = load_cube(mask_subset, dtype=bool)

        # Filter out islands that span fewer than minchans channels
        if minchans > 0:
            logging.info(f'[{log_prefix}_{idx}] Removing islands with fewer than {minchans} contiguous channels')
            labeled_cube, n_islands = label(cube)
            # Islands are connected, so the channel span from the bounding box equals
            # the number of distinct channels the island occupies
            keep = np.zeros(n_islands + 1, dtype=bool)
            for region_idx, bbox in enumerate(find_objects(labeled_cube), start=1):
                if bbox is not None:
                    keep[region_idx] = (bbox[2].stop - bbox[2].start) >= minchans
            cube = keep[labeled_cube]
            logging.info(f'[{log_prefix}_{idx}] Retained {keep.sum()} of {n_islands} islands')
            del labeled_cube

        # Dilate along spectral axis
        if specdilate > 0:
            logging.info(f'[{log_prefix}_{idx}] Dilating islands with {specdilate} iterations along spectral axis')
            recon_struct = np.array([1,1,1], dtype=bool)[None,None,:]
            cube = binary_dilation(cube, structure = recon_struct, iterations = specdilate)

        # Write out filtered images for the core channels only
        for i, filtered_fits in enumerate(output_fits):
            if exists[i] and not overwrite:
                logging.info(f'[{log_prefix}_{idx}] Skipping {filtered_fits} (overwrite disabled)')
            else:
                logging.info(f'[{log_prefix}_{idx}] Writing {filtered_fits}')
                template_image, header = get_image(mask_subset[core_lo + i])
                f_written = flush_image(cube[:, :, core_lo + i], header, filtered_fits)
                if not f_written:
                    logging.error(f'Failed to write FITS file: [{log_prefix}_{idx}] Writing {filtered_fits}')

        del cube
        gc.collect()

    except Exception:
        logging.exception(f'[{log_prefix}_{idx}] Filtering failed')
        raise


def count_islands(input_fits, orig_fits, idx):
    """
    Count the number of islands in a mask to inform cleaning.

    Args:
    input_fits (str): Path to the input mask FITS file.
    orig_fits (str): Path to the original FITS file.
    idx (int): Index for logging.
    """
    log_prefix = 'Counting'
    idx = str(idx).zfill(5)
    input_image,header = get_image(input_fits)
    labeled_mask_image, n_islands = label(input_image != 0)
    orig_image,orig_header = get_image(orig_fits)
    rms = np.nanstd(orig_image)
    logging.info(f'[{log_prefix}_{idx}] Channel mask parameters: {input_fits} {n_islands} {rms}')


def extract_islands(image_subset, mask_subset, core_lo, core_hi, truncated_lo, truncated_hi, opdir, catalogue, subcubes, submasks, padspatial, padspectral, catname, cubetag, overwrite, idx):
    """
    Load a subset of channels into a cube
    Label each region
    Determine centre of gravity of each region to get RA / Dec / freq ---> catalogue (if requested)
    Extract a FITS subcube for each region from the original data (if requested)

    Args:
    image_subset (list): list of input radio images
    mask_subset (list): List of mask files.
    core_lo, core_hi (int): Core channel range within the subsets (core_hi exclusive). An island
                            belongs to this worker if its first channel lies in the core, so each
                            island is extracted exactly once.
    truncated_lo, truncated_hi (bool): Whether more channels exist below / above the subset
                                       (used to warn about islands cut by the subset edge)
    catalogue (bool): Switch to write out catalogue or not
    subcubes (bool): Switch to write out subcubes or not
    padspatial (int): Padding for subcubes in the spatial dimensions
    padspectral (int): Padding for subcubes in the spectral dimension
    catname (str): output file for the catalogue (if applicable)
    cubetag (str): folder name and tag for the subcubes
    overwrite (bool): Whether to overwrite existing files.
    idx (int): Index for logging.
    """

    f_hi = 1420.40575177 # HI line in MHz
    tdl = get_tdl()

    log_prefix = 'Extract'
    idx = str(idx).zfill(5)
    logging.info(f'[{log_prefix}_{idx}] Worker process {idx} operating on image subset {mask_subset[core_lo]} -- {mask_subset[core_hi-1]} (+ halo)')

    logging.info(f'[{log_prefix}_{idx}] Reading filtered mask subset')
    cube = load_cube(mask_subset, dtype=bool)

    header = get_header(image_subset[0])
    wcs = WCS(header)
    celestial_wcs = wcs.celestial # RA/Dec only, works for 2, 3 or 4 axis headers
    # Actual frequency of every channel in the subset, read from each image, so that
    # catalogue frequencies stay correct if the channels are not evenly spaced
    freqs = [channel_frequency(get_header(f)) for f in image_subset]
    freqs = np.array([np.nan if f is None else f for f in freqs], dtype=float)
    chan_index = np.arange(len(freqs))

    # Label the cube and free up the RAM used by the boolean array
    labeled_cube, n_islands = label(cube)
    del cube
    gc.collect()
    
    logging.info(f'[{log_prefix}_{idx}] Image subset contains {n_islands} islands')
    
    bounding_boxes = find_objects(labeled_cube)
    nchan_subset = labeled_cube.shape[2]
    src_ids = []
    src_bboxes = []
    src_labels = []
    src_fcom = []
    n_truncated = 0

    for region_idx, bbox in enumerate(bounding_boxes, start = 1):

        # Only islands starting in the core belong to this worker; the rest are handled by a neighbour
        if bbox is not None and core_lo <= bbox[2].start < core_hi:

            if (bbox[2].start == 0 and truncated_lo) or (bbox[2].stop == nchan_subset and truncated_hi):
                n_truncated += 1

            sub_array = labeled_cube[bbox]
            com = center_of_mass(sub_array == region_idx)
            com = tuple(c + s.start for c, s in zip(com, bbox))
            ra, dec = celestial_wcs.pixel_to_world_values(com[1],com[0])
            ra = round(float(ra),5)
            dec = round(float(dec),5)
            ra_hms, dec_dms, src_id = format_ra_dec(ra,dec)
            src_ids.append(src_id)
            src_bboxes.append(bbox)
            src_labels.append(region_idx)
            ch_com = com[2]
            f_com = round(float(np.interp(ch_com, chan_index, freqs)) / 1e6, 3)
            src_fcom.append(f_com)
            z_com = round(((f_hi/f_com) - 1.0),4)
            ch0 = bbox[2].start
            ch1 = bbox[2].stop - 1 # last channel of the island (stop is exclusive)
            f0 = freqs[ch0] / 1e6
            f1 = freqs[ch1] / 1e6
            com_fits = image_subset[round(ch_com)].split('/')[-1]
            if catalogue:
                fname = f'{opdir}/cat_temp/{src_id}{tdl}{ra}{tdl}{dec}{tdl}{f0}{tdl}{f1}{tdl}{f_com}{tdl}{z_com}{tdl}{com_fits}'
                f = open(fname,'w')
                f.close()

    logging.info(f'[{log_prefix}_{idx}] Extracting {len(src_ids)} islands that start in this subset')
    if n_truncated > 0:
        logging.warning(f'[{log_prefix}_{idx}] {n_truncated} island(s) reach the edge of the subset and may be truncated; increase --overlap')

    if not submasks:
        del labeled_cube
        gc.collect()

    if subcubes:
        logging.info(f'[{log_prefix}_{idx}] Reading input image subset')
        data_cube = load_cube(image_subset)

        # 3D WCS (RA, Dec, Freq) for the subset, dropping any Stokes axis.
        # The spectral axis is reset for each subcube from the frequencies of its own channels.
        cube_wcs = wcs.sub(['celestial', 'spectral'])
        if cube_wcs.naxis != 3:
            cube_wcs = None # no spectral axis in the header, write a 2D WCS

        for src in range(0,len(src_ids)):
            src_id = src_ids[src]
            bbox = src_bboxes[src]
            # Centroid frequency in the name keeps islands at the same position apart
            stem = f'pony3d_{src_id}_{src_fcom[src]:.3f}MHz'
            subcube_fits_file = os.path.join(opdir, cubetag, f'{stem}_subcube.fits')
            submask_fits_file = os.path.join(opdir, cubetag, f'{stem}_subcube_mask.fits')
            if os.path.isfile(subcube_fits_file) and (not submasks or os.path.isfile(submask_fits_file)) and not overwrite:
                logging.info(f'[{log_prefix}_{idx}] Skipping {subcube_fits_file} (overwrite disabled)')
                continue
            logging.info(f'[{log_prefix}_{idx}] Writing subcube for {src_id}')

            # Apply padding to the bounding box, clipped to the data
            pads = (padspatial, padspatial, padspectral)
            expanded_bbox = tuple(
                slice(max(0, s.start - pad), min(s.stop + pad, data_cube.shape[i]))
                for i, (s, pad) in enumerate(zip(bbox, pads))
            )
            ys, xs, cs = expanded_bbox

            # Cube is (y, x, chan); FITS wants numpy order (chan, y, x)
            sub_cube_data = np.transpose(data_cube[expanded_bbox], (2, 0, 1))

            if submasks:
                # 0/1 mask of this island only (other islands in the box are excluded)
                island_mask = labeled_cube[expanded_bbox] == src_labels[src]
                sub_mask_data = np.transpose(island_mask, (2, 0, 1)).astype(np.uint8)

            # Header: WCS with CRPIX following the cut-out
            ch_mid = (cs.start + cs.stop - 1) // 2
            mid_header = get_header(image_subset[ch_mid])
            # Shift the spatial reference pixel by the cut-out offsets explicitly. (WCS.slice
            # clips offsets to the pixel shape of the header, which is 1 along the spectral
            # axis of a single-channel image.)
            sub_wcs = (cube_wcs if cube_wcs is not None else celestial_wcs).deepcopy()
            sub_wcs.wcs.crpix[0] -= xs.start
            sub_wcs.wcs.crpix[1] -= ys.start
            if cube_wcs is not None:
                # Spectral axis from the subcube's own channels: first channel is the reference
                sub_freqs = freqs[cs]
                even, spacing, gaps = check_channel_spacing(sub_freqs)
                sub_wcs.wcs.crpix[2] = 1.0
                sub_wcs.wcs.crval[2] = sub_freqs[0]
                if spacing is not None:
                    sub_wcs.wcs.cdelt[2] = spacing
                if not even:
                    logging.warning(f'[{log_prefix}_{idx}] {subcube_fits_file}: channels are not evenly spaced '
                                    f'(gap after subcube channel(s) {[int(g) for g in gaps]}); the linear frequency axis in the header is approximate')
            header = sub_wcs.to_header()
            # Carry across beam, units and other useful keywords from the central channel
            for key in ('BUNIT', 'BMAJ', 'BMIN', 'BPA', 'BTYPE', 'OBJECT', 'TELESCOP', 'OBSERVER', 'DATE-OBS', 'RESTFRQ', 'SPECSYS'):
                if key in mid_header:
                    header[key] = mid_header[key]
            header['COMMENT'] = f'{src_id}'
            header['COMMENT'] = f'pony3d-{__version__}'

            f_written = flush_image(sub_cube_data, header, subcube_fits_file)
            if not f_written:
                logging.error(f'Failed to write FITS file: sub_cube_data file {subcube_fits_file}')

            if submasks:
                mask_header = header.copy()
                for key in ('BUNIT', 'BTYPE'):
                    mask_header.remove(key, ignore_missing=True)
                f_written = flush_image(sub_mask_data, mask_header, submask_fits_file, dtype='uint8')
                if not f_written:
                    logging.error(f'Failed to write FITS file: sub_mask_data file {submask_fits_file}')
