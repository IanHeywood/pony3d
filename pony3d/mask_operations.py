#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk


import gc
import glob
import logging
import numpy as np
import os
import scipy.special
import time
import traceback
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from regions import Regions
from scipy.ndimage import binary_dilation, binary_erosion, convolve, minimum_filter
from scipy.ndimage import find_objects, center_of_mass, label, binary_fill_holes
from pony3d.file_operations import get_image, flush_image, load_cube
from pony3d import __version__


def get_tdl():
    tdl = '_-pony-_' # weird delimiter string to avoid split errors from the catalogue temp files
    return tdl


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

    regs = Regions.read(region_file) #, format='ds9')
    mask = np.zeros_like(image_data) #, dtype='int')

    if wcs.naxis > 2:
        wcs = wcs.sub([1,2])

    for reg in regs:
        if hasattr(reg, 'to_pixel'):
            reg = reg.to_pixel(wcs)
            mask += reg.to_mask().to_image(mask.shape)

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
    noise_image = -minimum_filter(input_image, box) / ratio
    noise_image[noise_image < 0.0] = 1.0e-10
    median_noise = np.median(noise_image)
    noise_image[noise_image < median_noise] = median_noise
    mask_image = input_image > threshold * noise_image

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
    mask_fits = os.path.join(opdir, masktag, input_fits.replace('.fits', f'.{masktag}.fits'))
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
    flush_image(mask_image, header, mask_fits)
    if savenoise:
        noise_fits = os.path.join(opdir, noisetag, input_fits.replace('.fits', f'.{noisetag}.fits'))
        logging.info(f'[{log_prefix}_{idx}] Writing noise image {noise_fits}')
        flush_image(noise_image, header, noise_fits)


def make_averaged_mask(input_fits_subset, threshold, boxsize, dilate, invert, trim, regionmask, opdir, masktag, noisetag, savenoise, averagetag, saveaverage, overwrite, idx):
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
    log_prefix = 'BoxcarMask'
    idx = str(idx).zfill(5)
    logging.info(f'[{log_prefix}_{idx}] Worker process {idx} operating on image subset {input_fits_subset[0]} -- {input_fits_subset[-1]}')
    nfits = len(input_fits_subset)
    input_fits = input_fits_subset[nfits // 2]
    input_image, header = get_image(input_fits) # Need a WCS for image output and any region masking
    mask_fits = os.path.join(opdir, masktag, input_fits.replace('.fits', f'.{masktag}.fits'))
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
    mask_image, noise_image = get_mask_and_noise(mean_image, threshold, boxsize, dilate, trim)
    if dilate > 0:
        mask_image = binary_dilation(mask_image, iterations=dilate)
    logging.info(f'[{log_prefix}_{idx}] Writing mask image {mask_fits}')
    flush_image(mask_image, header, mask_fits)
    if saveaverage:
        mean_fits = os.path.join(opdir, averagetag, input_fits.replace('.fits', f'.{averagetag}.fits'))
        logging.info(f'[{log_prefix}_{idx}] Writing averaged mask image {mean_fits}')
        flush_image(mean_image, header, mean_fits)
    if savenoise:
        noise_fits = os.path.join(opdir, noisetag, input_fits.replace('.fits', f'.{noisetag}.fits'))
        logging.info(f'[{log_prefix}_{idx}] Writing noise image {noise_fits}')
        flush_image(noise_image, header, noise_fits)


def filter_mask(mask_subset, minchans, specdilate, masktag, filtertag, overwrite, idx):
    """
    Filter mask images for islands spanning fewer than minchans along the spectral axis.

    Args:
    mask_subset (list): List of mask files.
    minchans (int): Minimum number of channels below which to reject an island
    specdilate (int): Number of iterations of binary dilation in the spectral dimension.
    masktag (str): Tag for mask files.
    filtertag (str): Tag for filtered files.
    overwrite (bool): Whether to overwrite existing files.
    idx (int): Index for logging.
    """
    
    try:

        log_prefix = 'Filter'
        idx = str(idx).zfill(5)
        logging.info(f'[{log_prefix}_{idx}] Worker process {idx} operating on image subset {mask_subset[0]} -- {mask_subset[-1]}')
        template_fits = []
        output_fits = []
        exists = []

        for input_fits in mask_subset:
            filtered_fits = input_fits.replace(masktag, filtertag)
            template_fits.append(input_fits)
            output_fits.append(filtered_fits)
            exists.append(os.path.isfile(filtered_fits))

        if all(exists) and not overwrite:
            logging.info(f'[{log_prefix}_{idx}] Subset is complete, skipping (overwrite disabled)')
            return

        logging.info(f'[{log_prefix}_{idx}] Reading mask subset')
        cube = load_cube(mask_subset) != 0
        cube = cube.astype(bool)

        # Filter out islands that span fewer than minchans channels
        if minchans > 0:
            logging.info(f'[{log_prefix}_{idx}] Removing islands with fewer than {minchans} contiguous channels')
            labeled_cube, n_islands = label(cube)
            island_sizes = np.zeros(n_islands + 1, dtype=int)
            flat_indices = np.where(labeled_cube > 0)
            flat_labels = labeled_cube[flat_indices]
            linear_indices = flat_labels * cube.shape[2] + flat_indices[2]
            counts = np.bincount(linear_indices, minlength=(n_islands + 1) * cube.shape[2])
            counts = counts.reshape(n_islands + 1, cube.shape[2])
        
            unique_counts = np.sum(counts > 0, axis=1)
            small_islands = np.where(unique_counts < minchans)[0]
            mask = np.isin(labeled_cube, small_islands)
            cube[mask] = False

        # Dilate along spectral axis
        if specdilate != 0:
            logging.info(f'[{log_prefix}_{idx}] Dilating islands with {specdilate} iterations along spectral axis')
            recon_struct = np.array([1,1,1])[None,None,:]
            cube = binary_dilation(cube, structure = recon_struct)

        # Write out filtered images
        for i, filtered_fits in enumerate(output_fits):
            if exists[i] and not overwrite:
                logging.info(f'[{log_prefix}_{idx}] Skipping {filtered_fits} (overwrite disabled)')
            else:
                logging.info(f'[{log_prefix}_{idx}] Writing {filtered_fits}')
                template_image, header = get_image(template_fits[i])
                flush_image(cube[:, :, i], header, filtered_fits)

        del labeled_cube
        del cube
        gc.collect()

    except:
        traceback.print_exc()


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
    input_image = input_image.byteswap().view(input_image.dtype.newbyteorder())
#   input_image = input_image.byteswap().newbyteorder()  
    labeled_mask_image, n_islands = label(input_image)
    orig_image,orig_header = get_image(orig_fits)
    rms = np.nanstd(orig_image)
    logging.info(f'[{log_prefix}_{idx}] Channel mask parameters: {input_fits} {n_islands} {rms}')


def extract_islands(image_subset, mask_subset, opdir, catalogue, subcubes, submasks, padspatial, padspectral, catname, cubetag, overwrite, idx):
    """
    Load a subset of channels into a cube
    Label each region
    Determine centre of gravity of each region to get RA / Dec / freq ---> catalogue (if requested)
    Extract a FITS subcube for each region from the original data (if requested)

    Args:
    image_subset (list): list of input radio images
    mask_subset (list): List of mask files.
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
    logging.info(f'[{log_prefix}_{idx}] Worker process {idx} operating on image subset {mask_subset[0]} -- {mask_subset[-1]}')
    template_fits = []
    output_fits = []
    exists = []

    logging.info(f'[{log_prefix}_{idx}] Reading filtered mask subset')
    cube = load_cube(mask_subset) 
    cube = cube.astype(bool)

    dummy_img, header = get_image(mask_subset[0])
    wcs = WCS(header)
    freq0 = header.get('CRVAL3')
    df = header.get('CDELT3')

    # Label the cube and free up the RAM used by the boolean array and the template FITS image
    labeled_cube, n_islands = label(cube)
    del cube 
    del dummy_img
    gc.collect()
    
    logging.info(f'[{log_prefix}_{idx}] Image subset contains {n_islands} islands')
    
    bounding_boxes = find_objects(labeled_cube)
    src_ids = []

    for region_idx, bbox in enumerate(bounding_boxes, start = 1):

        if bbox[2].stop < labeled_cube.shape[2]: # check that the island does not butt up against the top end of the cube
           
            sub_array = labeled_cube[bbox]
            com = center_of_mass(sub_array == region_idx)
            com = tuple(c + s.start for c, s in zip(com, bbox))
            ra, dec, xxx = wcs.pixel_to_world_values(com[1],com[0],0)
            ra = round(float(ra),5)
            dec = round(float(dec),5)
            ra_hms, dec_dms, src_id = format_ra_dec(ra,dec)
            src_ids.append(src_id)
            ch_com = com[2]
            f_com = round(((freq0+(ch_com*df))/1e6),3)
            z_com = round(((f_hi/f_com) - 1.0),4)
            ch0 = bbox[2].start 
            ch1 = bbox[2].stop
            f0 = (freq0+(ch0*df))/1e6
            f1 = (freq0+(ch1*df))/1e6
            com_fits = image_subset[round(ch_com)].split('/')[-1]
            if catalogue:
                fname = f'{opdir}/cat_temp/{src_id}{tdl}{ra}{tdl}{dec}{tdl}{f0}{tdl}{f1}{tdl}{f_com}{tdl}{z_com}{tdl}{com_fits}'
                f = open(fname,'w')
                f.close()

    if not submasks:
        del labeled_cube
        gc.collect()

    if subcubes:
        logging.info(f'[{log_prefix}_{idx}] Reading input image subset')
        data_cube = load_cube(image_subset)
        nchan = len(image_subset)

        for src in range(0,len(src_ids)):
            src_id = src_ids[src]
            logging.info(f'[{log_prefix}_{idx}] Writing subcube for {src_id}')
            bbox = bounding_boxes[src]
            centre_index = tuple((s.start + s.stop - 1) // 2 for s in bbox)
            ra, dec, xxx = wcs.pixel_to_world_values(centre_index[1],centre_index[0],0)
            ch0 = np.max((0,bbox[2].start))
            ch1 = np.min((bbox[2].stop,nchan))
            ch_mid = (ch0 + ch1 - 1) // 2

            sub_f0 = freq0 + (ch0*df)

            # Apply padding to bounding boxes if requested
            if padspatial != 0 or padspectral != 0:
                expanded_bbox = []
                for i, s in enumerate(bbox):
                    if i in (0, 1): 
                        expand = padspatial
                    elif i == 2: 
                        expand = padspectral
                    new_start = max(0, s.start - expand)  # Ensure start is not negative
                    new_stop = s.stop + expand
                    new_stop = min(new_stop, data_cube.shape[i])
                    expanded_bbox.append(slice(new_start, new_stop))
            else:
                expanded_bbox = bbox
            expanded_bbox = tuple(expanded_bbox)
            sub_cube_data = data_cube[expanded_bbox]
            sub_cube_data = np.transpose(sub_cube_data, (2, 1, 0)) # because why?

            if submasks:
                labeled_sub_cube_data = labeled_cube[expanded_bbox]
                labeled_sub_cube_data = np.transpose(labeled_sub_cube_data, (2, 1, 0))

            dummy_img, header = get_image(image_subset[ch_mid])

            # Sort out cube headers
            if 'CASAMBM' in header:
                header.remove('CASAMBM')
            header.remove('COMMENT', remove_all=True)
            header['CRVAL1'] = float(ra)  # RA reference value
            header['CRPIX1'] = int(centre_index[1] - bbox[1].start)  # Adjust RA reference pixel
            header['CRVAL2'] = float(dec)  # Dec reference value
            header['CRPIX2'] = int(centre_index[0] - bbox[0].start)  # Adjust Dec reference pixel
            header['CRVAL3'] = float(sub_f0)
            header['CRPIX3'] = 1 
            header['COMMENT'] = f'{src_id}'
            header['COMMENT'] = f'pony3d-{__version__}'
     
            subcube_fits_file = os.path.join(opdir, cubetag, f'pony3d_{src_id}_subcube.fits')
            flush_image(sub_cube_data, header, subcube_fits_file)
    
            if submasks:
                labeled_subcube_fits_file = os.path.join(opdir, cubetag, f'pony3d_{src_id}_subcube_label.fits')
                flush_image(labeled_sub_cube_data, header, labeled_subcube_fits_file)




