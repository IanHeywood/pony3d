#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk


import argparse
import os
import sys
import glob
import time
import numpy as np
from multiprocessing import Pool

from pony3d.terminal_operations import initialize_logging, hello, spacer
from pony3d.file_operations import create_directories, natural_sort
from pony3d.mask_operations import make_mask, make_averaged_mask
from pony3d.mask_operations import filter_mask, count_islands, extract_islands


def main():

    timestamp = time.strftime("%d%m%Y_%H%M%S")
    logger = initialize_logging()


    spacer()
    hello()
    spacer()


    parser = argparse.ArgumentParser(description='Parallelised production of deconvolution masks, and 3D source-finding')

    detection_group = parser.add_argument_group('detection arguments')
    detection_group.add_argument('--threshold', type=float, default=5.0, metavar='', help='Sigma threshold for masking (default = 5.0)')
    detection_group.add_argument('--boxsize', type=int, default=150, metavar='', help='Box size for background noise estimation (default = 150)')

    preproc_group = parser.add_argument_group('image pre-processing arguments')
    preproc_group.add_argument('--trim', type=int, default=0, metavar='', help='Trim this number of pixels from any NaN boundaries to avoid noisy edges (default = 0)')
    preproc_group.add_argument('--regionmask', default='', metavar='', help='Provide a region file that defines areas to exclude from the subsequent processing [NOT YET IMPLEMENTED]')
    preproc_group.add_argument('--invert', action='store_true', help='Multiply images by -1 prior to masking (default = do not invert images)')

    proc_group = parser.add_argument_group('processing arguments')
    proc_group.add_argument('--minchans', type=int, default=3, metavar='', help='Minimum number of contiguous channels that an island must have to be retained (must be less than chanchunk-2; default = 3)')
    proc_group.add_argument('--dilate', type=int, default=3, metavar='', help='Iterations of dilation in spatial dimensions (default = 3, set to 0 to disable)')
    proc_group.add_argument('--specdilate', type=int, default=3, metavar='', help='Iterations of dilation in the spectral dimension (default = 3, set to 0 to disable)')
    proc_group.add_argument('--boxcar', type=int, default=1, metavar='', help='Width of boxcar window to apply (odd numbers preferred, default = 1, i.e., no averaging)')

    extraction_group = parser.add_argument_group('island extraction arguments')
    extraction_group.add_argument('--catalogue', default=False, action = 'store_true', help='Write out the 3D locations of each island (default = do not write catalogue)')
    extraction_group.add_argument('--subcubes', default=False, action = 'store_true', help='Produce 3D FITS sub-cubes for each island (default = do not generate subcubes)')
    extraction_group.add_argument('--padspatial', default=50, metavar='', help='Pad the subcube spatial dimensions around each island with this number of pixels (default = 50)')
    extraction_group.add_argument('--padspectral', default=10, metavar='', help='Pad the subcube spectral dimensions around each island with this number of channels (default = 10)')

    output_group = parser.add_argument_group('output arguments')
    output_group.add_argument('--saveaverage', action='store_true', help='Save the boxcar-averaged images (default = do not save)')
    output_group.add_argument('--savenoise', action='store_true', help='Enable to export noise images as FITS files (default = do not save)')
    output_group.add_argument('--opdir', default='', metavar='', help='Name of folder for output products (default = auto generated)')
    output_group.add_argument('--masktag', default='mask', metavar='', help='Suffix and subfolder name for mask images (default = mask)')
    output_group.add_argument('--noisetag', default='noise', metavar='', help='Suffix and subfolder name for noise images (default = noise)')
    output_group.add_argument('--averagetag', default='boxavg', metavar='', help='Suffix and subfolder name for boxcar averaged images (default = boxavg)')
    output_group.add_argument('--filtertag', default='filtered', metavar='', help='Suffix and subfolder name for filtered images (default = filtered)')
    output_group.add_argument('--cubetag', default = 'subcubes', metavar = '', help='Suffix and folder name for subcubes (default = subcubes)')
    output_group.add_argument('--catname', default = '', metavar = '', help='Filename for output catalogue (default = auto generated)')
    output_group.add_argument('--force', dest='overwrite', action='store_true', help='Overwrite existing FITS outputs (default = do not overwrite)')

    parallel_group = parser.add_argument_group('parallelism arguments')
    parallel_group.add_argument('--j', '-j', type=int, default=24, metavar='', help='Number of worker processes (default = 24)')
    parallel_group.add_argument('--chanchunk', type=int, default=128, metavar='', help='Number of channels to load per worker when processing islands (default = 128)')
    parallel_group.add_argument('--overlap', type=int, default=-1, metavar = '', help='Number of overlapping channels between chunks (default = 2 x minchans)')

    parser.add_argument('input_pattern', help='Pattern for the sequence of FITS files to process')

    args = parser.parse_args()

    threshold = args.threshold
    boxsize = args.boxsize

    trim = abs(args.trim)
    regionmask = args.regionmask
    invert = args.invert

    minchans = args.minchans
    dilate = args.dilate
    specdilate = args.specdilate
    boxcar = args.boxcar

    catalogue = args.catalogue
    subcubes = args.subcubes
    padspatial = args.padspatial
    padspectral = args.padspectral

    saveaverage = args.saveaverage
    savenoise = args.savenoise
    opdir = args.opdir or f'pony3d.output.{timestamp}'
    masktag = args.masktag
    noisetag = args.noisetag
    averagetag = args.averagetag
    filtertag = args.filtertag
    cubetag = args.cubetag
    catname = args.catname
    overwrite = args.overwrite

    if catname == '':
        catname = f'pony3.sources.{timestamp}.txt'

    j = args.j
    chanchunk = args.chanchunk
    overlap = args.overlap

    if overlap == -1: overlap = int(2*minchans)


    # Input pattern for FITS files
    input_pattern = args.input_pattern

    pool = Pool(processes=j)
#   semaphore = multiprocessing.Semaphore(maxwrites)

    # Create directories
    create_directories(opdir, masktag, noisetag, averagetag, filtertag, cubetag, savenoise, saveaverage, catalogue, subcubes)
  
    fits_list = natural_sort(glob.glob(f'*{input_pattern}*'))
    if not fits_list:
        logger.error('The specified pattern returns no files')
        sys.exit()
    else:
        nfits = len(fits_list)
        logger.info(f'Number of input images ............... : {nfits}')
        if j > nfits: j = nfits

    nchunks = nfits // chanchunk


    # Report options for log
    logger.info(f'Detection threshold .................. : {threshold}')
    logger.info(f'Boxsize .............................. : {boxsize}')
    logger.info(f'Edge trimming iterations ............. : {"None" if trim == 0 else f"{trim}"}')
    logger.info(f'Region mask .......................... : {"None" if regionmask == "" else f"{regionmask}"}')
    logger.info(f'Invert input images .................. : {"Yes" if invert else "No"}')
    logger.info(f'Min channels per island .............. : {minchans}')
    logger.info(f'Spatial dilation iterations .......... : {dilate}')
    logger.info(f'Spectral dilation iterations ......... : {specdilate}')
    logger.info(f'Apply boxcar averaging ............... : {"Yes" if boxcar != 1 else "No"}')
    if boxcar != 1:
        logger.info(f'Channels per boxcar worker ........... : {boxcar}')
        logger.info(f'Sacrificial edge channels ............ : {boxcar // 2}')
        logger.info(f'Save boxcar averaged images .......... : {"Yes" if saveaverage else "No"}')
    logger.info(f'Write catalogue ...................... : {"Yes" if catalogue else "No"}')
    logger.info(f'Create subcubes ...................... : {"Yes" if subcubes else "No"}')
    if subcubes:
        logger.info(f'Spatial padding for subcubes ......... : {padspatial}')
        logger.info(f'Spectral padding for subcubes ........ : {padspectral}')
    logger.info(f'Save noise images .................... : {"Yes" if savenoise else "No"}')
    logger.info(f'Output folder ........................ : {opdir}')
    logger.info(f'Overwrite existing files ............. : {"Yes" if overwrite else "No"}')
    logger.info(f'Number of worker processes ........... : {j}')
    logger.info(f'Channels per processing worker ....... : {chanchunk}')
    spacer()

    t0 = time.time()

    # Make masks
    if boxcar == 1:
        iterable_params = zip(
            fits_list, [threshold]*nfits, [boxsize]*nfits, [dilate]*nfits,
            [trim]*nfits, [regionmask]*nfits, 
            [invert]*nfits, [opdir]*nfits, [masktag]*nfits, [noisetag]*nfits,
            [savenoise]*nfits, [overwrite]*nfits, np.arange(nfits)
        )
        pool.starmap(make_mask, iterable_params)
    else:
        input_fits_subsets = [
            fits_list[i:min(i + boxcar, nfits)]
            for i in range(nfits - boxcar + 1)
        ]
        ns = len(input_fits_subsets)
        logger.info(f'Sliding average will result in {ns} output images from {nfits} inputs')
        iterable_params = zip(
            input_fits_subsets, [threshold]*ns, [boxsize]*ns, 
            [dilate]*ns, [trim]*ns, [regionmask]*ns, [invert]*ns, [opdir]*ns, 
            [masktag]*ns, [noisetag]*ns, [savenoise]*ns, 
            [averagetag]*ns, [saveaverage]*ns, [overwrite]*ns, 
            np.arange(len(input_fits_subsets))
        )
        pool.starmap(make_averaged_mask, iterable_params)

    t_proc = time.time()

    # Filter masks
    if minchans != 0 or specdilate != 0:
        mask_list = natural_sort(glob.glob(f'{opdir}/{masktag}/*{input_pattern}*'))
        if not mask_list:
            logger.error('No mask images found')
            sys.exit()
        mask_subsets = [mask_list[i*chanchunk:(i+1)*chanchunk+overlap] for i in range(nchunks-1)] 
        mask_subsets.append(mask_list[(nchunks-1)*chanchunk:])
        logger.info(f'Filtering masks in {len(mask_subsets)} subset(s)')
        ns = len(mask_subsets)
        iterable_params = zip(
            mask_subsets, [specdilate]*ns, [minchans]*ns, [masktag]*ns, 
            [filtertag]*ns, [overwrite]*ns, np.arange(ns)
        )
        pool.starmap(filter_mask, iterable_params)

    t_filter = time.time()

    # Count islands
    if minchans != 0 or specdilate != 0:
        mask_list = natural_sort(glob.glob(f'{opdir}/{filtertag}/*{input_pattern}*'))
        orig_list = [mask_fits.split('/')[-1].replace(f'.{filtertag}', '') for mask_fits in mask_list]
    else:
        mask_list = natural_sort(glob.glob(f'{opdir}/{masktag}/*{input_pattern}*'))
        orig_list = [mask_fits.split('/')[-1].replace(f'.{masktag}', '') for mask_fits in mask_list]

    iterable_params = zip(mask_list, orig_list, np.arange(len(mask_list)))
    pool.starmap(count_islands, iterable_params)

    t_count = time.time()

    # Source extraction
    if catalogue or subcubes:
        if minchans != 0 or specdilate != 0:
            mask_list = natural_sort(glob.glob(f'{opdir}/{filtertag}/*{input_pattern}*'))
            orig_list = [mask_fits.split('/')[-1].replace(f'.{filtertag}', '') for mask_fits in mask_list]
        else:
            mask_list = natural_sort(glob.glob(f'{opdir}/{masktag}/*{input_pattern}*'))
            orig_list = [mask_fits.split('/')[-1].replace(f'.{masktag}', '') for mask_fits in mask_list]
        if not mask_list:
            logger.error('No mask images found')
            sys.exit()
        mask_subsets = [mask_list[i*chanchunk:(i+1)*chanchunk+overlap] for i in range(nchunks-1)] 
        mask_subsets.append(mask_list[(nchunks-1)*chanchunk:])
        input_fits_subsets = [orig_list[i*chanchunk:(i+1)*chanchunk+overlap] for i in range(nchunks-1)] 
        input_fits_subsets.append(orig_list[(nchunks-1)*chanchunk:])
        logger.info(f'Source extraction proceeding over {len(mask_subsets)} subset(s)')

        ns = len(mask_subsets)

        iterable_params = zip(
            input_fits_subsets, mask_subsets, [opdir]*ns, [catalogue]*ns, [subcubes]*ns, 
            [padspatial]*ns, [padspectral]*ns, [catname]*ns, [cubetag]*ns,
            [overwrite]*ns, np.arange(ns)
        )
        pool.starmap(extract_islands, iterable_params)

        if catalogue:
            logger.info(f'Writing source catalogue {catname}')
            src_list = sorted(glob.glob(f'{opdir}/cat_temp/*'))
            f = open(f'{opdir}/{catname}','w')
            for src in src_list:
                fp = src.split(tdl)
                f.write(f'{fp[0]:<25}{ra:<12}{dec:<12}{f_com:<12}{z_com:<12}\n')
            f.close()
            logger.info(f'Wrote {len(src_list)} sources')

    # Report timing

    t_count = round((t_count - t_filter),1)
    t_filter = round((t_filter - t_proc),1)
    t_proc = round((t_proc - t0),1)
    t_total = round((time.time() - t0),1)

    spacer()

    logger.info(f'Mask making .......................... : {t_proc} seconds ({(round(t_proc/nfits,1))} s/channel)')
    logger.info(f'Island processing .................... : {t_filter} seconds ({(round(t_filter/nfits,1))} s/channel)')
    logger.info(f'Island counting ...................... : {t_count} seconds ({(round(t_count/nfits,1))} s/channel)')
    logger.info(f'Total processing time ................ : {t_total} seconds ({(round(t_total/nfits,1))} s/channel)')

    spacer()

if __name__ == '__main__':

    main()

