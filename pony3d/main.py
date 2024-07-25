import os
import sys
import glob
import time
import numpy as np
from optparse import OptionParser
from multiprocessing import Pool

from pony3d.terminal_operations import initialize_logging, hello, spacer
from pony3d.file_operations import create_directories, natural_sort
from pony3d.mask_operations import make_mask, make_averaged_mask, filter_mask, count_islands

def main():
    # Initialize logging
    logger = initialize_logging()

    # Print welcome message
    hello()
    spacer()

    # Argument parsing
    parser = OptionParser(usage='%prog [options] input_image(s)')
    # Add all the options here...
    parser.add_option('--threshold', dest='threshold', metavar='T', help='Sigma threshold for masking (default = 5.0)', default=5.0)
    parser.add_option('--boxsize', dest='boxsize', metavar='B', help='Box size to use for background noise estimation (default = 80)', default=80)
    parser.add_option('--dilate', dest='dilate', metavar='D', help='Number of iterations of binary dilation in the spatial dimensions (default = 5, set to 0 to disable)', default=3)
    parser.add_option('--specdilate', dest='specdilate', metavar='S', help='Number of iterations of binary dilation in the spectral dimension (default = 2, set to 0 to disable, filtering must be enabled)', default=2)
    parser.add_option('--chanaverage', dest='chanaverage', metavar='N', help='Width sliding channel window to use when making masks (odd numbers preferred, default = 1, i.e. no averaging)', default=1)
    parser.add_option('--saveaverage', dest='saveaverage', help='Save the result of the sliding average (default = do not save averaged image)', action='store_true', default=False)
    parser.add_option('--chanchunk', dest='chanchunk', metavar='M', help='Number of channels to load per worker when filtering single channel instances (default = 16)', default=16)
    parser.add_option('--nofilter', dest='nofilter', help='Do not filter detections for single channel instances (default = filtering enabled)', action='store_true', default=False)
    parser.add_option('--nocount', dest='nocount', help='Do not report island counts and input RMS in the log (default = report values)', action='store_true', default=False)
    parser.add_option('--savenoise', dest='savenoise', help='Enable to export noise images as FITS files (default = do not save noise images)', action='store_true', default=False)
    parser.add_option('--invert', dest='invert', help='Multiply images by -1 prior to masking (default = do not invert images)', action='store_true', default=False)
    parser.add_option('--opdir', dest='opdir', help='Name of folder for output products (default = auto generated)', default='')
    parser.add_option('--masktag', dest='masktag', help='Suffix and subfolder name for mask images (default = mask)', default='mask')
    parser.add_option('--noisetag', dest='noisetag', help='Suffix and subfolder name for noise images (default = noise)', default='noise')
    parser.add_option('--averagetag', dest='averagetag', help='Suffix and subfolder name for boxcar averaged images (default = avg)', default='avg')
    parser.add_option('--filtertag', dest='filtertag', help='Suffix and subfolder name for filtered images (default = filtered)', default='filtered')
    parser.add_option('-f', '--force', dest='overwrite', help='Overwrite existing FITS outputs (default = do not overwrite)', action='store_true')
    parser.add_option('-j', dest='j', metavar='J', help='Number of worker processes (default = 24)', default=24)

    (options, args) = parser.parse_args()

    # Parse options
    threshold = float(options.threshold)
    boxsize = int(options.boxsize)
    dilate = int(options.dilate)
    specdilate = int(options.specdilate)
    chanaverage = int(options.chanaverage)
    saveaverage = options.saveaverage
    chanchunk = int(options.chanchunk)
    nofilter = options.nofilter
    nocount = options.nocount
    savenoise = options.savenoise
    invert = options.invert
    opdir = options.opdir or f'pony3d.output.{time.strftime("%d%m%Y_%H%M%S")}'
    masktag = options.masktag
    noisetag = options.noisetag
    averagetag = options.averagetag
    filtertag = options.filtertag
    overwrite = options.overwrite
    j = int(options.j)

    pool = Pool(processes=j)

    # Create directories
    create_directories(opdir, masktag, noisetag, averagetag, filtertag, savenoise, saveaverage, nofilter)

    # Get input FITS list
    if len(args) != 1:
        logger.error('Please specify an image pattern')
        sys.exit()
    else:
        pattern = args[0]
        fits_list = natural_sort(glob.glob(f'*{pattern}*'))
        if not fits_list:
            logger.error('The specified pattern returns no files')
            sys.exit()
        else:
            nfits = len(fits_list)
            logger.info(f'Number of input images: {nfits}')

    # Report options for log
    logger.info(f'Number of worker processes: {j}')
    logger.info(f'Output folder: {opdir}')
    logger.info(f'Detection threshold: {threshold}')
    logger.info(f'Boxsize: {boxsize}')
    if dilate > 0:
        logger.info(f'Spatial dilation iteration(s): {dilate}')
    logger.info(f'Frequency averaging: {"No" if chanaverage == 1 else "Yes"}')
    if chanaverage != 1:
        logger.info(f'Average channels per worker: {chanaverage}')
        logger.info(f'Sacrificial edge channels: {chanaverage // 2}')
    logger.info(f'Single channel filtering: {"No" if nofilter else "Yes"}')
    if not nofilter:
        logger.info(f'Spectral dilation iteration(s): {specdilate}')
        logger.info(f'Filter channels per worker: {chanchunk}')
    logger.info(f'Save noise maps: {"Yes" if savenoise else "No"}')
    logger.info(f'Overwrite existing files: {"Yes" if overwrite else "No"}')

    # Make masks
    if chanaverage == 1:
        iterable_params = zip(
            fits_list, [threshold]*nfits, [boxsize]*nfits, [dilate]*nfits,
            [invert]*nfits, [opdir]*nfits, [masktag]*nfits, [noisetag]*nfits,
            [savenoise]*nfits, [overwrite]*nfits, np.arange(nfits)
        )
        pool.starmap(make_mask, iterable_params)
    else:
        input_fits_subsets = [
            fits_list[i:min(i + chanaverage, nfits)]
            for i in range(nfits - chanaverage + 1)
        ]
        logger.info(f'Sliding average will result in {len(input_fits_subsets)} output images from {nfits} inputs')
        iterable_params = zip(
            input_fits_subsets, [threshold]*len(input_fits_subsets), [boxsize]*len(input_fits_subsets), 
            [dilate]*len(input_fits_subsets), [invert]*len(input_fits_subsets), [opdir]*len(input_fits_subsets), 
            [masktag]*len(input_fits_subsets), [noisetag]*len(input_fits_subsets), [savenoise]*len(input_fits_subsets), 
            [averagetag]*len(input_fits_subsets), [saveaverage]*len(input_fits_subsets), [overwrite]*len(input_fits_subsets), 
            np.arange(len(input_fits_subsets))
        )
        pool.starmap(make_averaged_mask, iterable_params)

    # Filter masks
    if not nofilter:
        mask_list = natural_sort(glob.glob(f'{opdir}/{masktag}/*{pattern}*'))
        if not mask_list:
            logger.error('No mask images found')
            sys.exit()
        nchunks = nfits // chanchunk
        mask_subsets = [mask_list[i*chanchunk:(i+1)*chanchunk+2] for i in range(nchunks)]
        mask_subsets.append(mask_list[(nchunks-1)*chanchunk:])
        logger.info(f'Filtering masks in {len(mask_subsets)} subsets')

        iterable_params = zip(
            mask_subsets, [specdilate]*len(mask_subsets), [masktag]*len(mask_subsets), 
            [filtertag]*len(mask_subsets), [overwrite]*len(mask_subsets), np.arange(len(mask_subsets))
        )
        pool.starmap(filter_mask, iterable_params)

    # Count islands
    if not nocount:
        if not nofilter:
            mask_list = natural_sort(glob.glob(f'{opdir}/{filtertag}/*{pattern}*'))
            orig_list = [mask_fits.split('/')[-1].replace(f'.{filtertag}', '') for mask_fits in mask_list]
        else:
            mask_list = natural_sort(glob.glob(f'{opdir}/{masktag}/*{pattern}*'))
            orig_list = [mask_fits.split('/')[-1].replace(f'.{masktag}', '') for mask_fits in mask_list]

        iterable_params = zip(mask_list, orig_list, np.arange(len(mask_list)))
        pool.starmap(count_islands, iterable_params)

if __name__ == '__main__':
    main()

