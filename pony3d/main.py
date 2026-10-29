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
from pony3d.file_operations import create_directories, natural_sort, get_image
from pony3d.mask_operations import get_tdl, make_mask, make_averaged_mask
from pony3d.mask_operations import filter_mask, count_islands, extract_islands


def run_pool(func, iterable_params, processes, stage, logger, maxtasksperchild=None):
    """
    Run func over iterable_params with a multiprocessing Pool.
    Any exception raised in a worker is logged with its traceback and the run is stopped,
    rather than being silently swallowed.
    """
    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        try:
            return pool.starmap(func, iterable_params)
        except Exception:
            logger.exception(f'{stage} failed, stopping')
            sys.exit(1)


def channel_chunks(nchan, chanchunk, halo):
    """
    Split nchan channels into chunks of up to chanchunk core channels, each with a halo
    of up to halo channels on either side.

    Returns a list of (lo, hi, core_lo, core_hi): the window to load is [lo, hi) and the
    core channels this chunk is responsible for are [core_lo, core_hi) relative to lo.
    """
    chunks = []
    for c0 in range(0, nchan, chanchunk):
        c1 = min(c0 + chanchunk, nchan)
        lo = max(0, c0 - halo)
        hi = min(nchan, c1 + halo)
        chunks.append((lo, hi, c0 - lo, c1 - lo))
    return chunks


def available_memory_bytes(memlimit_gb, logger):
    """
    Memory available for cube workers, in bytes.
    Uses --memlimit if given, else MemAvailable from /proc/meminfo (Linux), capped by a
    SLURM allocation if one is set. Falls back to half of physical memory elsewhere.
    """
    if memlimit_gb > 0:
        return memlimit_gb * 1e9, f'--memlimit {memlimit_gb} GB'
    mem, source = None, ''
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    mem, source = int(line.split()[1]) * 1024, 'MemAvailable'
                    break
    except OSError:
        pass
    if mem is None:
        try:
            mem = 0.5 * os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            source = 'half of physical memory'
        except (ValueError, OSError, AttributeError):
            return None, 'unknown'
    slurm_mb = os.environ.get('SLURM_MEM_PER_NODE')
    if slurm_mb and slurm_mb.isdigit() and int(slurm_mb) * 1e6 < mem:
        mem, source = int(slurm_mb) * 1e6, 'SLURM_MEM_PER_NODE'
    return mem, source


def choose_jcube(requested, nchunks, j, bytes_per_worker, mem_bytes, stage, logger):
    """
    Number of workers for a cube stage: the requested value if given, otherwise the smallest of
    the number of chunks, -j, and the number of workers whose estimated footprint fits in memory.
    """
    logger.info(f'{stage}: estimated memory per worker {bytes_per_worker/1e9:.1f} GB, {nchunks} chunk(s)')
    if requested > 0:
        jcube = min(requested, nchunks)
        if mem_bytes is not None and jcube * bytes_per_worker > mem_bytes:
            logger.warning(f'{stage}: --jcube {requested} may need ~{jcube*bytes_per_worker/1e9:.1f} GB, more than the {mem_bytes/1e9:.1f} GB available')
    else:
        fit = int(mem_bytes // bytes_per_worker) if mem_bytes is not None else j
        if fit < 1:
            logger.warning(f'{stage}: one worker may need ~{bytes_per_worker/1e9:.1f} GB, more than the {mem_bytes/1e9:.1f} GB available; reduce --chanchunk')
        jcube = max(1, min(nchunks, j, fit))
    logger.info(f'{stage}: using {jcube} worker process(es)')
    return jcube


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
    preproc_group.add_argument('--regionmask', default='', metavar='', help='Provide a region file that defines areas to exclude from the subsequent processing')
    preproc_group.add_argument('--invert', action='store_true', help='Multiply images by -1 prior to processing (default = do not invert images)')

    proc_group = parser.add_argument_group('processing arguments')
    proc_group.add_argument('--minchans', type=int, default=3, metavar='', help='Minimum number of contiguous channels that an island must have to be retained (must be less than chanchunk-2; default = 3)')
    proc_group.add_argument('--dilate', type=int, default=3, metavar='', help='Iterations of dilation in spatial dimensions (default = 3, set to 0 to disable)')
    proc_group.add_argument('--specdilate', type=int, default=3, metavar='', help='Iterations of dilation in the spectral dimension (default = 3, set to 0 to disable)')
    proc_group.add_argument('--boxcar', type=int, default=1, metavar='', help='Width of boxcar window to apply (odd numbers preferred, default = 1, i.e., no averaging)')

    extraction_group = parser.add_argument_group('island extraction arguments')
    extraction_group.add_argument('--catalogue', default=False, action = 'store_true', help='Write out the 3D locations of each island (default = do not write catalogue)')
    extraction_group.add_argument('--subcubes', default=False, action = 'store_true', help='Produce 3D FITS sub-cubes from the data for each island (default = do not produce)')
    extraction_group.add_argument('--submasks', default=False, action = 'store_true', help='Produce 3D FITS sub-cubes of the mask for each island (increases memory footprint; default = do not produce)')
    extraction_group.add_argument('--padspatial', type=int, default=50, metavar='', help='Pad the subcube spatial dimensions around each island with this number of pixels (default = 50)')
    extraction_group.add_argument('--padspectral', type=int, default=10, metavar='', help='Pad the subcube spectral dimensions around each island with this number of channels (default = 10)')

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
    parallel_group.add_argument('--j', '-j', type=int, default=24, metavar='', help='Number of worker processes for the per-channel stages: mask making and island counting (default = 24)')
    parallel_group.add_argument('--jcube', type=int, default=0, metavar='', help='Number of worker processes for the cube stages: filtering and extraction (default = smallest of number of chunks, -j, and the number of workers that fit in memory)')
    parallel_group.add_argument('--memlimit', type=float, default=0, metavar='', help='Memory (GB) available to cube workers, used to choose --jcube (default = detect)')
    parallel_group.add_argument('--chanchunk', type=int, default=128, metavar='', help='Number of channels to load per worker when processing islands (default = 128)')
    parallel_group.add_argument('--overlap', type=int, default=-1, metavar = '', help='Halo of extra channels read either side of each chunk; islands are only cut at chunk edges if they extend beyond the halo (default = max(minchans + specdilate, 16))')

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
    submasks = args.submasks
    padspatial = args.padspatial
    padspectral = args.padspectral

    saveaverage = args.saveaverage
    savenoise = args.savenoise
    opdir = args.opdir or f'pony3d_output_{timestamp}'
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
    jcube = args.jcube
    memlimit = args.memlimit
    chanchunk = args.chanchunk
    overlap = args.overlap

    if overlap < 0: overlap = max(minchans + specdilate, 16)
    if overlap < minchans + specdilate:
        logger.warning(f'--overlap {overlap} is less than minchans + specdilate ({minchans + specdilate}); filtering near chunk edges may depend on chunking')


    # Input pattern for FITS files
    input_pattern = args.input_pattern

#   semaphore = multiprocessing.Semaphore(maxwrites)

    fits_list = natural_sort(glob.glob(f'*{input_pattern}*'))
    if not fits_list:
        logger.error('The specified pattern returns no files')
        sys.exit()
    else:
        nfits = len(fits_list)
        logger.info(f'Number of input images ............... : {nfits}')
        if j > nfits: j = nfits


    # Create directories
    create_directories(opdir, masktag, noisetag, averagetag, filtertag, cubetag, savenoise, saveaverage, catalogue, subcubes)
  

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
    logger.info(f'Per-channel worker processes ......... : {j}')
    logger.info(f'Cube worker processes ................ : {"auto" if jcube <= 0 else jcube}')
    logger.info(f'Channels per processing worker ....... : {chanchunk}')
    logger.info(f'Halo channels either side of chunks .. : {overlap}')
    spacer()

    # Memory estimate for the cube stages, from the image size and chunk window
    ny, nx = get_image(fits_list[0])[0].shape
    window = min(chanchunk + 2 * overlap, nfits)
    voxels_per_worker = ny * nx * window
    mem_bytes, mem_source = available_memory_bytes(memlimit, logger)
    if mem_bytes is not None:
        logger.info(f'Memory available for cube workers .... : {mem_bytes/1e9:.1f} GB ({mem_source})')
    # Bytes per voxel: filtering = bool cube + int32 labels + filtered bool + dilated bool
    #                  extraction = bool cube + int32 labels (+ float32 data cube for subcubes)
    # plus 20% headroom
    filter_bytes = 1.2 * 7 * voxels_per_worker
    extract_bytes = 1.2 * (5 + (4 if subcubes else 0)) * voxels_per_worker
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
        run_pool(make_mask, iterable_params, j, 'Mask making', logger)
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
        run_pool(make_averaged_mask, iterable_params, j, 'Boxcar mask making', logger)

    t_proc = time.time()
    # Filter masks
    if minchans != 0 or specdilate != 0:
        mask_list = natural_sort(glob.glob(f'{opdir}/{masktag}/*{input_pattern}*'))
        if not mask_list:
            logger.error('No mask images found')
            sys.exit()
        chunks = channel_chunks(len(mask_list), chanchunk, overlap)
        mask_subsets = [mask_list[lo:hi] for lo, hi, _, _ in chunks]
        logger.info(f'Filtering masks in {len(mask_subsets)} subset(s)')
        ns = len(mask_subsets)
        iterable_params = zip(
            mask_subsets, [c[2] for c in chunks], [c[3] for c in chunks],
            [minchans]*ns, [specdilate]*ns, [masktag]*ns, 
            [filtertag]*ns, [overwrite]*ns, np.arange(ns)
        )
        jf = choose_jcube(jcube, ns, j, filter_bytes, mem_bytes, 'Mask filtering', logger)
        run_pool(filter_mask, iterable_params, jf, 'Mask filtering', logger, maxtasksperchild=1)

    t_filter = time.time()
    # Count islands
    if minchans != 0 or specdilate != 0:
        mask_list = natural_sort(glob.glob(f'{opdir}/{filtertag}/*{input_pattern}*'))
        orig_list = [mask_fits.split('/')[-1].replace(f'.{filtertag}', '') for mask_fits in mask_list]
    else:
        mask_list = natural_sort(glob.glob(f'{opdir}/{masktag}/*{input_pattern}*'))
        orig_list = [mask_fits.split('/')[-1].replace(f'.{masktag}', '') for mask_fits in mask_list]

    iterable_params = zip(mask_list, orig_list, np.arange(len(mask_list)))
    run_pool(count_islands, iterable_params, j, 'Island counting', logger)

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
        nmasks = len(mask_list)
        chunks = channel_chunks(nmasks, chanchunk, overlap)
        mask_subsets = [mask_list[lo:hi] for lo, hi, _, _ in chunks]
        input_fits_subsets = [orig_list[lo:hi] for lo, hi, _, _ in chunks]
        logger.info(f'Source extraction proceeding over {len(mask_subsets)} subset(s)')

        ns = len(mask_subsets)

        iterable_params = zip(
            input_fits_subsets, mask_subsets, [c[2] for c in chunks], [c[3] for c in chunks],
            [c[0] > 0 for c in chunks], [c[1] < nmasks for c in chunks], [opdir]*ns, [catalogue]*ns, [subcubes]*ns, 
            [submasks]*ns, [padspatial]*ns, [padspectral]*ns, [catname]*ns, [cubetag]*ns,
            [overwrite]*ns, np.arange(ns)
        )
        je = choose_jcube(jcube, ns, j, extract_bytes, mem_bytes, 'Source extraction', logger)
        run_pool(extract_islands, iterable_params, je, 'Source extraction', logger, maxtasksperchild=1)

        if catalogue:
            tdl = get_tdl()
            logger.info(f'Collating detections and writing source catalogue {catname}')
            src_list = sorted(glob.glob(f'{opdir}/cat_temp/*'))
            f = open(f'{opdir}/{catname}','w')
            f.write(f'# src_id, ra, dec, f_com, z_com, f0, f1\n')
            for src in src_list:
                fp = src.split(tdl)
                # src_id, ra, dec, f_com, z_com, f0, f1 # need to round frequencies
                src_id = fp[0].split('/')[-1]
                ra = fp[1]
                dec = fp[2]
                f_com = fp[5]
                z_com = fp[6]
                f0 = round(float(fp[3]),3)
                f1 = round(float(fp[4]),3)
                f.write(f'{src_id:<25}{ra:<12}{dec:<12}{f_com:<12}{z_com:<12}{f0:<12}{f1:<12}\n')
                # f.write(f'{fp[0].split("/")[-1]:<25}{fp[1]:<12}{fp[2]:<12}{fp[5]:<12}{fp[6]:<12}{fp[3]:<12}{fp[4]:<12}\n')
            f.close()
            logger.info(f'Wrote {len(src_list)} sources')

    t_extract = time.time()

    # Report timing

    t_total = round((time.time() - t0),2)
    t_extract = round((t_extract - t_count),2)
    t_count = round((t_count - t_filter),2)
    t_filter = round((t_filter - t_proc),2)
    t_proc = round((t_proc - t0),2)

    spacer()

    logger.info(f'Mask making .......................... : {t_proc} seconds ({(round(t_proc/nfits,1))} s/channel)')
    if minchans != 0 or specdilate != 0: logger.info(f'Island processing .................... : {t_filter} seconds ({(round(t_filter/nfits,1))} s/channel)')
    logger.info(f'Island counting ...................... : {t_count} seconds ({(round(t_count/nfits,1))} s/channel)')
    if catalogue or subcubes: logger.info(f'Source extraction .................... : {t_extract} seconds ({(round(t_extract/nfits,1))} s/channel)')
    logger.info(f'Total processing time ................ : {t_total} seconds ({(round(t_total/nfits,1))} s/channel)')

    spacer()

if __name__ == '__main__':

    main()
