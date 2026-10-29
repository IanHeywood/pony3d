#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk

"""
Turn a sequence of single-channel FITS images into a 3D spectral FITS cube.

The cube is written one channel at a time, so memory use is a single image
regardless of the size of the cube. Optionally the cube can be tile-compressed
(requires fitsio): GZIP_2 is lossless, RICE_1 quantises floating point data. Compression
stages the channels in a temporary memory-mapped file and compresses it in one write.
"""

import argparse
import glob
import inspect
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from pony3d.file_operations import channel_frequency, check_channel_spacing, natural_sort

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    def tqdm(iterable, **kwargs):
        return iterable


# Keywords carried across from the first input image (beam, units, frames, metadata)
COPY_KEYS = ('BUNIT', 'BMAJ', 'BMIN', 'BPA', 'BTYPE', 'SPECSYS', 'RESTFRQ', 'RESTFREQ',
             'OBJECT', 'TELESCOP', 'INSTRUME', 'OBSERVER', 'DATE-OBS', 'MJD-OBS',
             'OBSRA', 'OBSDEC', 'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z')


def read_image(fits_file):
    """Read the 2D image from a FITS file, dropping degenerate (length-1) axes."""
    with fits.open(fits_file, memmap=True, ignore_missing_simple=True) as hdul:
        return np.squeeze(np.asarray(hdul[0].data)), hdul[0].header.copy()


def cube_header(base_header, freqs, nx, ny, bitpix):
    """
    Build the header for the output cube.

    Spatial WCS (projection, reference, any rotation / PC matrix, RADESYS, EQUINOX) comes from
    the first input image; the spectral axis comes from the measured channel frequencies.
    Stokes and other degenerate axes are dropped.
    """
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = bitpix
    header['NAXIS'] = 3
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['NAXIS3'] = len(freqs)

    celestial = WCS(base_header).celestial
    header.extend(celestial.to_header(), update=True)
    header['WCSAXES'] = 3
    header['CTYPE3'] = 'FREQ'
    header['CUNIT3'] = 'Hz'
    header['CRPIX3'] = 1.0
    header['CRVAL3'] = float(freqs[0])
    header['CDELT3'] = float(np.median(np.diff(freqs))) if len(freqs) > 1 else float(base_header.get('CDELT3', 1.0))

    for key in COPY_KEYS:
        if key in base_header:
            header[key] = base_header[key]
    return header


def write_cube_streamed(fitslist, freqs, outfile, overwrite=False):
    """Write an uncompressed float32 cube channel by channel with astropy's StreamingHDU."""
    image0, header0 = read_image(fitslist[0])
    ny, nx = image0.shape
    header = cube_header(header0, freqs, nx, ny, bitpix=-32)

    if os.path.exists(outfile):
        if not overwrite:
            print(f'{outfile} exists, use --overwrite')
            sys.exit(1)
        os.remove(outfile)

    shdu = fits.StreamingHDU(outfile, header)
    try:
        for fitsfile in tqdm(fitslist, desc='Stacking FITS files  ', unit='file'):
            image, _ = read_image(fitsfile)
            if image.shape != (ny, nx):
                raise ValueError(f'{fitsfile}: expected spatial shape {(ny, nx)}, got {image.shape}')
            shdu.write(image.astype('>f4'))
    finally:
        shdu.close()


def compressed_tile_shape(fits_file):
    """ZTILE1-3 of the first tile-compressed HDU, read from the raw (compressed) header."""
    with fits.open(fits_file, disable_image_compression=True) as hdul:
        for hdu in hdul[1:]:
            if hdu.header.get('ZIMAGE'):
                return [hdu.header.get(f'ZTILE{i}', 1) for i in (1, 2, 3)]
    return None


def write_cube_compressed(fitslist, freqs, outfile, method, qlevel, tile_xy, overwrite=False):
    """
    Write a tile-compressed float32 cube with fitsio.

    Writing a lossy (RICE) compressed image piece by piece corrupts it in fitsio / cfitsio
    (each partial write re-quantises tiles; errors of tens of sigma in tests), so the channels
    are first streamed into a temporary uncompressed file next to the output, which is then
    compressed in a single write. The temporary file is memory-mapped, so memory use stays
    low, but it needs free disk space equal to the uncompressed cube.

    The compressed image is stored in the first extension (the primary HDU is empty,
    as the FITS tile-compression convention requires), and the WCS is written there.
    """
    try:
        import fitsio
    except ImportError:
        print('Compression needs the fitsio package: pip install fitsio')
        sys.exit(1)

    image0, header0 = read_image(fitslist[0])
    ny, nx = image0.shape
    nchan = len(fitslist)
    header = cube_header(header0, freqs, nx, ny, bitpix=-32)

    if os.path.exists(outfile) and not overwrite:
        print(f'{outfile} exists, use --overwrite')
        sys.exit(1)

    tx, ty = (int(s.strip()) for s in tile_xy.split(','))
    tile_dims = [1, max(1, min(ty, ny)), max(1, min(tx, nx))]  # numpy order: one channel per tile
    params = inspect.signature(fitsio.FITS.write).parameters
    tile_key = 'tile_dims' if 'tile_dims' in params else 'tiledims' # name differs between fitsio versions

    tmpfile = f'{outfile}.pony3d-tmp.raw'
    print(f'Staging channels in {tmpfile} ({nchan * ny * nx * 4 / 1e9:.2f} GB, removed afterwards)')
    try:
        staged = np.memmap(tmpfile, dtype=np.float32, mode='w+', shape=(nchan, ny, nx))
        for k, fitsfile in enumerate(tqdm(fitslist, desc='Stacking FITS files  ', unit='file')):
            image, _ = read_image(fitsfile)
            if image.shape != (ny, nx):
                raise ValueError(f'{fitsfile}: expected spatial shape {(ny, nx)}, got {image.shape}')
            staged[k] = image
        staged.flush()

        print(f'Compressing with {method}' + (f', qlevel {qlevel}' if qlevel is not None else ' (lossless)'))
        with fitsio.FITS(outfile, mode='rw', clobber=True) as f:
            f.write(staged, compress=method, qlevel=qlevel, **{tile_key: tile_dims})
            # WCS and other keywords; structural keywords are managed by fitsio
            structural = {'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'EXTEND'}
            f[-1].write_keys([{'name': card.keyword, 'value': card.value, 'comment': card.comment}
                              for card in header.cards if card.keyword and card.keyword not in structural])
        del staged
    finally:
        if os.path.exists(tmpfile):
            os.remove(tmpfile)

    got = compressed_tile_shape(outfile)
    if got is not None and got[2] != 1:
        print(f'Warning: compression tiles are {got[0]} x {got[1]} x {got[2]} pixels; expected one channel per tile')
    elif got is not None:
        print(f'Compression tiles of {got[0]} x {got[1]} x {got[2]} pixels')


def read_textfile(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def sort_by_freq(fitslist):
    """
    Sorts a list of FITS files by channel frequency.
    Exits the program if the frequency channels are not equally spaced.

    Returns:
    - sorted_files: List of FITS files sorted by increasing frequency.
    - sorted_freqs: Their frequencies (Hz).
    """
    freq_dict = {}
    for fitsfile in tqdm(fitslist, desc='Checking frequencies ', unit='file'):
        header = fits.getheader(fitsfile)
        freq = channel_frequency(header)
        if freq is None:
            freq = header.get('FREQ')
        if freq is None:
            print(f'Frequency information not found in {fitsfile}')
            sys.exit(1)
        freq_dict[fitsfile] = freq

    sorted_files = sorted(freq_dict.keys(), key=lambda f: freq_dict[f])
    sorted_freqs = [freq_dict[f] for f in sorted_files]

    if len(set(sorted_freqs)) != len(sorted_freqs):
        print('Some input files have the same frequency')
        sys.exit(1)
    even, spacing, gaps = check_channel_spacing(sorted_freqs)
    if not even:
        print('Input files are not contiguous in frequency, gaps after:')
        for g in gaps[:10]:
            print(f'     --- {sorted_files[g]}')
        sys.exit(1)

    if fitslist == sorted_files:
        print('Input files are in frequency order and are contiguous')
    else:
        print('Input files have been re-ordered based on their freq values')
    return sorted_files, sorted_freqs


def main():

    parser = argparse.ArgumentParser(description='Turn a sequence of 2D FITS images into a 3D spectral FITS cube')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pattern', type=str, default='', metavar='', help='Pattern to glob for to obtain image sequence')
    group.add_argument('--textfile', type=str, default='', metavar='', help='Open a text file containing the input filenames')
    parser.add_argument('--outfile', type=str, default='', metavar='', help='Name of output FITS cube', required=True)
    parser.add_argument('--compress', type=str, default='none', choices=['none', 'gzip', 'rice'],
                        help='Tile compression, needs fitsio: gzip = GZIP_2, lossless; rice = RICE_1, quantised (lossy) '
                             'at --qlevel. Needs temporary disk space equal to the uncompressed cube (default = none)')
    parser.add_argument('--qlevel', type=float, default=16.0, metavar='',
                        help='Quantisation level for rice compression: values are quantised to the tile noise / qlevel, '
                             'so higher keeps more precision (default = 16; fpack uses 4)')
    parser.add_argument('--tilesize', type=str, default='256,256', metavar='',
                        help='Compression tile size in spatial pixels, x,y (default = 256,256)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing output cube')
    args = parser.parse_args()

    if args.pattern:
        fitslist = natural_sort(glob.glob(f'*{args.pattern}*.fits'))
        if len(fitslist) == 0:
            print('Provided pattern returns no files. Note that files must have a .fits suffix.')
            sys.exit(1)

    elif args.textfile:
        print(f'Reading {args.textfile}')
        fitslist = read_textfile(args.textfile)
        found = True
        for fitsfile in tqdm(fitslist, desc='Checking files exist ', unit='file'):
            if not os.path.isfile(fitsfile):
                print(f'     --- {fitsfile} not found')
                found = False
        if not found:
            print('Please check missing files')
            sys.exit(1)

    fitslist, freqs = sort_by_freq(fitslist)
    print(f'Making cube from {len(fitslist)} input images')

    if args.compress == 'none':
        write_cube_streamed(fitslist, freqs, args.outfile, overwrite=args.overwrite)
    else:
        method = {'gzip': 'GZIP_2', 'rice': 'RICE_1'}[args.compress]
        qlevel = None if args.compress == 'gzip' else args.qlevel  # None = no quantisation (lossless)
        write_cube_compressed(fitslist, freqs, args.outfile, method, qlevel, args.tilesize, overwrite=args.overwrite)

    print(f'Written {args.outfile}')


if __name__ == '__main__':

    main()
