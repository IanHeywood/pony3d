#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk


import argparse
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm 


import glob
import numpy as np
import os
import sys


def make_fits_cube(fitslist, output_filename):
    """
    Stacks a list of 2D FITS images (with a 1-length frequency axis) into a 3D FITS cube
    and writes it to an output FITS file.

    Parameters:
    - fitslist: List of paths to the input FITS files.
    - output_filename: Path for the output FITS file.
    """
    
    # Read the first file to determine spatial dimensions and header
    with fits.open(fitslist[0], memmap=True) as hdul:
        first_data = hdul[0].data.squeeze()
        base_header = hdul[0].header
        wcs = WCS(base_header, naxis=2)  # Keep only spatial WCS
    
    spatial_shape = first_data.shape
    num_files = len(fitslist)
    print(f'Making cube from {num_files} input images')
    
    # Initialize cube data (N_files, RA, Dec)
    cube_data = np.zeros((num_files, spatial_shape[0], spatial_shape[1]), dtype=first_data.dtype)
    
    # Frequency values array
    freqs = []
    
    # Read all images and store them in the cube
    for i, fitsfile in enumerate(tqdm(fitslist, desc="Stacking FITS files  ", unit="file")):
        with fits.open(fitsfile, memmap=True) as hdul:
            image_data = hdul[0].data.squeeze()
            cube_data[i, :, :] = image_data
            
            # Extract frequency information if available
            freq = hdul[0].header.get('CRVAL3', i + 1.0)  # Use index as fallback
            freqs.append(freq)

    print(f"Writing output cube: {output_filename}")
    # Create new 3D WCS header
    new_wcs = WCS(naxis=3)
    new_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'FREQ']
    new_wcs.wcs.cunit = ['deg', 'deg', 'Hz']
    new_wcs.wcs.crpix = [base_header['CRPIX1'], base_header['CRPIX2'], 1]
    new_wcs.wcs.cdelt = [base_header['CDELT1'], base_header['CDELT2'], np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0]
    new_wcs.wcs.crval = [base_header['CRVAL1'], base_header['CRVAL2'], freqs[0]]

    # Convert WCS to header and update keywords
    cube_header = new_wcs.to_header()
    cube_header['NAXIS'] = 3
    cube_header['NAXIS1'] = spatial_shape[1]
    cube_header['NAXIS2'] = spatial_shape[0]
    cube_header['NAXIS3'] = num_files

    # Write the stacked cube to FITS
    hdu = fits.PrimaryHDU(data=cube_data, header=cube_header)
    hdu.writeto(output_filename, overwrite=True)

    print("Done")


def read_textfile(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


from astropy.io import fits
import numpy as np
import sys

def sort_by_freq(fitslist):
    """
    Sorts a list of FITS files based on their frequency values (FREQ or CRVAL3 keyword).
    Exits the program if the frequency channels are not equally spaced.

    Parameters:
    - fitslist: List of paths to FITS files.

    Returns:
    - sorted_files: List of FITS files sorted by increasing frequency.
    """
    freq_dict = {}
    for fitsfile in tqdm(fitslist, desc="Checking frequencies ", unit="file"):
        with fits.open(fitsfile, memmap=True) as hdul:
            header = hdul[0].header
            freq = header.get('FREQ') or header.get('CRVAL3')
            if freq is None:
                print(f"Frequency information not found in {fitsfile}")
                sys.exit(1) 
            freq_dict[fitsfile] = freq

    # Sort files based on freq
    sorted_files = sorted(freq_dict.keys(), key=lambda f: freq_dict[f])
    sorted_freqs = [freq_dict[f] for f in sorted_files]

    # Check for equal freq spacing
    freq_diffs = np.diff(sorted_freqs)
    if not np.allclose(freq_diffs, freq_diffs[0], atol=1e-6):  # Tolerance to avoid floating-point issues
        print("Input files are not contiguous in frequency")
        sys.exit(1)

    if fitslist == sorted_files:
        print('Input files are in frequency order and are contiguous')
    else:
        print('Input files have been re-ordered based on their freq values')
    return sorted_files


def main():

    parser = argparse.ArgumentParser(description='Turn a sequence of 2D FITS images into a 3D spectral FITS cube')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pattern', type=str, default='', metavar='', help='Pattern to glob for to obtain image sequence')
    group.add_argument('--textfile', type=str, default='', metavar='', help='Open a text file containing the input filenames')
    parser.add_argument('--outfile', type=str, default='', metavar='', help='Name of output FITS cube', required=True)
    args = parser.parse_args()    

    if args.pattern:
        fitslist = sorted(glob.glob(f'*{args.pattern}*.fits'))
        if len(fitslist) == 0:
            print('Provided pattern returns no files. Note that files must have a .fits suffix.')
            sys.exit()

    elif args.textfile:
        print(f'Reading {args.textfile}')
        fitslist = read_textfile(args.textfile)
        found = True
        for fitsfile in tqdm(fitslist, desc="Checking files exist ", unit="file"):
            if not os.path.isfile(fitsfile):
                print(f'     --- {fitsfile} not found')
                found = False
        if not found:
            print('Please check missing files')
            sys.exit()

    fitslist = sort_by_freq(fitslist)
 
    make_fits_cube(fitslist,args.outfile)



if __name__ == '__main__':

    main()

