#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk


from astropy.io import fits
import numpy as np
import os
import sys


def extract_frequency_planes(input_fits, output_dir):
    """
    Extracts each frequency plane from a 3D FITS spectral line cube and saves them as individual 2D FITS images,
    retaining a length-1 third axis in the header with the correct CRVAL3, CDELT3 values for that particular channel.

    If a CASA multiple beams table is present then it will be used to populate the beam information in the output
    images.

    Parameters:
    input_fits (str): Path to the input 3D FITS spectral line cube.
    output_dir (str): Directory where the individual FITS images will be saved.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with fits.open(input_fits) as hdul:

        if len(hdul) < 2:
            print('No multiple beams table found, output products will have no beam information!')
            beams = False
        else:
            beams = True
            beam_table = hdul[1].data

        data = hdul[0].data 
        header = hdul[0].header  

        if len(data.shape) != 3:
            raise ValueError('Input FITS file must be a 3D spectral cube.')

        nchans = data.shape[0]

        for i in range(nchans):
            chan_data = data[i, :, :]

            chan_header = header.copy()
            chan_header['NAXIS'] = 3
            chan_header['NAXIS3'] = 1
            chan_header['CRPIX3'] = 1
            chan_header['CRVAL3'] = header['CRVAL3'] + (i - (header['CRPIX3'] - 1)) * header['CDELT3']
            chan_header['CDELT3'] = header['CDELT3']

            opstr = f'{i:05d}{chan_header["CRVAL3"]:>25}{chan_header["CDELT3"]:>25}'

            if beams:
                bmaj = beam_table[i][0]/3600.0
                bmin = beam_table[i][1]/3600.0
                bpa = beam_table[i][2]
                chan_header['BMAJ'] = bmaj
                chan_header['BMIN'] = bmin
                chan_header['BPA'] = bpa
                opstr += f'{bmaj:>25}{bmin:>25}{bpa:>25}'

            if 'CASAMBM' in chan_header:
                chan_header['CASAMBM'] = False

            if 'WCSAXES' in chan_header:
                chan_header['WCSAXES'] = 3

            prefix = os.path.basename(input_fits).replace('.fits', '')
            output_filename = os.path.join(output_dir, f'{prefix}_chan{i:05d}.fits')
            opstr += f' {output_filename}'
            print(opstr
                )
            fits.writeto(output_filename, chan_data[np.newaxis, :, :], chan_header, overwrite=True)

input_fits = sys.argv[1]
output_dir = sys.argv[2]

extract_frequency_planes(input_fits, output_dir)
