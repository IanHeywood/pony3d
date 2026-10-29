#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk


import argparse
from astropy.io import fits
import numpy as np
import os
import sys


def read_beam_table(hdul, nchans):
    """
    Read per-channel beams from a CASA multiple beams table (BEAMS extension), if present.

    Returns a list of (bmaj_deg, bmin_deg, bpa_deg) per channel, or None.
    Uses the column names and units in the table, and the first polarisation only.
    """
    beam_hdu = None
    for hdu in hdul[1:]:
        if isinstance(hdu, fits.BinTableHDU) and hdu.name.upper() == 'BEAMS':
            beam_hdu = hdu
            break
    if beam_hdu is None:
        return None

    table = beam_hdu.data
    names = [n.upper() for n in table.columns.names]
    units = {n.upper(): (u or '').lower() for n, u in zip(table.columns.names, table.columns.units)}
    for col in ('BMAJ', 'BMIN', 'BPA'):
        if col not in names:
            print(f'Beam table has no {col} column, ignoring it')
            return None

    def to_deg(values, col):
        unit = units.get(col, '')
        if unit.startswith('arcsec') or (unit == '' and col != 'BPA'):
            return values / 3600.0
        if unit.startswith('arcmin'):
            return values / 60.0
        if unit.startswith('rad'):
            return np.degrees(values)
        return values

    bmaj = to_deg(np.asarray(table['BMAJ'], dtype=float), 'BMAJ')
    bmin = to_deg(np.asarray(table['BMIN'], dtype=float), 'BMIN')
    bpa = to_deg(np.asarray(table['BPA'], dtype=float), 'BPA')
    chan = np.asarray(table['CHAN']) if 'CHAN' in names else np.arange(len(table))
    pol = np.asarray(table['POL']) if 'POL' in names else np.zeros(len(table), dtype=int)

    beams = [None] * nchans
    for k in np.where(pol == pol.min())[0]:
        if 0 <= chan[k] < nchans:
            beams[chan[k]] = (bmaj[k], bmin[k], bpa[k])
    if any(b is None for b in beams):
        print('Beam table does not cover every channel, ignoring it')
        return None
    return beams


def extract_frequency_planes(input_fits, output_dir):
    """
    Extracts each frequency plane from a FITS spectral line cube and saves them as individual 2D FITS images,
    retaining a length-1 third axis in the header with the correct CRVAL3, CDELT3 values for that particular channel.

    The cube may be 3D, or 4D with a length-1 Stokes axis. The frequency axis must be axis 3.

    If a CASA multiple beams table is present then it will be used to populate the beam information in the output
    images.

    Parameters:
    input_fits (str): Path to the input FITS spectral line cube.
    output_dir (str): Directory where the individual FITS images will be saved.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with fits.open(input_fits) as hdul:

        data = hdul[0].data
        header = hdul[0].header

        if data.ndim == 4:
            if data.shape[0] != 1:
                raise ValueError(f'4D cube has {data.shape[0]} Stokes planes, only one is supported.')
            data = data[0]
        if data.ndim != 3:
            raise ValueError('Input FITS file must be a 3D spectral cube (or 4D with a single Stokes plane).')
        if 'FREQ' not in str(header.get('CTYPE3', '')).upper():
            print(f'Warning: CTYPE3 is {header.get("CTYPE3")}, expected FREQ')

        nchans = data.shape[0]

        beams = read_beam_table(hdul, nchans)
        if beams is None:
            print('No usable multiple beams table found, output products will have the cube beam (if any)')

        prefix = os.path.basename(input_fits)
        if prefix.endswith('.fits'):
            prefix = prefix[:-5]

        for i in range(nchans):
            chan_data = data[i, :, :]

            chan_header = header.copy()
            chan_header['NAXIS'] = 3
            chan_header['NAXIS3'] = 1
            chan_header['CRVAL3'] = header['CRVAL3'] + (i - (header.get('CRPIX3', 1) - 1)) * header['CDELT3']
            chan_header['CRPIX3'] = 1
            chan_header['CDELT3'] = header['CDELT3']
            for key in ('NAXIS4', 'CTYPE4', 'CRVAL4', 'CDELT4', 'CRPIX4', 'CUNIT4', 'CROTA4'):
                chan_header.remove(key, ignore_missing=True)

            opstr = f'{i:05d}{chan_header["CRVAL3"]:>25}{chan_header["CDELT3"]:>25}'

            if beams is not None:
                bmaj, bmin, bpa = beams[i]
                chan_header['BMAJ'] = bmaj
                chan_header['BMIN'] = bmin
                chan_header['BPA'] = bpa
                opstr += f'{bmaj:>25}{bmin:>25}{bpa:>25}'

            if 'CASAMBM' in chan_header:
                chan_header['CASAMBM'] = False

            if 'WCSAXES' in chan_header:
                chan_header['WCSAXES'] = 3

            output_filename = os.path.join(output_dir, f'{prefix}_chan{i:05d}.fits')
            opstr += f' {output_filename}'
            print(opstr)
            fits.writeto(output_filename, chan_data[np.newaxis, :, :], chan_header, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description='Split a FITS spectral cube into one FITS image per channel')
    parser.add_argument('input_fits', help='Input FITS cube')
    parser.add_argument('output_dir', help='Folder for the output images (created if needed)')
    args = parser.parse_args()
    extract_frequency_planes(args.input_fits, args.output_dir)


if __name__ == '__main__':
    main()
