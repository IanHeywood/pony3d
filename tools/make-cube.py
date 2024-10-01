from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

def stack_fits_to_cube(fits_files, output_filename):
    """
    Stacks a list of 2D FITS images (with a 1-length frequency axis) into a 3D FITS cube
    and writes it to an output FITS file.
    
    Parameters:
    - fits_files: List of paths to the input FITS files.
    - output_filename: Path for the output FITS file.
    """
    
    # Read the first file to get the shape of the spatial dimensions
    with fits.open(fits_files[0]) as hdul:
        first_data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
    
    # Get the spatial shape from the first image (excluding the frequency axis)
    spatial_shape = first_data.shape[-2:]
    num_files = len(fits_files)
    
    # Initialize the cube with the correct shape: (num_files, RA, Dec)
    cube_data = np.zeros((num_files, spatial_shape[0], spatial_shape[1]))
    
    # Populate the cube with the data from each FITS file
    for i, fits_file in enumerate(fits_files):
        with fits.open(fits_file) as hdul:
            cube_data[i, :, :] = hdul[0].data.squeeze()
    
    # Modify the header to include the third axis information
    header['NAXIS'] = 3
    header['NAXIS3'] = num_files  # Number of frequency channels (or images)
    
    # Set the third axis WCS information (frequency axis, e.g., could be in GHz)
    header['CTYPE3'] = 'FREQ'
    header['CUNIT3'] = 'Hz'  # Frequency unit, can be 'Hz', 'GHz', etc.
    
    # Assuming the frequency axis values are uniformly spaced (you can adjust this as needed)
    header['CRVAL3'] = 1.0  # Reference value of the frequency axis
    header['CDELT3'] = 1.0  # Increment per frequency channel (adjust accordingly)
    header['CRPIX3'] = 1     # Reference pixel
    
    # Write the stacked data cube to a new FITS file
    hdu = fits.PrimaryHDU(data=cube_data, header=header)
    hdu.writeto(output_filename, overwrite=True)

    print(f"Output FITS cube saved as {output_filename}")

# Example usage:
# stack_fits_to_cube(['image1.fits', 'image2.fits', 'image3.fits'], 'output_cube.fits')
