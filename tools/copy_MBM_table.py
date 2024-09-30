import sys
from astropy.io import fits

src_file = sys.argv[1]
dest_file = sys.argv[2]

with fits.open(src_file) as src_fits:

    if len(src_fits) < 2:
        print(f'No additional HDUs found in {src_fits}')
        sys.exit()

    mbm_hdu = src_fits[1]  
    
    with fits.open(dest_file, mode='append') as dest_fits:
        dest_fits.append(mbm_hdu)
