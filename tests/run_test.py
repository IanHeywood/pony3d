#!/usr/bin/env python
"""
End-to-end test for pony3d using a small synthetic image sequence.

Makes a set of single-channel FITS images (wsclean-like 4-axis headers) containing
Gaussian noise, a NaN border, single-channel noise spikes, and 3D sources with known
positions and channel ranges. Sources are placed at the first and last channels and
across chunk boundaries. pony3d is then run with several chunk sizes and the script checks:

  1. every run completes
  2. filtered masks are identical whatever the chunk size
  3. catalogues are identical whatever the chunk size (no duplicates, no losses)
  4. every injected source is found at the right position and frequency
  5. single-channel spikes are removed by filtering
  6. a source inside a region-file exclusion zone is not detected
  7. subcubes have (chan, y, x) layout and WCS headers that point at the source
  8. the boxcar averaging path runs

Usage (from anywhere, in an environment with numpy, scipy, astropy and regions):

    python tests/run_test.py [--keep] [--workdir DIR]
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

try:
    import regions  # noqa: F401
    HAVE_REGIONS = True
except ImportError:
    HAVE_REGIONS = False


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NY, NX, NCHAN = 192, 192, 60
SIGMA = 1.0e-3
FREQ0, DF = 1.30e9, 2.0e5
PIX_DEG = 2.0 / 3600.0
PREFIX = 'synth-'
SPECDILATE = 3

# Sources: (x, y, first channel, last channel, peak in units of sigma)
SOURCES = [
    (40, 50, 0, 7, 10.0),      # starts at the first channel
    (150, 40, 52, 59, 10.0),   # ends at the last channel
    (60, 140, 12, 21, 12.0),   # crosses the 16-channel chunk boundary
    (120, 120, 29, 37, 12.0),  # crosses the 32-channel boundary
    (100, 70, 40, 44, 15.0),   # short (5 channels)
    (130, 160, 3, 18, 10.0),   # long (16 channels)
]
EXCLUDED = (75, 100, 20, 28, 15.0)  # inside the region-file exclusion zone
SPIKES = [(90, 30, 10), (30, 100, 33), (160, 110, 47)]  # single-channel (x, y, chan), filtered out


def chan_header(i):
    h = fits.Header()
    h['NAXIS'] = 4
    h['NAXIS1'], h['NAXIS2'], h['NAXIS3'], h['NAXIS4'] = NX, NY, 1, 1
    h['BUNIT'] = 'JY/BEAM'
    h['BMAJ'] = h['BMIN'] = 6.0 / 3600.0
    h['BPA'] = 0.0
    h['CTYPE1'], h['CRPIX1'], h['CRVAL1'], h['CDELT1'], h['CUNIT1'] = 'RA---SIN', NX / 2 + 1, 150.0, -PIX_DEG, 'deg'
    h['CTYPE2'], h['CRPIX2'], h['CRVAL2'], h['CDELT2'], h['CUNIT2'] = 'DEC--SIN', NY / 2 + 1, -30.0, PIX_DEG, 'deg'
    h['CTYPE3'], h['CRPIX3'], h['CRVAL3'], h['CDELT3'], h['CUNIT3'] = 'FREQ', 1.0, FREQ0 + i * DF, DF, 'Hz'
    h['CTYPE4'], h['CRPIX4'], h['CRVAL4'], h['CDELT4'], h['CUNIT4'] = 'STOKES', 1.0, 1.0, 1.0, ''
    h['RADESYS'] = 'FK5'
    h['EQUINOX'] = 2000.0
    h['SPECSYS'] = 'TOPOCENT'
    return h


def make_data(workdir, seed=42):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:NY, 0:NX]
    for i in range(NCHAN):
        img = rng.normal(0.0, SIGMA, (NY, NX))
        for (x, y, c0, c1, peak) in SOURCES + ([EXCLUDED] if HAVE_REGIONS else []):
            if c0 <= i <= c1:
                img += peak * SIGMA * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2))
        for (x, y, c) in SPIKES:
            if i == c:
                img += 20 * SIGMA * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))
        img[:6, :] = np.nan
        img[:, -6:] = np.nan
        fits.writeto(os.path.join(workdir, f'{PREFIX}{i:04d}-image.fits'),
                     img[np.newaxis, np.newaxis].astype(np.float32), chan_header(i), overwrite=True)

    # Region file excluding the EXCLUDED source
    w = WCS(chan_header(0)).celestial
    ra, dec = w.pixel_to_world_values(EXCLUDED[0], EXCLUDED[1])
    with open(os.path.join(workdir, 'exclude.reg'), 'w') as f:
        f.write('# Region file format: DS9\nfk5\n')
        f.write(f'circle({ra:.6f},{dec:.6f},15")\n')


def run_pony3d(workdir, opdir, extra):
    cmd = [sys.executable, '-m', 'pony3d.main', '--threshold', '5', '--boxsize', '50',
           '--minchans', '3', '--specdilate', str(SPECDILATE), '--dilate', '2'] + (['--regionmask', 'exclude.reg'] if HAVE_REGIONS else []) + [
           '--catalogue', '--subcubes', '--padspatial', '5', '--padspectral', '2',
           '--catname', 'cat.txt', '--opdir', opdir, '-j', '4'] + extra + [PREFIX]
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''))
    p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def read_catalogue(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            src_id, ra, dec, f_com, z_com, f0, f1 = line.split()
            rows.append((src_id, ra, dec, float(f_com), float(z_com), float(f0), float(f1)))
    return sorted(rows)


def read_masks(workdir, opdir):
    files = sorted(glob.glob(os.path.join(workdir, opdir, 'filtered', '*.fits')))
    return files, np.stack([np.squeeze(fits.getdata(f)) for f in files]) if files else None


class Checker:
    def __init__(self):
        self.failures = 0

    def check(self, ok, msg):
        print(f'  [{"PASS" if ok else "FAIL"}] {msg}')
        if not ok:
            self.failures += 1
        return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--keep', action='store_true', help='Keep the working folder')
    parser.add_argument('--workdir', default='', help='Working folder (default = new temporary folder)')
    args = parser.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix='pony3d_test_')
    os.makedirs(workdir, exist_ok=True)
    print(f'Working folder: {workdir}')
    c = Checker()

    print('Making synthetic data')
    make_data(workdir)
    w2 = WCS(chan_header(0)).celestial

    # Reference run: one chunk holds every channel. Then smaller chunks with a halo
    # longer than any island (sources up to 16 channels + 2 x specdilate).
    runs = {
        'single': ['--chanchunk', '1000'],
        'chunk16': ['--chanchunk', '16', '--overlap', '24'],
        'chunk7': ['--chanchunk', '7', '--overlap', '24', '--jcube', '3'],
    }
    results = {}
    for name, extra in runs.items():
        print(f'Running pony3d ({name}: {" ".join(extra)})')
        rc, log = run_pony3d(workdir, f'out_{name}', extra)
        ok = c.check(rc == 0, f'{name}: pony3d exited cleanly')
        if not ok:
            print('\n'.join('      ' + l for l in log.splitlines()[-30:]))
            continue
        c.check('Traceback' not in log, f'{name}: no tracebacks in the log')
        files, masks = read_masks(workdir, f'out_{name}')
        c.check(masks is not None and len(files) == NCHAN, f'{name}: {len(files)} filtered masks written (expected {NCHAN})')
        results[name] = (masks, read_catalogue(os.path.join(workdir, f'out_{name}', 'cat.txt')))

    if 'single' not in results:
        print('Reference run failed, stopping')
        return 1

    ref_masks, ref_cat = results['single']
    print('Comparing chunked runs with the single-chunk run')
    for name in ('chunk16', 'chunk7'):
        if name in results:
            masks, cat = results[name]
            c.check(masks is not None and np.array_equal(masks, ref_masks), f'{name}: filtered masks identical')
            c.check(cat == ref_cat, f'{name}: catalogue identical ({len(cat)} vs {len(ref_cat)} entries)')

    print('Checking detections in the single-chunk run')
    c.check(len(ref_cat) == len(SOURCES), f'{len(ref_cat)} catalogue entries (expected {len(SOURCES)})')
    cat_coords = SkyCoord([r[1] for r in ref_cat], [r[2] for r in ref_cat], unit=(u.hourangle, u.deg)) if ref_cat else None
    for (x, y, c0, c1, peak) in SOURCES:
        ra, dec = w2.pixel_to_world_values(x, y)
        src = SkyCoord(ra * u.deg, dec * u.deg)
        # The mask spans the source channels plus SPECDILATE either side, clipped to the data
        m0, m1 = max(0, c0 - SPECDILATE), min(NCHAN - 1, c1 + SPECDILATE)
        f_mid = (FREQ0 + 0.5 * (m0 + m1) * DF) / 1e6
        if cat_coords is None:
            c.check(False, f'source at ({x},{y}) chans {c0}-{c1} found')
            continue
        sep = src.separation(cat_coords).arcsec
        k = int(np.argmin(sep))
        c.check(sep[k] < 2 * PIX_DEG * 3600 and abs(ref_cat[k][3] - f_mid) < 1.0 * DF / 1e6,
                f'source at ({x},{y}) chans {c0}-{c1}: nearest entry {sep[k]:.1f}" away, f_com {ref_cat[k][3]:.3f} MHz (expected {f_mid:.3f})')

    for (x, y, ch) in SPIKES:
        c.check(not ref_masks[ch, y, x], f'single-channel spike at ({x},{y}) chan {ch} filtered out')
    x, y, c0, c1, _ = EXCLUDED
    if HAVE_REGIONS:
        c.check(not ref_masks[c0:c1 + 1, y, x].any(), f'source in region-file exclusion zone at ({x},{y}) not masked')
    else:
        print('  [SKIP] region-file exclusion (regions package not installed)')

    print('Checking subcubes')
    subcubes = sorted(glob.glob(os.path.join(workdir, 'out_single', 'subcubes', '*_subcube.fits')))
    c.check(len(subcubes) == len(SOURCES), f'{len(subcubes)} subcubes written (expected {len(SOURCES)})')
    for sc in subcubes:
        data = fits.getdata(sc)
        hdr = fits.getheader(sc)
        wsc = WCS(hdr)
        kc, ky, kx = np.unravel_index(np.nanargmax(data), data.shape)
        ra, dec, freq = wsc.pixel_to_world_values(kx, ky, kc)
        pos = SkyCoord(ra * u.deg, dec * u.deg)
        matches = [s for s in SOURCES
                   if pos.separation(SkyCoord(*w2.pixel_to_world_values(s[0], s[1]), unit='deg')).arcsec < 2 * PIX_DEG * 3600]
        ok = (data.ndim == 3 and wsc.naxis == 3 and len(matches) == 1
              and FREQ0 + (matches[0][2] - 0.5) * DF <= freq <= FREQ0 + (matches[0][3] + 0.5) * DF)
        c.check(ok, f'{os.path.basename(sc)}: shape {data.shape}, peak maps to a source position and channel range')
        c.check(all(k in hdr for k in ('BMAJ', 'BMIN', 'BPA', 'BUNIT')), f'{os.path.basename(sc)}: beam and units in header')

    print('Running pony3d with boxcar averaging')
    rc, log = run_pony3d(workdir, 'out_boxcar', ['--boxcar', '3', '--chanchunk', '1000'])
    if not c.check(rc == 0 and 'Traceback' not in log, 'boxcar run exited cleanly'):
        print('\n'.join('      ' + l for l in log.splitlines()[-30:]))
    nmask = len(glob.glob(os.path.join(workdir, 'out_boxcar', 'mask', '*.fits')))
    c.check(nmask == NCHAN - 2, f'boxcar run wrote {nmask} masks (expected {NCHAN - 2})')

    print()
    if c.failures:
        print(f'{c.failures} check(s) FAILED. Working folder kept: {workdir}')
        return 1
    print('All checks passed')
    if not args.keep and not args.workdir:
        shutil.rmtree(workdir)
    else:
        print(f'Working folder kept: {workdir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
