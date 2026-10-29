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
  7. subcubes have (chan, y, x) layout and WCS headers that point at the source, with
     a clean 3-axis header; mask subcubes are 0/1 masks of the target island only
  8. the boxcar averaging path runs
  9. a missing channel is reported, and catalogue frequencies use the true frequencies
 10. pony3d-makecube builds a cube matching the inputs (and, with fitsio installed, a
     lossless gzip-compressed one), and pony3d-breakcube splits it back

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
           '--catalogue', '--subcubes', '--submasks', '--padspatial', '5', '--padspectral', '2',
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


def run_tool(module, args, cwd):
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''))
    p = subprocess.run([sys.executable, '-m', module] + args, cwd=cwd, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def check_cube(c, cube_file, inputs, label, hdu=0, exact=True):
    """Check a cube against the input images: data, shape, spatial WCS and channel frequencies."""
    with fits.open(cube_file) as hdul:
        data = hdul[hdu].data
        hdr = hdul[hdu].header
    expected = np.stack(inputs)
    if not c.check(data is not None and data.shape == expected.shape, f'{label}: shape {None if data is None else data.shape} (expected {expected.shape})'):
        return
    if exact:
        same = np.array_equal(np.isnan(data), np.isnan(expected)) and np.array_equal(data[~np.isnan(data)], expected[~np.isnan(expected)])
        c.check(same, f'{label}: data identical to the input images')
    w = WCS(hdr)
    ok_axes = w.naxis == 3 and hdr.get('CTYPE3') == 'FREQ' and 'CTYPE4' not in hdr
    _, _, f_first = w.pixel_to_world_values(0, 0, 0)
    _, _, f_last = w.pixel_to_world_values(0, 0, NCHAN - 1)
    ok_freq = np.isclose(f_first, FREQ0) and np.isclose(f_last, FREQ0 + (NCHAN - 1) * DF)
    ra, dec = w.celestial.pixel_to_world_values(10, 20)
    ra0, dec0 = WCS(chan_header(0)).celestial.pixel_to_world_values(10, 20)
    ok_pos = np.isclose(ra, ra0) and np.isclose(dec, dec0)
    c.check(ok_axes and ok_freq and ok_pos and 'BMAJ' in hdr and 'BUNIT' in hdr,
            f'{label}: 3-axis WCS matching the inputs (frequencies {f_first:.0f} -- {f_last:.0f} Hz), beam and units present')


def check_cube_tools(c, workdir):
    tooldir = os.path.join(workdir, 'cubetools')
    os.makedirs(tooldir, exist_ok=True)
    for f in glob.glob(os.path.join(workdir, f'{PREFIX}*.fits')):
        shutil.copy(f, tooldir)
    inputs = [np.squeeze(fits.getdata(os.path.join(tooldir, f'{PREFIX}{i:04d}-image.fits'))) for i in range(NCHAN)]

    rc, log = run_tool('pony3d.tools.makecube', ['--pattern', PREFIX, '--outfile', 'cube.fits'], tooldir)
    if c.check(rc == 0, 'pony3d-makecube exited cleanly'):
        check_cube(c, os.path.join(tooldir, 'cube.fits'), inputs, 'makecube cube')
    else:
        print('\n'.join('      ' + l for l in log.splitlines()[-15:]))
        return

    try:
        import fitsio  # noqa: F401
        have_fitsio = True
    except ImportError:
        have_fitsio = False
    if have_fitsio:
        rc, log = run_tool('pony3d.tools.makecube', ['--pattern', PREFIX, '--outfile', 'cube_gzip.fits', '--compress', 'gzip', '--tilesize', '64,64'], tooldir)
        if c.check(rc == 0, 'pony3d-makecube --compress gzip exited cleanly'):
            check_cube(c, os.path.join(tooldir, 'cube_gzip.fits'), inputs, 'gzip cube', hdu=1)
        else:
            print('\n'.join('      ' + l for l in log.splitlines()[-15:]))
        rc, log = run_tool('pony3d.tools.makecube', ['--pattern', PREFIX, '--outfile', 'cube_rice.fits', '--compress', 'rice'], tooldir)
        for line in log.splitlines():
            if line.startswith(('Compression', 'Warning')):
                print(f'      {line}')
        if c.check(rc == 0, 'pony3d-makecube --compress rice exited cleanly'):
            for name in ('cube_gzip.fits', 'cube_rice.fits'):
                with fits.open(os.path.join(tooldir, name), disable_image_compression=True) as zhdul:
                    zh = zhdul[1].header.copy()  # raw compressed header, with the Z keywords
                ztile = [zh.get(f'ZTILE{i}') for i in (1, 2, 3)]
                c.check(ztile[2] == 1, f'{name}: one channel per compression tile (ZTILE1-3 = {ztile}, ZQUANTIZ = {zh.get("ZQUANTIZ")})')
            check_cube(c, os.path.join(tooldir, 'cube_rice.fits'), inputs, 'rice cube', hdu=1, exact=False)
            rice = fits.getdata(os.path.join(tooldir, 'cube_rice.fits'), ext=1)
            err = np.nanmax(np.abs(rice - np.stack(inputs)))
            c.check(err < 0.25 * SIGMA, f'rice cube: max quantisation error {err / SIGMA:.3f} sigma')
        else:
            print('\n'.join('      ' + l for l in log.splitlines()[-15:]))
    else:
        print('  [SKIP] compressed cubes (fitsio not installed)')

    rc, log = run_tool('pony3d.tools.breakcube', ['cube.fits', 'broken'], tooldir)
    if c.check(rc == 0, 'pony3d-breakcube exited cleanly'):
        out = natural_sorted(glob.glob(os.path.join(tooldir, 'broken', '*.fits')))
        ok = len(out) == NCHAN
        for i, f in enumerate(out if ok else []):
            d = np.squeeze(fits.getdata(f))
            h = fits.getheader(f)
            if not (np.array_equal(np.isnan(d), np.isnan(inputs[i])) and np.allclose(d[~np.isnan(d)], inputs[i][~np.isnan(inputs[i])])
                    and np.isclose(h['CRVAL3'], FREQ0 + i * DF)):
                ok = False
                break
        c.check(ok, f'pony3d-breakcube: {len(out)} channel images matching the inputs in data and frequency')
    else:
        print('\n'.join('      ' + l for l in log.splitlines()[-15:]))


def natural_sorted(names):
    import re
    return sorted(names, key=lambda k: [int(t) if t.isdigit() else t for t in re.split('([0-9]+)', k)])


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
    cat_coords = SkyCoord([r[1] for r in ref_cat], [r[2] for r in ref_cat], unit=(u.deg, u.deg)) if ref_cat else None
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
        detail = f'peak at chan {(freq - FREQ0) / DF:.1f} of the input'
        if matches:
            detail += f', source chans {matches[0][2]}-{matches[0][3]}'
        else:
            detail += f', no source within {2 * PIX_DEG * 3600:.0f}" of the peak position'
        c.check(ok, f'{os.path.basename(sc)}: shape {data.shape}, {detail}')
        c.check(all(k in hdr for k in ('BMAJ', 'BMIN', 'BPA', 'BUNIT')), f'{os.path.basename(sc)}: beam and units in header')

        # Header sanity: 3 axes, no Stokes leftovers, spectral axis referenced to the first
        # channel of the subcube, at an input channel frequency, with the input spacing
        k0 = (hdr.get('CRVAL3', np.nan) - FREQ0) / DF
        problems = []
        if hdr.get('NAXIS') != 3 or hdr.get('WCSAXES', 3) != 3:
            problems.append(f'NAXIS={hdr.get("NAXIS")} WCSAXES={hdr.get("WCSAXES")}')
        if any(k in hdr for k in ('NAXIS4', 'CTYPE4', 'CRVAL4', 'CDELT4', 'CRPIX4')):
            problems.append('Stokes keywords present')
        if [hdr.get(f'CTYPE{i}') for i in (1, 2, 3)] != ['RA---SIN', 'DEC--SIN', 'FREQ']:
            problems.append(f'CTYPEs {[hdr.get(f"CTYPE{i}") for i in (1, 2, 3)]}')
        if hdr.get('CRPIX3') != 1 or abs(k0 - round(k0)) > 1e-6 or not np.isclose(hdr.get('CDELT3', 0), DF):
            problems.append(f'CRPIX3={hdr.get("CRPIX3")} CRVAL3 at input chan {k0:.3f} CDELT3={hdr.get("CDELT3")}')
        for key in ('RADESYS', 'SPECSYS'):
            if key not in hdr:
                problems.append(f'{key} missing')
        c.check(not problems, f'{os.path.basename(sc)}: header sane' + (f' ({"; ".join(problems)})' if problems else ''))

        # Mask subcube: same shape, 0/1, covers the peak, and holds only this island
        mf = sc.replace('_subcube.fits', '_subcube_mask.fits')
        if c.check(os.path.isfile(mf), f'{os.path.basename(mf)} written'):
            mdata = fits.getdata(mf)
            from scipy.ndimage import label as nd_label
            ok = (mdata.shape == data.shape and set(np.unique(mdata)) <= {0, 1}
                  and mdata[kc, ky, kx] == 1 and nd_label(mdata)[1] == 1)
            c.check(ok, f'{os.path.basename(mf)}: 0/1 mask of a single island covering the peak')

    print('Running pony3d with boxcar averaging')
    rc, log = run_pony3d(workdir, 'out_boxcar', ['--boxcar', '3', '--chanchunk', '1000'])
    if not c.check(rc == 0 and 'Traceback' not in log, 'boxcar run exited cleanly'):
        print('\n'.join('      ' + l for l in log.splitlines()[-30:]))
    nmask = len(glob.glob(os.path.join(workdir, 'out_boxcar', 'mask', '*.fits')))
    c.check(nmask == NCHAN - 2, f'boxcar run wrote {nmask} masks (expected {NCHAN - 2})')

    print('Running pony3d with a missing channel')
    gap_chan = 25
    gapdir = os.path.join(workdir, 'gap')
    os.makedirs(gapdir, exist_ok=True)
    for f in glob.glob(os.path.join(workdir, f'{PREFIX}*.fits')) + glob.glob(os.path.join(workdir, '*.reg')):
        if not f.endswith(f'{PREFIX}{gap_chan:04d}-image.fits'):
            shutil.copy(f, gapdir)
    rc, log = run_pony3d(gapdir, 'out_gap', ['--chanchunk', '1000'])
    if not c.check(rc == 0 and 'Traceback' not in log, 'missing-channel run exited cleanly'):
        print('\n'.join('      ' + l for l in log.splitlines()[-30:]))
    else:
        c.check('not evenly spaced' in log, 'missing channel reported in the log')
        gap_cat = read_catalogue(os.path.join(gapdir, 'out_gap', 'cat.txt'))
        # Sources whose masks lie entirely after the gap must keep their true frequencies
        for (x, y, c0, c1, peak) in SOURCES:
            m0, m1 = max(0, c0 - SPECDILATE), min(NCHAN - 1, c1 + SPECDILATE)
            if m0 > gap_chan:
                f_mid = (FREQ0 + 0.5 * (m0 + m1) * DF) / 1e6
                best = min(gap_cat, key=lambda r: abs(r[3] - f_mid)) if gap_cat else None
                c.check(best is not None and abs(best[3] - f_mid) < 1.0 * DF / 1e6,
                        f'after the gap: source chans {c0}-{c1} f_com {best[3] if best else None} MHz (expected {f_mid:.3f})')

    print('Round trip through pony3d-makecube and pony3d-breakcube')
    check_cube_tools(c, workdir)

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
