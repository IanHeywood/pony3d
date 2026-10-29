# pony3d

## What is this?

* `pony3d` is a tool for automatically generating FITS masks for spectral line deconvolution.

* The input is a set of dirty (or restored) images, with a single FITS image per frequency plane and a sequential filename ordering that corresponds to the frequency ordering (as produced by e.g. [`wsclean`](https://gitlab.com/aroffringa/wsclean)).

* Rapid execution time is achieved by parallelising the operations over channels (or groups of channels).

* Filtering for single-channel islands can be applied, allowing the detection threshold to be relatively low. Boxcar averaging and 3D dilation is also included.

* The software is also a prototype 3D source finder, and is in an active state of development.

## Installation

```
pip install pony3d
```

## Command line options

Note that `input_pattern` is a pattern. Specifying `my_images` will return a naturally-sorted list of all files in the current directory that match `*my_images*`.

```
usage: pony3d [-h] [--threshold THRESHOLD] [--boxsize BOXSIZE] [--trim TRIM]
              [--regionmask REGIONMASK] [--invert] [--minchans MINCHANS] [--dilate DILATE]
              [--specdilate SPECDILATE] [--boxcar BOXCAR] [--catalogue] [--subcubes] [--submasks]
              [--padspatial PADSPATIAL] [--padspectral PADSPECTRAL] [--saveaverage] [--savenoise]
              [--opdir OPDIR] [--masktag MASKTAG] [--noisetag NOISETAG] [--averagetag AVERAGETAG]
              [--filtertag FILTERTAG] [--cubetag CUBETAG] [--catname CATNAME] [--force] [--j J]
              [--jcube JCUBE] [--memlimit MEMLIMIT] [--chanchunk CHANCHUNK] [--overlap OVERLAP]
              input_pattern

Parallelised production of deconvolution masks, and 3D source-finding

positional arguments:
  input_pattern         Pattern for the sequence of FITS files to process

options:
  -h, --help            show this help message and exit

detection arguments:
  --threshold THRESHOLD
                        Sigma threshold for masking (default = 5.0)
  --boxsize BOXSIZE     Box size for background noise estimation (default = 150)

image pre-processing arguments:
  --trim TRIM           Trim this number of pixels from any NaN boundaries to avoid noisy edges
                        (default = 0)
  --regionmask REGIONMASK
                        Provide a region file that defines areas to exclude from the subsequent
                        processing
  --invert              Multiply images by -1 prior to processing (default = do not invert images)

processing arguments:
  --minchans MINCHANS   Minimum number of contiguous channels that an island must have to be
                        retained (must be less than chanchunk-2; default = 3)
  --dilate DILATE       Iterations of dilation in spatial dimensions (default = 3, set to 0 to
                        disable)
  --specdilate SPECDILATE
                        Iterations of dilation in the spectral dimension (default = 3, set to 0 to
                        disable)
  --boxcar BOXCAR       Width of boxcar window to apply (odd numbers preferred, default = 1, i.e.,
                        no averaging)

island extraction arguments:
  --catalogue           Write out the 3D locations of each island (default = do not write
                        catalogue)
  --subcubes            Produce 3D FITS sub-cubes from the data for each island (default = do not
                        produce)
  --submasks            Produce 3D FITS sub-cubes of the mask for each island (increases memory
                        footprint; default = do not produce)
  --padspatial PADSPATIAL
                        Pad the subcube spatial dimensions around each island with this number of
                        pixels (default = 50)
  --padspectral PADSPECTRAL
                        Pad the subcube spectral dimensions around each island with this number of
                        channels (default = 10)

output arguments:
  --saveaverage         Save the boxcar-averaged images (default = do not save)
  --savenoise           Enable to export noise images as FITS files (default = do not save)
  --opdir OPDIR         Name of folder for output products (default = auto generated)
  --masktag MASKTAG     Suffix and subfolder name for mask images (default = mask)
  --noisetag NOISETAG   Suffix and subfolder name for noise images (default = noise)
  --averagetag AVERAGETAG
                        Suffix and subfolder name for boxcar averaged images (default = boxavg)
  --filtertag FILTERTAG
                        Suffix and subfolder name for filtered images (default = filtered)
  --cubetag CUBETAG     Suffix and folder name for subcubes (default = subcubes)
  --catname CATNAME     Filename for output catalogue (default = auto generated)
  --force               Overwrite existing FITS outputs (default = do not overwrite)

parallelism arguments:
  --j, -j J             Number of worker processes for the per-channel stages: mask making and
                        island counting (default = 24)
  --jcube JCUBE         Number of worker processes for the cube stages: filtering and extraction
                        (default = smallest of number of chunks, -j, and the number of workers
                        that fit in memory)
  --memlimit MEMLIMIT   Memory (GB) available to cube workers, used to choose --jcube (default =
                        detect)
  --chanchunk CHANCHUNK
                        Number of channels to load per worker when processing islands (default =
                        128)
  --overlap OVERLAP     Halo of extra channels read either side of each chunk; islands are only
                        cut at chunk edges if they extend beyond the halo (default = max(minchans
                        + specdilate, 16))
```

## Parallelism and memory

Processing runs in two kinds of stage:

* **Per-channel stages** (mask making, island counting) work on one image at a time. They are limited by CPU and disk throughput, and use `-j` workers.
* **Cube stages** (filtering, extraction) load chunks of `--chanchunk` channels, plus a halo of `--overlap` channels either side, as 3D arrays. They are limited by memory. `--jcube` sets the number of workers; by default it is the smallest of the number of chunks, `-j`, and the number of workers that fit in the available memory (detected, or set with `--memlimit`). The estimate is written to the log.

Each cube worker writes only its own core channels, and an island is extracted by the chunk containing its first channel, so results do not depend on the chunk size as long as `--overlap` is at least `minchans + specdilate` (for filtering) and longer than the spectral extent of the islands (for extraction; a warning is logged otherwise).

## Tools

Installing pony3d also installs these commands:

* `pony3d-makecube`: stack a sequence of single-channel images into a FITS cube. The cube is written one channel at a time, so memory use is one image whatever the cube size. `--compress gzip` (lossless) or `--compress rice` (quantised, see `--qlevel`) writes a tile-compressed cube, and needs `pip install pony3d[compress]`. Compression stages the channels in a temporary uncompressed file next to the output, so it needs free disk space equal to the uncompressed cube
* `pony3d-breakcube`: split a FITS cube into one image per channel, carrying across CASA per-channel beams
* `pony3d-contsub`: image-domain continuum subtraction for a cube or an image sequence
* `pony3d-movie`: render an MP4 of a channel sequence, optionally with mask contours; needs `pip install pony3d[tools]`

Run any of them with `-h` for their options. The scripts in `tools/` are wrappers that run the same commands from a clone without installing.

## Testing

`tests/run_test.py` makes a small synthetic image sequence with injected 3D sources, runs `pony3d` on it with several chunk sizes, and checks that the sources are recovered and that the results do not depend on chunking. It also round-trips the images through `pony3d-makecube` and `pony3d-breakcube`:

```
python tests/run_test.py
```
