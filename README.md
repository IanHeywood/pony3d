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

Note that `input_image(s)` is a pattern. Specifying `my_images` will return a naturally-sorted list of all files in the current directory that match `*my_images*`. 

```
Usage: pony3d [options] input_image(s)

Options:
  -h, --help            show this help message and exit
  --threshold=T         Sigma threshold for masking (default = 5.0)
  --boxsize=B           Box size to use for background noise estimation
                        (default = 80)
  --dilate=D            Number of iterations of binary dilation in the spatial
                        dimensions (default = 5, set to 0 to disable)
  --specdilate=S        Number of iterations of binary dilation in the
                        spectral dimension (default = 2, set to 0 to disable,
                        filtering must be enabled)
  --chanaverage=N       Width sliding channel window to use when making masks
                        (odd numbers preferred, default = 1, i.e. no
                        averaging)
  --saveaverage         Save the result of the sliding average (default = do
                        not save averaged image)
  --chanchunk=M         Number of channels to load per worker when filtering
                        single channel instances (default = 16)
  --nofilter            Do not filter detections for single channel instances
                        (default = filtering enabled)
  --nocount             Do not report island counts and input RMS in the log
                        (default = report values)
  --savenoise           Enable to export noise images as FITS files (default =
                        do not save noise images)
  --invert              Multiply images by -1 prior to masking (default = do
                        not invert images)
  --opdir=OPDIR         Name of folder for output products (default = auto
                        generated)
  --masktag=MASKTAG     Suffix and subfolder name for mask images (default =
                        mask)
  --noisetag=NOISETAG   Suffix and subfolder name for noise images (default =
                        noise)
  --averagetag=AVERAGETAG
                        Suffix and subfolder name for boxcar averaged images
                        (default = avg)
  --filtertag=FILTERTAG
                        Suffix and subfolder name for filtered images (default
                        = filtered)
  -f, --force           Overwrite existing FITS outputs (default = do not
                        overwrite)
  -j J                  Number of worker processes (default = 24)
  ```