#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk


import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont


def render_png(fits_image_path, mask_fits_path, pixmin, pixmax, png_name, ccol, pixel_size=(2048, 2048), dpi=300):
    """
    Saves a FITS image (bitmap) with a Boolean mask overlay (contours)
    
    Parameters:
    fits_image_path (str): Path to the FITS image file.
    mask_fits_path (str): Path to the FITS file containing the Boolean mask.
    png_name (str): Path to save the output PNG file.
    pixel_size (tuple): Size of the output image in pixels (width, height).
    dpi (int): Dots per inch (default 100).
    
    The FITS image and the mask should have the same dimensions.
    """
    inch_size = (pixel_size[0] / dpi, pixel_size[1] / dpi) 
    with fits.open(fits_image_path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
        freq = hdr.get('CRVAL3')    
    if data.ndim == 3:
        data = data[0]
    if mask_fits_path != '':
        with fits.open(mask_fits_path) as hdul_mask:
            mask = hdul_mask[0].data    
        if mask.ndim == 3:
            mask = mask[0]   
        mask = mask.astype(bool)    
        if data.shape != mask.shape:
            raise ValueError("FITS image and mask dimensions do not match")    
    fig, ax = plt.subplots(figsize=inch_size)    
    fig.patch.set_facecolor('black')  
    ax.set_facecolor('black') 
    if pixmin == 0: pixmin = np.nanmin(data)
    if pixmax == 0: pixmax = np.nanmin(data)
    ax.imshow(data, cmap='gray', origin='lower', vmin=pixmin, vmax=pixmax)
    if mask_fits_path != '':
        ax.contour(mask, colors=ccol, linewidths=0.3, levels=[0.5])    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)    
    plt.savefig(png_name, bbox_inches='tight', pad_inches=0, dpi=dpi, facecolor='black')  # facecolor sets the PNG background
    plt.close(fig) 
    return freq


def annotate_png(png_name, fits_name, freq, i, nframes, nolabel, fontpath, fontsize):
    """
    Annotate PNG image with relevant info.

    Parameters:
    png_name (str): Path to the input PNG image
    fits_name (str): Name of FITS image for the labels
    freq (float): Frequency in Hz (will be scaled to MHz)
    i (int): Frame number
    nframes (int): Total number of frames 
    output_name (str): Path to save the annotated image. If None, overwrites the original image.
    """

    if fontpath != '':
        labelfont = ImageFont.truetype(fontpath, fontsize)
    else:
        labelfont = None

    freq_mhz = freq / 1e6
    z_hi = (1420.40575 / freq_mhz) - 1.0
    z_oh = (1667.0 / freq_mhz) - 1.0

    img = Image.open(png_name)
    xx, yy = img.size
    if not nolabel:
        draw = ImageDraw.Draw(img)
        draw.text((0.03 * xx, 0.87 * yy), f'Frame : {str(i).zfill(len(str(nframes)))} / {nframes}', fill='white', font=labelfont)
        draw.text((0.03 * xx, 0.90 * yy), f'Freq  : {round(freq_mhz, 4)} MHz', fill='white', font=labelfont)
        draw.text((0.03 * xx, 0.93 * yy), f'z     : {round(z_hi, 6)} (HI) | {round(z_oh, 6)} (OH)', fill='white', font=labelfont)
        draw.text((0.03 * xx, 0.96 * yy), f'Image : {fits_name}', fill='white', font=labelfont)

    img.save(png_name) 


def process_frame(i, datafits, maskfits, output_png, nframes, pixmin, pixmax, ccol, fontpath, fontsize, nolabel):
    print(f'Processing: {datafits} {maskfits}')
    freq = render_png(datafits, maskfits, pixmin, pixmax, output_png, ccol, pixel_size=(2048, 2048), dpi=300)
    annotate_png(output_png, datafits, freq, i, nframes, nolabel, fontpath, fontsize,)



def main():

    parser = argparse.ArgumentParser(description='Render MP4 movies from a sequence of FITS images')

    parser.add_argument('--imagepath', type=str, default='', metavar='', help='Path to folder containing data images (required)')
    parser.add_argument('--maskpath', type=str, default='', metavar='', help='Path to folder containing mask images (for contours; optional)')
    parser.add_argument('--outfile', type=str, default='', metavar='', help='Name of output mp4 video file (required)')
    parser.add_argument('--pixmin', type=float, default=-0.005, metavar='', help='Minimum pixel value for rendering images in map units')
    parser.add_argument('--pixmax', type=float, default=0.008, metavar='', help='Maximum pixel value for rendering images in map units')
    parser.add_argument('--ccol', type=str, default='cyan', metavar='', help='Contour colour (default = cyan)')
    parser.add_argument('--fontpath', type=str, default='', metavar='', help='Font (default = /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf)')
    parser.add_argument('--fontsize', type=int, default='18', metavar='', help='Font size (default = 18)')
    parser.add_argument('--nolabel', default=False, action='store_true', help='Do not add annotations')
    parser.add_argument('--framesize', type=str, default=2048, metavar='', help='Movie size (square aspect only; default = 2048)')
    parser.add_argument('--fps', type=int, default=10, metavar='', help='Frames per second (default = 10)')
    parser.add_argument('--renderer', type=str, default='opencv', help='Select either "opencv" or "ffmpeg" to render movie, latter must be in path (default = opencv)')
    parser.add_argument('--j', '-j', type=int, default=32, metavar='', help='Number of parallel processes to use when rendering images')

    args = parser.parse_args()

    imagepath = args.imagepath
    maskpath = args.maskpath
    outfile = args.outfile
    pixmin = args.pixmin
    pixmax = args.pixmax
    ccol = args.ccol
    fontpath = args.fontpath
    fontsize = args.fontsize
    nolabel = args.nolabel
    framesize = args.fps
    fps = args.fps
    renderer = args.renderer
    j = args.j


    f_list = sorted(glob.glob(imagepath+'/*.fits'))[1:-1]
    m_list = sorted(glob.glob(maskpath+'/*.fits'))
    if len(m_list) == 0:
        plotmasks = False
    else:
        plotmasks = True

    nfits = len(f_list)
    if nfits != len(m_list):
        raise ValueError("Data FITS list and mask FITS list must be of the same length!")

    opdir = outfile+'_frames'
    if not os.path.isdir(opdir):
        os.mkdir(opdir)

    if fontpath == '':
        print('No font specified, using Pillow default, which may be unreadable.')

    tasks = []
    for i in range(nfits):
        datafits = f_list[i]
        if plotmasks:
            maskfits = m_list[i]
        else:
            maskfits = ''
        output_png = maskfits.split('/')[-1].replace('.fits','_'+str(i).zfill(5)+'.png')
        output_png = os.path.join(opdir,output_png)
        tasks.append((i, datafits, maskfits, output_png, nfits, pixmin, pixmax, ccol, fontpath, fontsize, nolabel))

    pool = Pool(processes=j)
    pool.starmap(process_frame, tasks)


    image_files = sorted(glob.glob(f'{opdir}/*png'))
    print(f'Using {renderer} to write {outfile}')
    if renderer == 'opencv':
        frame = cv2.imread(image_files[0])
        height = framesize
        width = framesize
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
        for image_file in image_files:
            frame = cv2.imread(image_file)
            video_writer.write(frame)
        video_writer.release()
    elif renderer == 'ffmpeg':
        frame = f'{framesize}x{framesize}'
        ff_images = maskfits.split('/')[-1].replace('.fits','_%05d.png')
        ff_images = os.path.join(opdir,ff_images)
        os.system('ffmpeg -r '+str(fps)+' -f image2 -s '+frame+' -i pic_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p '+outfile)


if __name__ == '__main__':

    main()


