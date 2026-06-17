import sys
import detector

# Standard library imports
import os
import cv2
import sys
import glob
import time
from scipy.interpolate import interp2d

from tqdm import tqdm
import numpy as np
import scipy.signal as signal
from scipy import interpolate
import matplotlib.pyplot as plt
from skimage import img_as_float64
import matplotlib.patches as patches
from IPython.display import clear_output
from scipy.interpolate import griddata
from skimage import measure, img_as_ubyte
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pprint
import warnings;warnings.filterwarnings("ignore")


def opticalFlow(images, output_dir, paths=True):
    os.makedirs(output_dir, exist_ok=True)

    # Get image dimensions from first frame
    if paths:
        first_frame = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    else:
        first_frame = images[0]

    H, W = first_frame.shape
    n_frames = len(images) - 1  # optical flow is between pairs so n-1 frames

    # Initialize memmaps
    vx_path = os.path.join(output_dir, 'vx.dat')
    vy_path = os.path.join(output_dir, 'vy.dat')

    vx_mm = np.memmap(vx_path, dtype='float32', mode='w+', shape=(n_frames, H, W))
    vy_mm = np.memmap(vy_path, dtype='float32', mode='w+', shape=(n_frames, H, W))

    for n in tqdm(range(n_frames), desc="Computing optical flow"):

        if paths:
            frame1 = cv2.imread(images[n], cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread(images[n + 1], cv2.IMREAD_GRAYSCALE)
        else:
            frame1 = images[n]
            frame2 = images[n + 1]

        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Write directly to memmap at index n
        vx_mm[n] = flow[..., 0]
        vy_mm[n] = flow[..., 1]

        # Flush to disk periodically to avoid memory buildup
        if n % 100 == 0:
            vx_mm.flush()
            vy_mm.flush()

    # Final flush
    vx_mm.flush()
    vy_mm.flush()

    # Save shape metadata - required for reloading
    np.save(os.path.join(outputdir, 'flow_shape.npy'), np.array([n_frames, H, W]))

    return vx_mm, vy_mm


if __name__ == "__main__":

    #load in images
    akaawase_images = sorted(glob.glob('./elemental/background_removed_matlab_stbv2/' + '*.jpeg'))
    my_images = np.load('./drone/processed/1057subset_images.npy', mmap_mode='r')

    outputdir = './elemental/opticalFlow/'

    # opticalFlow(my_images, outputdir, False)

    #load optical flow output
    shape = tuple(np.load(os.path.join(outputdir, 'akaawase_flow_shape.npy')))
    vx_path = './elemental/opticalFlow/akaawase_vx.dat'
    vy_path = './elemental/opticalFlow/akaawase_vy.dat'
    vx = np.memmap(vx_path, dtype='float32', mode='r', shape=shape)
    vy = np.memmap(vy_path, dtype='float32', mode='r', shape=shape)

    #fps
    fps = 29.97
    dt = 1/fps

    #spatial resolution in meters
    drone = np.load('./drone/1057metadata.npy')
    dx = float(drone[-2][1])  # dx and dy are the same value
    gsd = dx  # ground sampling distance

    #load KM 2011 thresholds
    loaded_thresh = np.genfromtxt('./elemental/Akaawase_mthd_thresholds_from_pdf.csv', delimiter=',')
    # loaded_thresh = np.genfromtxt('./elemental/My_mthd_thresholds_from_pdf.csv', delimiter=',')
    orange = loaded_thresh[0] #peak curvature
    yellow = loaded_thresh[1] #end of positive curvature

    #akaawase
    frame_a = cv2.imread(akaawase_images[0], cv2.IMREAD_GRAYSCALE)
    frame_b = cv2.imread(akaawase_images[1], cv2.IMREAD_GRAYSCALE)

    #mine
    # frame_a = my_images[0] * 255
    # frame_b = my_images[1] * 255

    #calculate velocities in m/s
    cx = vx * gsd/dt
    cy = vy * gsd/dt

    #deal with nans
    cx = np.nan_to_num(cx, copy=True, nan=0.0, posinf=None, neginf=None)
    cy = np.nan_to_num(cy, copy=True, nan=0.0, posinf=None, neginf=None)

    #brightness threshold
    brightness_threshold = orange

    #image shape
    Ih, Iw = np.shape(frame_b)[0], np.shape(frame_b)[1]

    #visualize contours
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.imshow(frame_a)
    plt.contour(frame_a, levels=[brightness_threshold * 255], colors='k', linewidths=2)
    plt.contour(frame_b, levels=[brightness_threshold * 255], colors='m', linewidths=2)
    plt.xlim(0, Iw)
    plt.ylim(Ih, 0)
    plt.title('Contours')
    plt.xlabel('X ')
    plt.ylabel('Y')
    plt.show()
    plt.close('all')


    #TODO: consider only doing optical flow inside the contours to save time and disk space

