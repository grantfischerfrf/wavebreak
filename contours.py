import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import glob
import pickle
from collections import deque
import cv2

import utils


def threshold_image(image, mask):
    valid_vals = image[mask > 0]  # grab the valid pixel values inside the image - toss the rectified padding for the otsu calculation
    thresh_val, _ = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # apply otsu thresholding to get a threshold value
    _, binary = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)  # apply to full 2D image
    # plt.imshow(binary, cmap='gray')
    # plt.show()

    # define kernels
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # morphological cleanup
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cross, iterations=2) #originally ellipse kernel here

    return binary


def contour_area_dist(contourList, sigma=7):

    # plot distribution of contours
    cont_areas = [cv2.contourArea(c) for conts in contourList for c in conts]

    # use scotts rule to determine number of bins the histogram should have
    n = len(cont_areas)
    cont_areas_arr = np.array(cont_areas)
    bin_width_scott = 3.5 * np.std(cont_areas_arr) / n ** (1 / 3)
    n_bins_scott = int(np.ceil((cont_areas_arr.max() - cont_areas_arr.min()) / bin_width_scott))

    # calculate histogram
    hist, bins = np.histogram(cont_areas, bins=n_bins_scott, range=(0, 256))
    knee, _ = utils.find_knee(hist, sigma=sigma)
    # plt.hist(hist, bins)
    # plt.xscale('log')
    # plt.show()
    # plt.close('all')

    return knee


def generate_contours(data_dir, save_dir, sanity_plot_dir=None, plot_every=50):

    #load in rectified images per chunk
    n_chunks = len(glob.glob(os.path.join(data_dir, '*.dat')))

    for n in tqdm(range(n_chunks), colour='Yellow', leave=False):
        tqdm.write(f"Creating Contours for [Chunk {n}]...")
        chunk_index = n #go sequentially through chunks

        #load chunk of rectified images
        chunk, meta = utils.load_chunk(data_dir, chunk_index) #loads images already in range 0-255

        #for each image in the chunk draw contour
        N_frames = chunk.shape[0]

        mask = utils.get_mask(chunk[0]) #get a mask of the image - toss the black rectification padding

        history = deque(maxlen=5) #short rolling history of contours for validation of good pixels

        chunk_contours = []  # collect this chunk's contours, one entry per frame
        for t in tqdm(range(N_frames), desc='Frames:', colour='blue', leave=False):
            frame = chunk[t]

            binary = threshold_image(frame, mask) #create binary mask of the image - Add mask here

            history.append(binary) #append the binary mask to the deque

            if len(history) == 5:  #if there is 5 binary masks in the deque ...
                stack = np.stack(history, axis=0)  #stack them pixelwise
                persistent = (np.sum(stack > 0, axis=0) >= 3).astype(  #creates a boolean mask, 255 = True, 0=False. if a pixel is True 3/5 times it is set to 255 and kept. else it is set to False and 0
                    np.uint8) * 255  #
            else:
                persistent = binary

            conts, _ = cv2.findContours(persistent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find the contours - only the external contours.

            chunk_contours.append(conts)  # store this frame's contours

            if sanity_plot_dir is not None and t % plot_every == 0:
                frame_vis = frame.copy()
                if frame_vis.ndim == 2:
                    frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(frame_vis, conts, -1, (0, 0, 255), 2)

                title = f"chunk {chunk_index:04d} | frame {t} | n_contours={len(conts)}"

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB))
                ax.set_title(title, fontsize=8)
                ax.axis('off')
                fig.savefig(os.path.join(
                    sanity_plot_dir, f'chunk{chunk_index:04d}_frame{t:04d}.png'
                ), dpi=150, bbox_inches='tight')
                plt.close(fig)

        # save all contours for this chunk in one pickle
        out_path = os.path.join(save_dir, f'Unfiltered_contours_chunk_{chunk_index:04d}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(chunk_contours, f)


def smooth_contour(contour, window_size=5):

    contour = contour.squeeze()  # (N, 1, 2) → (N, 2)  #if the input is cv2 generated contours

    # Smoothing function using Hann window
    hann_window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))
    hann_window /= hann_window.sum()  # Normalize the Hann window

    # Separate x and y coordinates from the contour
    x_vals = contour[:, 0]
    y_vals = contour[:, 1]

    # Pad the contour data to apply the moving window (to handle the edges)
    x_padded = np.pad(x_vals, (window_size // 2,), mode='wrap')
    y_padded = np.pad(y_vals, (window_size // 2,), mode='wrap')

    # Apply the Hann window using convolution
    x_smoothed = np.convolve(x_padded, hann_window, mode='valid')
    y_smoothed = np.convolve(y_padded, hann_window, mode='valid')

    # Combine the smoothed x and y values
    smoothed_contour = np.vstack((x_smoothed, y_smoothed)).T

    return smoothed_contour

def filter_contours(data_dir, optical_flow_dir, save_dir, spacing=5, variance_thresh=0.2):

    #get all chunk paths
    chunk_paths = glob.glob(os.path.join(data_dir, '*.pkl'))

    n_chunks = len(chunk_paths)

    # distribution of contours areas - find knee point between noise/contamination and waves - min area size (in pixels) to filter contours out by
    unfiltered_contours = utils.loadContours(glob.glob(os.path.join(unfiltered_contours_dir, '*.pkl'))[0]) #load first file
    area_thresh = contour_area_dist(unfiltered_contours, sigma=7)
    #sigma of 5 or 7 normally good - area threshold based on the histogram of contour areas - ELIMINATES NOISE BUT MAY LOSE SOME SMALL WAVES, now different with clipped images

    #loop through all chunks and filter the contours:
    for n in tqdm(range(n_chunks), colour='yellow', desc='Chunk'):
        tqdm.write(f'Filtering Chunk {n:04d}')

        #load contours
        contours = utils.loadContours(chunk_paths[n])  #should have shape (N_frames, N_contours, N_points in contour, 1, 2)

        # get optical flow output here match the chunk, optical flow files should be of shape (n_frames, h, w)
        vx_path = os.path.join(optical_flow_dir, f'vx_{n:04d}.dat') #path to optical flow x velocities
        vy_path = os.path.join(optical_flow_dir, f'vy_{n:04d}.dat') #path to y velocities
        shape = tuple(np.load(os.path.join(optical_flow_dir, f'flow_shape_{n:04d}.npy')))
        n_frames, h, w = shape  #shape of the memmaps - n_frames should no longer have that buffer because it was passed through optical flow #
        # - the last 5 frames in the contours variable are not needed

        chunk_filtered_contours = [] #empty list to hold the filtered contours for each chunk
        for frame_idx in tqdm(range(n_frames), leave=False, colour='blue', desc='Frame'):  #loop through all frames in the chunk

            frame_contours = contours[frame_idx] #contours for a particular frame, should have shape (N_contours, N_points in contour, 1, 2)

            vx = np.memmap(vx_path, dtype='float32', mode='r', shape=shape)[frame_idx, :, :]  # optical flow velocities 'x' frames apart by the spacing (5 in the original case)
            vy = np.memmap(vy_path, dtype='float32', mode='r', shape=shape)[frame_idx, :, :]

            #de-trend the flow field - Subtract median flow vector to remove background translation
            vx_detrend = vx - np.median(vx)
            vy_detrend = vy - np.median(vy)

            #magnitude of the velocities
            magnitude = np.sqrt(vx_detrend**2 + vy_detrend**2)

            #direction of the flow_field in radians
            dir = np.arctan2(vy_detrend, vx_detrend)

            count = 0  # count for number of rejected contours - mostly used for debugging
            frame_filtered_contours = []
            for j in range(len(frame_contours)):
                #contour should have shape (N_points, 1, 2) - looping through a single contour now
                contour_mask = np.zeros((h, w), dtype='uint8') #initialize empty mask
                cv2.drawContours(contour_mask, frame_contours, j, color=(255, 0, 0), thickness=-1)  # fill contour area

                # mask values outside contours
                mag_c = magnitude[contour_mask > 0]
                dir_c = dir[contour_mask > 0]

                # reject nearly static regions
                valid = mag_c >= np.percentile(mag_c, 90)  # only consider pixels with magnitude above 90th percentile within contour

                mag_valid = mag_c[valid]
                dir_valid = dir_c[valid]

                norm_mag_valid = mag_valid / np.max(mag_valid)  # normalize magnitudes to prevent large magnitudes from dominating circular variance calculation

                # filter by contour area - small contours are likely noise
                area = cv2.contourArea(frame_contours[j])
                if area < area_thresh:
                    count += 1
                    continue

                # smooth the contour using a hanning window
                smoothed_contour = smooth_contour(frame_contours[j].squeeze())

                # Use magnitude weighted circular variance to determine if contour has coherent motion - works better at rejecting large turbulent foam patches
                R = np.abs(np.sum(norm_mag_valid * np.exp(1j * dir_valid)) / np.sum(norm_mag_valid))
                # e^i*theta creates a unit vector on a circle in direction of theta. weighted by magnitude. Summed over all pixels in contour. Divided by sum of magnitudes to normalize

                V = 1 - R  # circular variance
                if V <= variance_thresh:  # threshold for variance. 0 = all directions the same, 1 = uniform distribution - widely spread
                    frame_filtered_contours.append(smoothed_contour)
                else:
                    count += 1

            chunk_filtered_contours.append(frame_filtered_contours)
            # print(f'Rejected {count} contours')

        # save all contours for this chunk in one pickle
        out_path = os.path.join(save_dir, f'Filtered_contours_chunk_{n:04d}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(chunk_filtered_contours, f)


if __name__ == "__main__":

    rectified_data_dir = '/mnt/d/Wave_Breaking/rectified_images' #directory of saved rectified images

    save_dir = '/mnt/d/Wave_Breaking/unfiltered_contours' #directory to save contours in

    # generate_contours(rectified_data_dir, save_dir, sanity_plot_dir='/mnt/d/Wave_Breaking/figs', plot_every=50)

    unfiltered_contours_dir = '/mnt/d/Wave_Breaking/unfiltered_contours'
    filtered_contours_save_dir = '/mnt/d/Wave_Breaking/filtered_contours'
    optical_flow_dir = '/mnt/d/Wave_Breaking/optical_flow'

    # filter_contours(unfiltered_contours_dir, optical_flow_dir, filtered_contours_save_dir)


#TODO: maybe run everything from scratch

#TODO: fix the addition of the waves that








