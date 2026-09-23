import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import glob
import pickle
from collections import deque
import cv2
from skimage.filters import threshold_triangle

import utils


def threshold_image(image, mask, VARIANCE_THRESH=200, GRADIENT_THRESH=0.5):
    valid_vals = image[mask > 0]  # grab the valid pixel values inside the image - toss the rectified padding for the otsu calculation
    thresh_val, _ = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # apply otsu thresholding to get a threshold value
    _, binary = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)  # apply to full 2D image
    # plt.imshow(binary, cmap='gray')
    # plt.show()

    # define kernels
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    close_alongcrest = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5))

    # morphological cleanup
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cross, iterations=2) #erosion followed by dilation. Closes small holes outside the area of interest.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cross, iterations=1) #dilation followed by erosion. Closes small holes inside the area of interest.
    # plt.imshow(binary, cmap='gray')
    # plt.show()

    # gradient filtering - waves have large gradients on the edges
    image_blur = cv2.GaussianBlur(image, (3, 3), 0)
    grad_x = cv2.Sobel(image_blur, cv2.CV_32F, 1, 0, ksize=5) #2 and 9 are the original values
    grad_y = cv2.Sobel(image_blur, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    grad_scale = np.percentile(grad_mag, 99)

    grad_mag = np.clip(grad_mag, 0, grad_scale) #clip outliers

    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, grad_bin_tri = cv2.threshold(grad_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE) #find the best threshold for the edge gradient

    # contour level gradient filtering
    # labels connected regions in a binary image and returns statistics for each region. 8 defines pixel neighborhood
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    #num_labels = total number of blobs found
    #labels = an array of the same shape as binary where every pixel holds the integer ID of the blob it belongs to (background = 0)
    #stats = a table, one row per blob, with bounding boxinfo and pixel area

    grad_bin_bool = grad_bin_tri > 0 #turn the binary gradient image into a boolean mask to it can be combined with other boolean masks
    keep = np.zeros_like(binary)
    for label_id in range(1, num_labels):  # loop through all labels/blobs
        blob_mask = labels == label_id #isolates a single blob, pixels become true where that blob is and false everywhere else
        edge_pixels_in_blob = np.count_nonzero(grad_bin_bool & blob_mask) #finds the overlap between the blobs pixels and pixels with a strong gradient and counts them
        blob_area = stats[label_id, cv2.CC_STAT_AREA]  #returns the individual blobs total pixel count
        gradient_ok = (edge_pixels_in_blob / blob_area >= GRADIENT_THRESH)  # if at least X% of the blobs area coincides with a strong pixel gradient the entire blob is kept - set to 255. Otherwise continue

        # check the texture of the blob (broken or unbroken wave)
        blurred = cv2.GaussianBlur(image, (15, 15), 0) #blur the image
        highpass = image.astype(np.float32) - blurred.astype(np.float32) #highpass filter to take away slowly varying spatial gradients
        variance_ok = np.var(highpass[blob_mask]) >= VARIANCE_THRESH #take the variance of only the pixels inside the individual blob - threshold determined by inspection.
        # TODO: a true threshold can be determined by running all blobs on all frames and plotting a histogram - determining separation point

        if gradient_ok and variance_ok:
            keep[blob_mask] = 255

        '''VARIANCE THRESHOLD INSPECTION'''
        # cx, cy = centroids[label_id]  # blob centroid in (x, y) image coordinates
        # blob_overlay = np.ma.masked_where(~blob_mask, blob_mask)
        # fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        # axes.imshow(image, cmap='grey', zorder=0)
        # axes.imshow(blob_overlay, cmap='spring', zorder=1)
        # axes.annotate(
        #     f'blob {label_id}',
        #     xy=(cx + 30, cy - 30),  # point the arrow is pointing TO
        #     xytext=(cx + 150, cy - 150),  # where the text/arrow tail starts — offset so it doesn't sit on top of the blob
        #     arrowprops=dict(facecolor='cyan', edgecolor='cyan', width=2, headwidth=8, shrink=0.05),
        #     color='cyan',
        #     fontsize=10,
        #     zorder=3,
        # )
        # axes.invert_yaxis()
        # fig.suptitle(f'blob_{label_id}_variance: {np.var(highpass[blob_mask])}')
        # plt.savefig(f'/mnt/d/Wave_Breaking/figs/temp/{chunk_idx}_{frame_idx}_{label_id}_var.png')
        # plt.close('all')

    binary = keep #replace the old mask

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_alongcrest, iterations=2)  # clean up small holes

    # #SANITY PLOT
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    #
    # axes[0].imshow(image)
    # axes[0].set_title('Original')
    # axes[0].invert_yaxis()
    #
    # axes[1].imshow(binary, cmap='gray')
    # axes[1].set_title(f'modified binary')
    # axes[1].invert_yaxis()
    #
    # plt.tight_layout()
    # plt.show()
    # plt.close('all')

    return binary


def contour_area_dist(unfiltered_contours_dir):

    unfiltered_contours_paths = glob.glob(os.path.join(unfiltered_contours_dir, '*.pkl'))

    max_area = 1200  #area in pixels that almost all contours will not be greater than
    n_bins = 500

    bins = np.linspace(0, max_area, n_bins + 1)
    global_hist = np.zeros(n_bins, dtype=np.int64)  #global histogram for iteratively accumulating data

    for contour_path in unfiltered_contours_paths:  #loop through chunks

        contourList = utils.loadContours(contour_path)  #load the contours for each chunk

        for conts in contourList:  #load the contours for each frame
            for c in conts:  #for each contour
                area = cv2.contourArea(c)  #determine the area

                # Put areas above max_area into the final bin
                bin_index = min(
                    int(area / max_area * n_bins),
                    n_bins - 1
                )

                global_hist[bin_index] += 1  #add to the bin

    knee = threshold_triangle(global_hist) #find the knee point of the full histogram

    area_threshold = int(bins[knee])  #round the area threshold to the nearest integer

    plt.hist(global_hist, bins)
    # plt.xscale('log')
    plt.show()
    plt.close('all')

    return area_threshold


def generate_contours(data_dir, save_dir, sanity_plot_dir=None, VARIANCE_THRESH=200, GRADIENT_THRESH=0.5, plot_every=50):

    #load in rectified images per chunk
    n_chunks = len(glob.glob(os.path.join(data_dir, '*.dat')))

    for n in tqdm(range(n_chunks), colour='Yellow', leave=False):
        tqdm.write(f"Creating Contours for [Chunk {n}]...")
        chunk_index = n #go sequentially through chunks

        #load chunk of rectified images
        chunk, meta = utils.load_chunk(data_dir, chunk_index) #loads images already in range 0-255
        extent = (meta['x_grid_min'], meta['x_grid_max'], meta['y_grid_min'], meta['y_grid_max']) #extents of the image in world coordinates

        #for each image in the chunk draw contour
        N_frames = chunk.shape[0]

        mask = utils.get_mask(chunk[0]) #get a mask of the image - toss the black rectification padding

        chunk_contours = []  # collect this chunk's contours, one entry per frame
        for t in tqdm(range(N_frames), desc='Frames', colour='blue', leave=False):
            frame = chunk[t]

            binary = threshold_image(frame, mask, VARIANCE_THRESH, GRADIENT_THRESH) #create binary mask of the image

            conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find the contours - only the external contours.

            chunk_contours.append(conts)  # store this frame's contours

            if sanity_plot_dir is not None and t % plot_every == 0:
                frame_vis = frame.copy()
                if frame_vis.ndim == 2:
                    frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)  #convert the image from grayscale to bgr in order to have colored contours
                cv2.drawContours(frame_vis, conts, -1, (0, 0, 255), 2)

                title = f"chunk {chunk_index:04d} | frame {t} | n_contours={len(conts)}"

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB), extent=extent)
                ax.invert_yaxis()
                ax.set_title(title, fontsize=8)
                fig.savefig(os.path.join(
                    sanity_plot_dir, f'chunk{chunk_index:04d}_frame{t:04d}.png'
                ), dpi=150, bbox_inches='tight')
                plt.close(fig)

        # save all contours for this chunk in one pickle
        out_path = os.path.join(save_dir, f'Unfiltered_contours_chunk_{chunk_index:04d}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(chunk_contours, f)


def generate_contours_parallel(chunk_index, data_dir, save_dir, sanity_plot_dir=None, VARIANCE_THRESH=200, GRADIENT_THRESH=0.5, plot_every=50):

    # load chunk of rectified images
    chunk, meta = utils.load_chunk(data_dir, chunk_index)  # loads images already in range 0-255
    extent = (meta['x_grid_min'], meta['x_grid_max'], meta['y_grid_min'], meta['y_grid_max'])  # extents of the image in world coordinates

    # for each image in the chunk draw contour
    N_frames = chunk.shape[0]

    mask = utils.get_mask(chunk[0])  # get a mask of the image - toss the black rectification padding

    chunk_contours = []  # collect this chunk's contours, one entry per frame
    for t in tqdm(range(N_frames), desc=f'Chunk {chunk_index:04d}', colour='blue', position=chunk_index % os.cpu_count() + 1, leave=False):
        frame = chunk[t]

        binary = threshold_image(frame, mask, VARIANCE_THRESH, GRADIENT_THRESH)  # create binary mask of the image

        conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find the contours - only the external contours.

        chunk_contours.append(conts)  # store this frame's contours

        if sanity_plot_dir is not None and t % plot_every == 0:
            frame_vis = frame.copy()
            if frame_vis.ndim == 2:
                frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)  # convert the image from grayscale to bgr in order to have colored contours
            cv2.drawContours(frame_vis, conts, -1, (0, 0, 255), 2)

            title = f"chunk {chunk_index:04d} | frame {t} | n_contours={len(conts)}"

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB), extent=extent)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=8)
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


def filter_contours(data_dir, optical_flow_dir, save_dir, spacing=5, persistence_thresh=0.5, variance_thresh=0.2, plot=False, plot_interval=100, rectified_data_dir=None, plot_dir=None):

    #get all chunk paths
    chunk_paths = glob.glob(os.path.join(data_dir, '*.pkl'))

    n_chunks = len(chunk_paths)

    # distribution of contours areas - find knee point between noise/contamination and waves - min area size (in pixels) to filter contours out by
    # area_thresh = contour_area_dist(unfiltered_contours_dir)
    area_thresh = 10 #small area threshold - more permissive for small waves - filters out degenerate contours
    #area threshold based on the histogram of contour areas - ELIMINATES NOISE BUT MAY LOSE SOME SMALL WAVES, now different with clipped images

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

        # pull images for the corresponding frame
        images, meta = utils.load_chunk(rectified_data_dir, n)

        # get mask for buffer zone around the rectified image
        valid_mask = utils.get_mask(images[0])
        valid_image = np.where(valid_mask > 0, True, False)

        history = deque(maxlen=5)  #short rolling history of contours for validation of good pixels - motion compensated masks

        chunk_filtered_contours = [] #empty list to hold the filtered contours for each chunk
        for frame_idx in tqdm(range(n_frames), leave=False, colour='blue', desc='Frame'):  #loop through all frames in the chunk

            frame_contours = contours[frame_idx] #contours for a particular frame, should have shape (N_contours, N_points in contour, 1, 2)

            vx = np.memmap(vx_path, dtype='float32', mode='r', shape=shape)[frame_idx, :, :]  # optical flow velocities 'x' frames apart by the spacing (5 in the original case)
            vy = np.memmap(vy_path, dtype='float32', mode='r', shape=shape)[frame_idx, :, :]

            #de-trend the flow field - Subtract median flow vector to remove background translation
            vx_detrend = vx - np.median(vx[valid_image])  #TODO: fine for a first pass, may need a spatially varying detrend later if only certain patches are being removed
            vy_detrend = vy - np.median(vy[valid_image])  #mask the boarders of the rectified image so areas with no flow do not influence the median calculation

            #magnitude of the velocities
            magnitude = np.sqrt(vx_detrend**2 + vy_detrend**2)

            #direction of the flow_field in radians
            dir = np.arctan2(vy_detrend, vx_detrend)

            # #reconstruct the frames binary from the contours
            # frame_binary = np.zeros((h, w), dtype=np.uint8)
            # valid_raw = [c for c in frame_contours if c is not None and len(c) >= 5]  #define the valid contours, not empty and have a number of points greater than or equal to 5
            # if valid_raw:
            #     cv2.drawContours(frame_binary, valid_raw, -1, color=(255, 255, 255), thickness=-1) #fills in the contours reconstructing the 2D mask needed for filtering

            # #motion compensated pixel persistence filtering
            # dx = vx_detrend / spacing
            # dy = vy_detrend / spacing #per frame displacement from the coarse flow (5 frames) we are assuming that the flow is roughly linear here. because fps is high, this is likely to be true.

            # history.append((frame_binary, dx, dy))  # store this frames mask alongside the flow that generated it so that it can be advected in the future
            #
            # warped_history = deque(maxlen=5) #rolling history
            # for past_mask, past_dx, past_dy in history: #loop through all the past masks
            #     warped_history.append(utils.warp_mask(past_mask, past_dx, past_dy)) #rewarp every mask using the stored dx, dy such that it is aligned with the current frame
            # warped_history.append(frame_binary) #append to the deque, current frame does not need warping
            #
            # if len(warped_history) == 5: #once the deque of warped frames is full...
            #     stack = np.stack(warped_history, axis=0) #stack the aligned masks pixelwise
            #     persistent = (np.sum(stack > 0, axis=0) >= 3).astype(np.uint8) * 255 #determine that a pixel is persistent if 3/5 aligned frames are white (255), trying to filter spurious glint
            # else:
            #     persistent = frame_binary  # not enough history yet — don't reject anything on this basis

            count = 0  # count for number of rejected contours - mostly used for debugging, TODO: text file of all contours rejected?
            frame_filtered_contours = []
            for j in range(len(frame_contours)):
                if frame_contours[j] is None or len(frame_contours[j]) < 5:  #protect against empty contours or small contours that will cause issues later on
                    count += 1
                    continue
                #contour should have shape (N_points, 1, 2) - looping through a single contour now
                contour_mask = np.zeros((h, w), dtype='uint8') #initialize empty mask
                cv2.drawContours(contour_mask, frame_contours, j, color=(255, 0, 0), thickness=-1)  # fill contour area

                # #motion compensated persistance check
                # #reject a contour if most of its area does not overlap with pixels that were persistently bright - real advecting waves should track through this filter
                # persistent_pixels = np.count_nonzero(persistent[contour_mask > 0]) #count the number of pixels in the persistence mask that are greater than zero
                # contour_area_px = np.count_nonzero(contour_mask > 0)  #count the total number of pixels in the contour that are greater than zero
                # if contour_area_px == 0 or (persistent_pixels / contour_area_px) < persistence_thresh:    #if no pixels are 255 or the ratio is less than the threshold value reject the contour
                #     count += 1
                #     continue

                # mask values outside contours
                mag_c = magnitude[contour_mask > 0]
                dir_c = dir[contour_mask > 0]

                if mag_c.size == 0:  #protect against empty contours (invalid regions)
                    count += 1
                    continue

                # reject nearly static regions
                valid = mag_c >= np.percentile(mag_c, 50)  # only consider pixels with magnitude above 90th percentile within contour

                mag_valid = mag_c[valid]
                dir_valid = dir_c[valid]

                norm_mag_valid = mag_valid / np.max(mag_valid)  # normalize magnitudes to prevent large magnitudes from dominating circular variance calculation

                # filter by contour area - small contours are likely noise
                area = cv2.contourArea(frame_contours[j])
                if area < area_thresh:
                    count += 1
                    continue
                # print('contours_rejected:', count) #number of rejected contours thus far

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

            if plot and frame_idx % plot_interval == 0:
                #plot the filtered contours for each frame as a sanity check
                #load rectified chunk
                chunk, meta = utils.load_chunk(rectified_data_dir, n)
                extent = (meta['x_grid_min'], meta['x_grid_max'], meta['y_grid_min'], meta['y_grid_max']) #extents of the image in world coordinates
                image = chunk[frame_idx]  #get the frame from the chunk

                fig, axes = plt.subplots(1, 1, figsize=(10, 10))
                # axes.imshow(image, cmap='gray')
                plot_contours = [
                    c.reshape(-1, 1, 2).astype(np.int32)  #need to reshape the contour to (n_points, 1, 2) for contour drawing, otherwise a silent error will occur.
                    for c in frame_filtered_contours
                    if c is not None and len(c) >= 3
                ]
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                c = cv2.drawContours(image_rgb, plot_contours, -1, color=(255, 0, 0), thickness=2)
                axes.imshow(c, cmap='gray', extent=extent)
                axes.invert_yaxis()
                axes.set_title(f'Filtered Contours for Frame {frame_idx} in Chunk {n:04d}')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'Filtered_contours_chunk_{n:04d}_frame_{frame_idx:04d}.png'))
                # plt.show()
                plt.close('all')


            chunk_filtered_contours.append(frame_filtered_contours)
            # print(f'Rejected {count} contours')

        # save all contours for this chunk in one pickle
        out_path = os.path.join(save_dir, f'Filtered_contours_chunk_{n:04d}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(chunk_filtered_contours, f)


if __name__ == "__main__":

    rectified_data_dir = '/mnt/d/Wave_Breaking/rectified_images' #directory of saved rectified images

    save_dir = '/mnt/d/Wave_Breaking/unfiltered_contours' #directory to save contours in

    #some waves are so small that the turbulence (high variance) cannot be resolved by the camera. They get reclassified as non breaking and tossed.
    generate_contours(rectified_data_dir, save_dir, sanity_plot_dir='/mnt/d/Wave_Breaking/figs/unfiltered_contours', VARIANCE_THRESH=200, GRADIENT_THRESH=0.5, plot_every=25)

    unfiltered_contours_dir = '/mnt/d/Wave_Breaking/unfiltered_contours'
    filtered_contours_save_dir = '/mnt/d/Wave_Breaking/filtered_contours'
    optical_flow_dir = '/mnt/d/Wave_Breaking/optical_flow'

    plot_dir = '/mnt/d/Wave_Breaking/figs/filtered_contours' #directory to save plots of filtered contours

    filter_contours(unfiltered_contours_dir, optical_flow_dir, filtered_contours_save_dir, variance_thresh=0.2, plot=True, plot_interval=25, rectified_data_dir=rectified_data_dir, plot_dir=plot_dir)






