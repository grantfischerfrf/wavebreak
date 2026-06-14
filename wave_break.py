import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.signal import savgol_filter
import re
import torch
from resnet import UNet
from tqdm import tqdm
from collections import deque
import pickle


def numerical_sort(string):

    parts = re.split(r'(\d+)', string)
    return [int(part) if part.isdigit() else part for part in parts]


def processVideo(file_path, save_path, decimate=False, new_fps=30):

    capture = cv2.VideoCapture(file_path)

    original_fps = capture.get(cv2.CAP_PROP_FPS)  # get original fps of video
    print(f'FPS: {original_fps}')

    out_name = os.path.basename(file_path).split('.')[0] #create output name for frames
    # create output directory if it doesn't exist
    if not os.path.exists(save_path + out_name):
        os.makedirs(save_path + out_name)

    print('Processing ...')

    frame_count = 0

    start_frame = 2297 #start of waves in video
    end_frame = 2597 #end of clip
    while capture.isOpened():

        ret, frame = capture.read()

        if not start_frame <= frame_count <= end_frame:
            frame_count += 1
            continue

        if not ret:
            print("Can't recieve frame. Exiting...")
            break  #automatically breaks if no more frames to read

        if decimate:
            assert new_fps <= original_fps, "New fps must be less than or equal to original fps"

            frame_interval = int(original_fps/new_fps)  #define frame interval to create new fps

            if frame_count % frame_interval == 0:  #if the frame count is a multiple of the frame interval, save the frame

                cv2.imwrite(f'{save_path}{out_name}/{out_name}' + '_' + str(frame_count) + '.jpeg', frame)  # save frame as JPEG file

        else:
            cv2.imwrite(f'{save_path}{out_name}/{out_name}' + '_' + str(frame_count) + '.jpeg', frame)  # save all frames as JPEG files

        frame_count +=1

    capture.release()
    print('Done')

    return original_fps


def correctImages(images):   #FIXME: need to combine this with consolodate images
    #normalize images based on background lighting - Kleiss and Melville 2011
    #use kleiss 2009 I' = (I/S(x, y)) * Sbar

    # FIXME: dividing by the median background lighting may work better to get rid of effects from bright white boat. Subtracting the pixel intensity of the boat to isolate the background may also work.
    S = np.mean([cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in images], axis=0) / 255 #normalized mean background lighting, averaged over time dimension
    Sbar = np.mean(S, axis=(0, 1))#average over x and y dimensions

    # grey = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    # plt.hist(grey.ravel(), bins=256, density=True)
    # plt.show()

    #white cap coverage from Kleiss and Melville 2011
    for image in tqdm(images):

        #normalize image
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img_norm_01 = img/255  #normalize to 0-1 scale

        #normalize through pixelwise division by mean background lighting. Paper finds this better than subtraction
        img_norm = (img_norm_01 / S) * Sbar

        # plt.hist(img_norm.ravel(), bins=256, density=True)
        # plt.show()
        # plt.close('all')

        np.save('./drone/processed/subset20171791057_corrected/' + os.path.basename(image).split('.')[0] + '_corrected.npy', img_norm) #TODO: This must get a save path - its hardcoded now
        #values greater than 1 should be targeted as white caps/breaking waves - bc dividing by median background lighting


def clip_images(images, x_min, x_max, y_min, y_max):
    clipped = images[:, y_min:y_max, x_min:x_max]
    return clipped


def consolidate_images(path, savepath):

    #save all images as a file to allow memory mapping
    files = sorted(glob.glob(path + '/*.npy'), key=numerical_sort)

    # Load one file to get shape/dtype
    sample = np.load(files[0])
    shape = (len(files),) + sample.shape
    dtype = sample.dtype

    # Create output file on disk - not RAM
    print('saving to disk...')
    stack = np.lib.format.open_memmap(
        f'{savepath}',
        mode='w+',
        dtype=dtype,
        shape=shape
    )

    # Fill it incrementally
    for i, f in enumerate(tqdm(files)):
        stack[i] = np.load(f)  # loads one file at a time

    # Flush
    stack.flush()
    del stack

    #this will take a while to write
    print('saved')


def contour(image, rgb_images, count):

    thresh_val, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(thresh_val)
    # plt.imshow(binary)
    # plt.show()
    # plt.close('all')

    # define kernels
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # morphological cleanup
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cross, iterations=2) #originally ellipse kernel here
    #FIXME: this is still giving false positives on brightly illuminated waves - YOU NEED TO CHECK IF FUTURE PROCESSING TOSSES THESE

    #PLOTTING
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    #
    # axes[0].imshow(cv2.imread(rgb_images))
    # axes[0].set_title('Original')
    #
    # axes[1].imshow(binary, cmap='gray')
    # axes[1].set_title(f'modified binary')

    # plt.tight_layout()
    # plt.show()
    # plt.close('all')

    # # gradient filtering - waves have large gradients on the edges
    # grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    # grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    #
    # # normalize for stability
    # grad_mag = grad_mag / (np.max(grad_mag) + 1e-8)
    #
    # # combine with Otsu
    # binary = np.where(grad_mag > 0.15, binary, 0).astype(np.uint8)
    #
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2) #clean up small holes

    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    #
    # axes[0].imshow(cv2.imread(rgb_images))
    # axes[0].set_title('Original')
    #
    # axes[1].imshow(binary, cmap='gray')
    # axes[1].set_title(f'modified binary')
    #
    # plt.tight_layout()
    # plt.show()
    # plt.close('all')

    # conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # #print plot of contours
    # # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # c = cv2.drawContours(image, conts, -1, color=(255, 0, 0), thickness=1)  # draw contours
    # plt.imshow(c, alpha=0.5)
    # # plt.savefig(f'./drone/outputs/all_contours_image_sample.png', dpi=300)
    # plt.show()
    # plt.close('all')

    # save the contours
    # with open(f'./drone/processed/subset20171791057_contours/1057contours_image_{count}.pkl', 'wb') as f:  #TODO: make the save path a function argument
    #     pickle.dump(conts, f)

    return binary



def loadContours(path):
    with open(path, 'rb') as f:
        contour_list = pickle.load(f)
    return contour_list


def find_knee(hist, sigma=5):
    # find where histogram drops off from the spike
    threshold = 0.01 * hist.max()
    t_min = np.argmax(hist < threshold)

    t_values = range(t_min, len(hist) - sigma)
    ratios = []

    for t in t_values:
        left = hist[
               t - sigma:t]  # checks the variance from bins left and right from a point. At the knee point, the ratio in left and right variance spikes
        right = hist[t:t + sigma]

        if len(left) > 1 and len(right) > 1:
            var_left = np.var(left)
            var_right = np.var(right)

            if var_right > 0:
                ratios.append(var_left / var_right)
            else:
                ratios.append(0)  # guard against divide by 0
        else:
            ratios.append(0)

    T = t_values[np.argmax(ratios)]
    return T, ratios


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
    knee, _ = find_knee(hist, sigma=sigma)
    # plt.hist(hist, bins)
    # plt.xscale('log')
    # plt.show()
    # plt.close('all')

    return knee


def filter_contours(image, image_next, rgb_image, contours, image_count, variance_thresh, area_thresh):
    #calculate optical flow between consecutive images using the KM 2011 method
    #get rid of noise using the farneback method

    h, w = image.shape[0], image.shape[1]

    #optical flow on images
    flow = cv2.calcOpticalFlowFarneback((image*255).astype(np.uint8), (image_next*255).astype(np.uint8), None,0.5, 3, 15, 3, 5, 1.2, 0) #convert 0-1 to 0-255 for optical flow, previous image - next image
    u = flow[..., 0]  # selects all previous dimensions and the 0th index of the last dimension  #FIXME:  move this to the GPU as well speed it up
    v = flow[..., 1] #gives dense optical flow. Pixel displacement per frame. Not an actual speed

    #demean flow field
    # Subtract median flow vector to remove background translation
    # (drone drift, tidal advection, front convergence)
    u = u - np.median(u)
    v = v - np.median(v)

    # #visualize optical flow results
    # step = 10  # downsample for clarity
    #
    # y, x = np.mgrid[0:h:step, 0:w:step]
    # u_s = u[::step, ::step]
    # v_s = v[::step, ::step]
    #
    # plt.figure(figsize=(10, 6))
    # plt.imshow(image, cmap='gray')
    # plt.quiver(x, y, u_s, v_s, color='r', angles='xy', scale_units='xy', scale=1)
    #
    # plt.title('Optical Flow (Farneback)')
    # plt.axis('off')
    # plt.show()

    magnitude = np.sqrt(u**2 + v**2).astype(np.float32)

    dir = np.arctan2(v, u).astype(np.float32) #direction of flow in radians

    filtered_contours = []
    #use directional variance of each contour to filter out sunglint
    global_magnitude_thresh = np.percentile(magnitude, 10) #pixels only excluded if they are low magnitude relative to entire image.

    count=0  #count for number of rejected contours
    for j in range(len(contours)):  #loop through individual contours
        contour_mask = np.zeros((h, w), dtype=np.uint8)  # initialize empty mask
        c = cv2.drawContours(contour_mask, contours, j, color=(255, 0, 0), thickness=-1)  # fill contour area

        #mask values outside contours
        mag_c = magnitude[contour_mask > 0]
        dir_c = dir[contour_mask > 0]

        #reject nearly static regions
        valid = mag_c >= np.percentile(mag_c, 90)  #only consider pixels with magnitude above 50th percentile within contour #FIXME: this probably needs to be cranked up to 90 ish - can also find a knee point in a histogram distribution here
        # valid = mag_c >= global_magnitude_thresh  #only consider pixels with magnitude above 10th percentile of entire image
        mag_valid = mag_c[valid]
        dir_valid = dir_c[valid]

        # if len(dir_valid) < 10:  # not enough pixels to estimate variance reliably
        #     filtered_contours.append(conts[j])  # keep it — benefit of the doubt
        #     # count += 1 #reject the contour
        #     continue

        norm_mag_valid = mag_valid / np.max(mag_valid)  #normalize magnitudes to prevent large magnitudes from dominating circular variance calculation
        # norm_mag_valid = mag_valid / np.percentile(mag_valid, 95)  # robust to single outlier
        # norm_mag_valid = np.clip(norm_mag_valid, 0, 1)

        #filter by contour area - small contours are likely noise
        area = cv2.contourArea(contours[j])
        if area < area_thresh:
            count += 1
            continue

        #Use magnitude weighted circular variance to determine if contour has coherent motion - works better at rejecting large turbulent foam patches
        # R = np.abs(np.sum(mag_c * np.exp(1j * dir_c)) / np.sum(mag_c)) # e^i*theta creates a unit vector on a circle in direction of theta. weighted by magnitude. Summed over all pixels in contour. Divided by sum of magnitudes to normalize
        R = np.abs(np.sum(norm_mag_valid * np.exp(1j * dir_valid)) / np.sum(norm_mag_valid))

        V = 1 - R  #circular variance

        if V <= variance_thresh:  #threshold for variance. 0 = all directions the same, 1 = uniform distribution - widely spread
            filtered_contours.append(contours[j])

        else:
            count += 1

        # print(f'Rejected {count} contours')

    background = cv2.cvtColor(cv2.imread(rgb_image), cv2.COLOR_BGR2RGB)
    y_min, y_max = 700, 1500
    clipped_background = background[y_min:y_max, :, :]

    c = cv2.drawContours(clipped_background, filtered_contours, -1, color=(255, 255, 0), thickness=2)  # draw filtered contours
    plt.axis('off')
    plt.imshow(c)
    plt.tight_layout()
    plt.savefig('./drone/outputs/temp/subset/filtered_contours_' + str(image_count) + '.png', dpi=300)
    # plt.show()
    plt.close('all')

    with open(f'./drone/processed/subset20171791057_contours_filtered/1057contours_image_{image_count}.pkl', 'wb') as f:
        pickle.dump(filtered_contours, f)


def fit_ellipse(rgb_images, N_min, contours):

    count = 0
    print('Fitting Ellipses ...')
    for f in tqdm(range(len(rgb_images) - N_min)):

        frame = cv2.imread(rgb_images[f], cv2.COLOR_BGR2RGB)

        y_min, y_max = 700, 1500
        clipped_frame = frame[y_min:y_max, :, :]  #TODO: find a better way to do this - this is dumb

        # fit ellipses to the Convex Hulls
        ellipses = [cv2.fitEllipse(c) for c in contours[f] if len(c) >= 5] #min points to fit an ellipse

        # Draw contours
        frame = cv2.drawContours(clipped_frame, contours[f], -1, color=(255, 255, 0), thickness=2)

        for ellipse in ellipses:
            if ellipse is not None:
                frame = cv2.ellipse(clipped_frame, ellipse, (0, 0, 255), 2)

        # Plot
        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        axes.imshow(clipped_frame)
        axes.set_title("ConvexHull + Ellipses")
        axes.axis("off")
        x_max = 3840
        # axes.set_xlim(3000, x_max)
        # axes.set_ylim(1500, 700)

        plt.tight_layout()
        plt.savefig(f'./drone/outputs/temp/subset_ellipses/ellipses_{count}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close('all')

        count += 1



if __name__ == "__main__":

    #rgb images - non modified
    rgb_images = sorted(glob.glob('./drone/processed/subset20171791057/*.jpeg'), key=numerical_sort)[:282]

    #FIXME: finally fix the correct and consolodate images functions. Then add the clip images before to speed up processing time. Less pixels in the array to correct for

    #load corrected images - image processing pipeline from kleiss and melville 2011
    images = np.load('./drone/processed/1057subset_images.npy', mmap_mode='r')[:282] #already opened images using cv2.imread - from consolidate images function FIXE: just clip images from here instead

    x_min, x_max = 0, images[0].shape[1]
    y_min, y_max = 700, 1500
    clipped_images = clip_images(images, x_min, x_max, y_min, y_max)

    '''Loop through each corrected image and apply processing pipeline - detecting active wave breaking in images.'''
    print('Creating Contours...')

    history = deque(maxlen=5)  # TODO: play with deque length - if needed

    count = 0
    for frame_idx in tqdm(range(len(clipped_images))):  #input cropped images here - was originally images
        image = (clipped_images[frame_idx] * 255).astype(np.uint8) #change datatype for open CV operations

        #create contours
        binary = contour(image, rgb_images[frame_idx], count)

        history.append(binary)

        if len(history) == 5:
            stack = np.stack(history, axis=0)
            persistent = (np.sum(stack > 0, axis=0) >= 3).astype(np.uint8) * 255  #require 3/5 frames to be an active whitecap to be considered a good pixel - otherwise discard
        else:
            persistent = binary

        conts, _ = cv2.findContours(persistent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with open(f'./drone/processed/subset20171791057_contours/1057contours_image_{count}.pkl', 'wb') as f:
            pickle.dump(conts, f)

        count += 1

    '''APPLY REST OF PROCESSING PIPELINE'''
    #load saved contours
    conts = [loadContours(file) for file in sorted(glob.glob(f'./drone/processed/subset20171791057_contours/*.pkl'), key=numerical_sort)]

    #distribution of contours areas - find knee point between noise/contamination and waves - min area size (in pixels) to filter contours out by
    knee = contour_area_dist(conts, sigma=7) #sigma of 5 or 7 normally good - area threshold based on the histogram of contour areas - ELIMINATES NOISE BUT MAY LOSE SOME SMALL WAVES, now different with clipped images
    # knee=5   #standard area threshold to avoid errors when ellipse fitting - KEEPS MORE NOISE BUT GETS ALL WAVES
    # print(knee)

    #filter contours
    print('Filtering Contours ...')
    fps = 29.97
    time_thresh = float(2 / 3)
    # N_min = int(np.round(fps * time_thresh)) #If the timestep is too large, optical flow will not work accurately. You violate DI/DT=0
    N_min = 5 #0.1668 sec
    image_count = 0
    for frame_idx in tqdm(range(len(clipped_images) - N_min)):
        image_next = clipped_images[frame_idx + N_min]
        filter_contours(clipped_images[frame_idx], image_next, rgb_images[frame_idx], conts[frame_idx], image_count, variance_thresh=0.2, area_thresh=knee)

        image_count += 1

    #load save contours
    filtered_contours = [loadContours(file) for file in sorted(glob.glob(f'./drone/processed/subset20171791057_contours_filtered/*.pkl'), key=numerical_sort)]

    '''CONVEX HULL - not necessary'''
    #fit convex hull to contours
    smoothed_contours = [[cv2.convexHull(c) for c in conts
                          if len(c) >= 5] for conts in filtered_contours]
    for frame_idx, conts in enumerate(filtered_contours):
        hulls = [cv2.convexHull(c) for c in conts if len(c) >= 5]

        # Save each frame's hulls separately
        with open(f'./drone/processed/subset20171791057_convexHulls/hulls_frame_{frame_idx}.pkl', 'wb') as f:
            pickle.dump(hulls, f)

    fit_ellipse(rgb_images, N_min, smoothed_contours)  #This is not needed for the wave tracking script

    #TODO: install a rectify images script to correct the images before wave detection - needed for any spatial measurements.

    # TODO: make a video at some point of the contours over time






