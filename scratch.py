import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import re
from scipy.stats import pearsonr
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
import scipy
import mat73
import utils

def numerical_sort(string):

    parts = re.split(r'(\d+)', string)
    return [int(part) if part.isdigit() else part for part in parts]

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


def get_mask(frame, erode_px=15):

    #mask the dark border around the rectified image so that it does not contaminate the contour detection
    h, w = frame.shape #get the frame dimensions

    # flood fill from all 4 corners to capture the connected black padding
    padding = np.zeros((h, w), dtype=np.uint8)  #holds the flood filled padding across all seed corners

    ff_img = frame.copy()  #copy the frame to flood fill on so the in place modification does not affect the original array
    seeds = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)] #corners of the image (row, col) format
    for seed in seeds:
        # skip corners that are already marked
        if padding[seed] == 0:  #if a corner has not been already captured by a previous flood fill, skip it
            temp_flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)  #output mask to flood fill, must be 2px larger on height and width by documentation
            cv2.floodFill(
                ff_img.copy(), temp_flood_mask, seedPoint=(seed[1], seed[0]),
                newVal=255, loDiff=5, upDiff=5,  # exact-match flood - fill only goes to exact matches of the seed (0 - black here)
                flags=8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)  #count the diagonal neighbors - write result to the flood mask only - dont alter the original image
            )
            padding = cv2.bitwise_or(padding, temp_flood_mask[1:-1, 1:-1])
            # temp_flood_mask comes back as the padded (h+2, w+2) size with fill values of 1,
            # so [1:-1, 1:-1] crops back to the original (h, w) frame size, bitwise or allows the running total, so all 4 corners' padding
            # regions accumulate into one combined mask

    # valid footprint = everything not flood-filled padding
    valid_mask = cv2.bitwise_not(padding)
    # invert: padding pixels (255) become 0 (invalid), everything else becomes 255 (valid image content)

    # shrink inward to exclude the blurred rectification edge zone
    # if erode_px > 0:
    #     erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
    #     valid_mask = cv2.erode(valid_mask, erode_kernel)
        # build a circular structuring element sized so erosion pulls the mask boundary
        # inward by erode_px pixels on each side

    return valid_mask


'''Camera calibration'''
# cam = scipy.io.loadmat('./drone/DJI_3840x2160.mat')
# results = scipy.io.loadmat('./drone/Calib_Results.mat')
# image_coords = scipy.io.loadmat('./drone/img_coords.mat')
# gridE = image_coords['gridE']
# gridN = image_coords['gridN']
# calib = cam['CopterCurrents_CamCalib']
# nx = calib['nx'][0][0][0]
# ny = calib['ny'][0][0][0]
# z_offset = calib['camera_offset_Z']
# fc = calib['fc'] #focal length
# cc = calib['cc'] #principle point
# kc = calib['kc'] # distortion coefficients
#
# # rect_fc = results['fc']
# # fc_error = results['fc_error']
# # rect_cc = results['cc']
# # rect_kc = results['kc']
# # dy = results['dY']
# # dx = results['dX']
# # ex = results['ex']
#
# # caltech = scipy.io.loadmat('./drone/Phantom3_v1_FOV_3840x2160.mat')
#
# # output = mat73.loadmat('./drone/20171791057_avg.mat')
# # metadata = output['MetaData']
# # np.save('./drone/1057metadata.npy', metadata)
# drone = np.load('./drone/1057metadata.npy')  #CAN GET GSD FROM HERE NOW!
#
# minE, maxE = gridE.min(), gridE.max()
# minN, maxN = gridN.min(), gridN.max()  #The distance north is less than the distance east because the image is projected onto the world where the long axis of the image is more oriented up and down.
#
# length_of_image_E = np.abs(maxE - minE)
# length_of_image_N = np.abs(maxN - minN)
#
# dx = float(drone[-2][1]) #FIXME THIS IS ALREADY THE GROUND SAMPLING DISTANCE.
# altitude = float(drone[1][1])
# avg_focal_length = (fc[0][0][0] + fc[0][0][1]) / 2
# avg_GSD = (altitude * dx) / avg_focal_length



'''image clipping'''
# images = np.load('./drone/processed/1057subset_images.npy', mmap_mode='r')[:282]
# def clip_images(images, x_min, x_max, y_min, y_max):
#     clipped = images[:, y_min:y_max, x_min:x_max]
#     return clipped
# x_min, x_max = 0, images[0].shape[1]
# y_min, y_max = 700, 1500
# clipped_images = clip_images(images, x_min, x_max, y_min, y_max)
# plt.imshow(clipped_images[-1])
# plt.show()
# plt.close('all')



'''Optical Flow stuff'''
# rgbimages = sorted(glob.glob('./drone/processed/subset20171791057/*.jpeg'), key=numerical_sort)[:282]
# rgbimage = cv2.imread(rgbimages[-10])
# rgbimagenext = cv2.imread(rgbimages[-1])
# h, w = rgbimage.shape[0], rgbimage.shape[1]
# flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(rgbimage, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rgbimagenext, cv2.COLOR_BGR2GRAY), None,0.5, 3, 15, 3, 5, 1.2, 0) #convert 0-1 to 0-255 for optical flow, previous frame, next frame
# u = flow[..., 0]  # selects all previous dimensions and the 0th index of the last dimension  #FIXME: can look at this for individual pixel rejection if wanted in the future - move this to the GPU as well speed it up
# v = flow[..., 1] #gives dense optical flow. Pixel displacement per frame. Not an actual speed
# # visualize optical flow results
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



# sampleimage = cv2.imread(rgbimages[-2])
# nextsampleimage = cv2.imread(rgbimages[-1])

# image = np.load('./drone/processed/1057subset_images.npy', mmap_mode='r')[:282][-1]
# image = (image * 255).astype(np.uint8)
# nextimage = np.load('./drone/processed/1057subset_images.npy', mmap_mode='r')[:282][-1]
# nextimage = (nextimage * 255).astype(np.uint8)

"""CURRENT PROCESSING"""
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to HSV to isolate brightness
# brt = hsv[:, :, 2]
# gaussian blur to suppress noise
# brt = cv2.GaussianBlur(image, (3, 3), sigmaX=0, borderType=cv2.BORDER_REFLECT)

# thresh_val, binary = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # plt.imshow(binary)
# # plt.show()
# # plt.close('all')
#
# #define kernels
# kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#
# # morphological cleanup
# binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
# binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
# # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
# # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cross, iterations=2)
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#
# # axes[0].imshow(cv2.imread(rgbimages[1]))
# axes[0].imshow(img_norm)
# axes[0].set_title('Original')
#
# axes[1].imshow(binary, cmap='gray')
# axes[1].set_title(f'modified binary')
#
# plt.tight_layout()
# plt.show()
# plt.close('all')
#
# conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# #plot distribution of contours
# cont_areas = [cv2.contourArea(c) for c in conts]
#
# #use scotts rule to determine number of bins the histogram should have
# n = len(cont_areas)
# cont_areas_arr = np.array(cont_areas)
# bin_width_scott = 3.5 * np.std(cont_areas_arr) / n**(1/3)
# n_bins_scott    = int(np.ceil((cont_areas_arr.max() - cont_areas_arr.min()) / bin_width_scott))
#
# #calculate histogram
# hist, bins = np.histogram(cont_areas, bins=n_bins_scott, range=(0, 256))
# knee, _ = find_knee(hist, sigma=7)
# # plt.hist(hist, bins)
# # plt.xscale('log')
#
# #create contours
# # conts = [loadContours(file) for file in sorted(glob.glob('./drone/processed/subset20171791057_contours_filtered/*.pkl'), key=numerical_sort)][-1]
# smoothed = [cv2.convexHull(c) for c in conts
#                           if len(c) >= 5]
# # Filter by contour area
# min_area = 5  #TODO: Can combine knee point with the other filters
#
# filtered_contours = [c for c in smoothed
#     if min_area < cv2.contourArea(c)]
#
# # filtered_contours = [c for c in conts
# #                      if min_area < cv2.contourArea(c)]
#
# ellipses = [cv2.fitEllipse(smoothed_conts) for smoothed_conts in filtered_contours]
#
# img = cv2.imread(rgbimages[1])
#
# # Draw contours
# img = cv2.drawContours(img, filtered_contours, -1, color=(0, 255, 255), thickness=2)
#
# # Draw ellipses
# for ellipse in ellipses:
#     if ellipse is not None:
#         img = cv2.ellipse(img, ellipse, (0, 0, 255), 2)
# #
# # Plot
# fig, axes = plt.subplots(1, 1, figsize=(14, 6))
# axes.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# axes.set_title("ConvexHull + Ellipses")
# axes.axis("off")
# x_max=3840
# # axes.set_xlim(3000, x_max)
# # axes.set_ylim(1500, 700)
#
# plt.tight_layout()
# # plt.savefig('./drone/outputs/temp/convexhull_ellipse.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close('all')


'''pixel intensities for thresholding'''

image = cv2.imread('/mnt/d/Wave_Breaking/20171791057_raw_images/20171791057_1868.png', cv2.IMREAD_GRAYSCALE)
# thresh_val, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plt.imshow(binary, cmap='gray')
# plt.hist(image.flatten(), bins=64)
# plt.show()
# plt.close('all')


rectified_chunk_dir = '/mnt/d/Wave_Breaking/rectified_images'
chunk, meta = utils.load_chunk(rectified_chunk_dir, 0)

rectified_image = chunk[0]


#
mask = get_mask(rectified_image)
# plt.imshow(mask)
# plt.show()
#
valid_vals = rectified_image[mask > 0]  # grab the valid pixel values inside the image - toss the rectified padding for the otsu calculation
rect_thresh_val, rect_binary = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # apply otsu thresholding to get a threshold value
_, rectified_binary = cv2.threshold(rectified_image, rect_thresh_val, 255, cv2.THRESH_BINARY)  # apply to full 2D image
# print(rect_thresh_val)
plt.imshow(rectified_binary, cmap='gray')
# plt.hist(valid_vals, bins=64)
plt.show()
plt.close('all')







