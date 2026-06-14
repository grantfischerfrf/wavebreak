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
images = np.load('./drone/processed/1057subset_images.npy', mmap_mode='r')[:282]
def clip_images(images, x_min, x_max, y_min, y_max):
    clipped = images[:, y_min:y_max, x_min:x_max]
    return clipped
x_min, x_max = 0, images[0].shape[1]
y_min, y_max = 700, 1500
clipped_images = clip_images(images, x_min, x_max, y_min, y_max)
plt.imshow(clipped_images[-1])
plt.show()
plt.close('all')



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






