import numpy as np
import cv2
import glob
import os
import re
from tqdm import tqdm
import json
import pickle

from scipy.interpolate import RegularGridInterpolator as reg_interp


def numerical_sort(string):

    parts = re.split(r'(\d+)', string)
    return [int(part) if part.isdigit() else part for part in parts]


def clip_images(images, x_min, x_max, y_min, y_max):
    clipped = images[:, y_min:y_max, x_min:x_max]
    return clipped

def loadContours(path):
    with open(path, 'rb') as f:
        contour_list = pickle.load(f)
    return contour_list


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
                newVal=255, loDiff=0, upDiff=0,  # exact-match flood - fill only goes to exact matches of the seed (0 - black here)
                flags=8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)  #count the diagonal neighbors - write result to the flood mask only - dont alter the original image
            )
            padding = cv2.bitwise_or(padding, temp_flood_mask[1:-1, 1:-1])
            # temp_flood_mask comes back as the padded (h+2, w+2) size with fill values of 1,
            # so [1:-1, 1:-1] crops back to the original (h, w) frame size
            # bitwise or allows the running total, so all 4 corners' padding
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

def load_opticalFlow_frame(optical_flow_dir, chunk_idx, frame):
    #loads the vx, vy, shape optical flow memmap output for a given directory and chunk index and particular frame in the memmap
    vx_path = os.path.join(optical_flow_dir, f'vx_{chunk_idx:04d}.dat')  # path to optical flow x velocities
    vy_path = os.path.join(optical_flow_dir, f'vy_{chunk_idx:04d}.dat')  # path to y velocities
    shape = tuple(np.load(os.path.join(optical_flow_dir, f'flow_shape_{chunk_idx:04d}.npy')))  # path to the shape of the memory maps
    vx = np.memmap(vx_path, dtype='float32', mode='r', shape=shape)[frame, :, :] # in relative units of pixels/frame
    vy = np.memmap(vy_path, dtype='float32', mode='r', shape=shape)[frame, :, :]

    return vx, vy, shape


def find_knee(hist, sigma=5):
    # find where histogram drops off from the spike - sigma is the sliding window size that the variance is calculated over.
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



def load_chunk(save_dir, chunk_index):
    meta_path = os.path.join(save_dir, f'rectified_chunk_{chunk_index:04d}_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    memmap_path = os.path.join(save_dir, f'rectified_chunk_{chunk_index:04d}.dat')
    arr = np.memmap(memmap_path, dtype=meta['dtype'], mode='r',
                     shape=tuple(meta['memmap_shape']))
    return arr, meta


def uv2XYZ(intrinsics, extrinsics, uv_points, z_ground=0):
    """
    Back-projects UV pixel coordinates to world XYZ at a known ground elevation

    intrinsics  : 1x11 array [nx, ny, ccx, ccy, fcx, fcy, k1, k2, k3, p1, p2]
    extrinsics  : 1x6  array [x, y, z, azimuth, tilt, swing]
    uv_points   : Nx2  array of (U, V) pixel coordinates
    z_ground    : ground plane elevation (default 0 MSL)

    returns     : Nx2 array of (easting, northing) world coordinates
    """
    K, R, IC, P, C = intrinsicsExtrinsics2P(intrinsics, extrinsics)

    cam_pos = np.array([extrinsics[0], extrinsics[1], extrinsics[2]])  # camera XYZ in world

    world_points = []
    for (u, v) in uv_points:
        # Homogeneous pixel coordinate
        uv_h = np.array([u, v, 1.0])

        # Back-project: ray direction in world coords
        # P = K @ R @ IC, so K_inv @ R_inv un-rotates the pixel ray into world space
        K = np.array(K)
        ray_cam = np.linalg.inv(K) @ uv_h  # ray in camera frame (normalized)
        ray_world = R.T @ ray_cam  # rotate to world frame

        # Intersect ray with ground plane: cam_pos + t * ray_world, solve for z = z_ground
        # t = (z_ground - cam_z) / ray_world[2]
        t = (z_ground - cam_pos[2]) / ray_world[2]
        ground_pt = cam_pos + t * ray_world

        world_points.append(ground_pt[:2])  # easting, northing

    return np.array(world_points)


def xyz2DistUV(intrinsics, extrinsics, xyz):
    '''
    Converts XYZ world coordinates to UV image coordinates
    :param intrinsics: 1x11 array of camera intrinsics
    :param extrinsics: 1x6 array of camera extrinsics
    :param xyz: nx3 array of world coordinates

    :return:
        Ud: array of distorted U image coordinates
        Vd: array of distorted V image coordinates
    '''

    K, R, IC, P, C = intrinsicsExtrinsics2P(intrinsics, extrinsics)

    xyz_transpose = xyz.conj().transpose()
    ones = np.ones((1, xyz.shape[0]))
    xyz_aug = np.vstack((xyz_transpose, ones))
    UV = P @ xyz_aug

    UV = UV / np.tile(UV[2, :], (3, 1))

    U = UV[0, :]
    V = UV[1, :]

    Ud, Vd, flag = distortUV(U, V, intrinsics)

    #reshape ud and vd to a grid shape if you want them as a grid

    # flag negative z values
    xyzC = R @ (IC @ xyz_aug)
    bind = np.where(xyzC[2, :] <= 0)
    flag[bind] = 0

    #UVd = np.vstack((Ud, Vd))

    # return Ud * flag, Vd * flag
    return Ud, Vd


def intrinsicsExtrinsics2P(intrinsics, extrinsics):

    #create K matrix
    fx = intrinsics[4]
    fy = intrinsics[5]
    c0U = intrinsics[2]
    c0V = intrinsics[3]

    K = [[-fx, 0, c0U], [0, -fy, c0V], [0, 0, 1]]

    #create R matrix
    azimuth = extrinsics[3]
    tilt = extrinsics[4]
    swing = extrinsics[5]

    R = Angles2R(azimuth, tilt, swing)

    #create IC matrix
    x = extrinsics[0]
    y = extrinsics[1]
    z = extrinsics[2]

    I = np.identity(3)
    C = np.array([-x, -y, -z])
    C = C.conj().transpose()
    IC = np.column_stack((I,C))

    #create P matrix
    P = K @ R @ IC
    #normalize for homogenous coordinates
    P = P/P[-1][-1]

    return K, R, IC, P, C


def Angles2R(azimuth, tilt, swing):

    #initialize empty array to create R matrix
    R = np.empty((3,3))

    R[0][0] = -np.cos(azimuth) * np.cos(swing) - np.sin(azimuth) * np.cos(tilt) * np.sin(swing)
    R[0][1] = np.cos(swing) * np.sin(azimuth) - np.sin(swing) * np.cos(tilt) * np.cos(azimuth)
    R[0][2] = -np.sin(swing) * np.sin(tilt)
    R[1][0] = -np.sin(swing) * np.cos(azimuth) + np.cos(swing) * np.cos(tilt) * np.sin(azimuth)
    R[1][1] = np.sin(swing) * np.sin(azimuth) + np.cos(swing) * np.cos(tilt) * np.cos(azimuth)
    R[1][2] = np.cos(swing) * np.sin(tilt)
    R[2][0] = np.sin(tilt) * np.sin(azimuth)
    R[2][1] = np.sin(tilt) * np.cos(azimuth)
    R[2][2] = -np.cos(tilt)

    return R


def distortUV(U, V, intrinsics):

    # assign coefficients
    NU = intrinsics[0]
    NV = intrinsics[1]
    c0U = intrinsics[2]
    c0V = intrinsics[3]
    fx = intrinsics[4]
    fy = intrinsics[5]
    d1 = intrinsics[6]
    d2 = intrinsics[7]
    d3 = intrinsics[8]
    t1 = intrinsics[9]
    t2 = intrinsics[10]

    # calculate distorted coordinates
    # normalize distances
    x = (U - c0U) / fx
    y = (V - c0V) / fy

    # radial distortion
    r2 = x * x + y * y
    fr = 1.0 + d1 * r2 + d2 * (r2 * r2) + d3 * (r2 * (r2 * r2))

    #Tangential distortion
    dx = 2.0 * t1 * x * y + t2 * (r2 + 2.0 * x * x)
    dy = t1 * (r2 + 2.0 * y * y) + 2.0 * t2 * x * y

    #apply correction, answer in chip pixel units
    xd = x * fr + dx
    yd = y * fr + dy
    Ud = xd * fx + c0U
    Vd = yd * fy + c0V

    #determine if points are within image
    #initialize flag that all are acceptable (negative coordinates)
    mask = (Ud < 0) | (Ud > NU) | (Vd < 0) | (Vd > NV)
    Ud[mask] = 0
    Vd[mask] = 0

    #determine if tangential distortion is within range
    #find maximum possible tangential distortion at corners
    Um = np.array((0, 0, NU, NU))
    Vm = np.array((0, NV, NV, 0))

    #normalization
    xm = (Um - c0U) / fx
    ym = (Vm - c0V) / fy
    r2m = xm * xm + ym * ym

    #tangential distortion
    dxm = 2.0 * t1 * xm * ym + t2 * (r2m + 2.0 * xm * xm)
    dym = t1 * (r2m + 2.0 * ym * ym) + 2.0 * t2 * xm * ym

    #find values larger than x and y limits
    flag = np.ones_like(Ud)
    flag[np.where(np.abs(dy) > np.max(np.abs(dym)))] = 0.0
    flag[np.where(np.abs(dx) > np.max(np.abs(dxm)))] = 0.0

    return Ud, Vd, flag


def getPixels(image, Ud, Vd, s, method='nearest'):

    """
    Pulls rgb or gray pixel intensities from image at specified
    pixel locations corresponding to X,Y coordinates calculated in either
    xyz2DistUV or dlt2UV.

    Args:
        image (ndarray): image where pixels will be taken from
        Ud: Nx1 vector of distorted U coordinates for N points
        Vd: Nx1 vector of distorted V coordinates for N points
        s: shape of output image

    Returns:
        ir (ndarray): pixel intensities

    """

    # Use regular grid interpolator to grab points
    im_s = image.shape
    if len(im_s) > 2:
        ir = np.full((s[0], s[1], im_s[2]), np.nan)
        for i in range(im_s[2]):
            rgi = reg_interp(
                (np.arange(0, image.shape[0]), np.arange(0, image.shape[1])),
                image[:, :, i],
                method=method,  #TODO: this can smear pixel intensities and make it hard to do image processing for breaking waves
                bounds_error=False,
                fill_value=np.nan,
            )
            ir[:, :, i] = rgi((Vd, Ud))
    else:
        ir = np.full((s[0], s[1], 1), np.nan)
        rgi = reg_interp(
            (np.arange(0, image.shape[0]), np.arange(0, image.shape[1])),
            image,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        ir[:, :, 0] = rgi((Vd, Ud))

    # Mask out values out of range
    with np.errstate(invalid="ignore"):
        mask_u = np.logical_or(Ud <= 1, Ud >= image.shape[1])
        mask_v = np.logical_or(Vd <= 1, Vd >= image.shape[0])
    mask = np.logical_or(mask_u, mask_v)
    if len(im_s) > 2:
        ir[mask, :] = np.nan
    else:
        ir[mask] = np.nan

    return ir









