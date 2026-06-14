import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import glob
import scipy
import mat73
import pyproj
from pyproj import Transformer
import sys
from math import sin, cos
import corefunctions as cf
import supportfunctions as sf
from tqdm import tqdm
import imageio
from scipy.interpolate import RegularGridInterpolator as reg_interp

def numerical_sort(string):

    parts = re.split(r'(\d+)', string)
    return [int(part) if part.isdigit() else part for part in parts]

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


def localTransformExtrinsics(extrinsics, origin:list, coords:str):

    local_origin = np.array((origin[0], origin[1]))
    angle = origin[2]

    if coords == "geo":
        # World to local

        extrinsics[0], extrinsics[1] = localTransformPoints(
            local_origin,
            angle,
            1,
            extrinsics[0],
            extrinsics[1]
        )
        extrinsics[3] = extrinsics[3] + angle
    else:
        # local to world
        extrinsics[0], extrinsics[1] = localTransformPoints(
            local_origin,
            angle,
            0,
            extrinsics[0],
            extrinsics[1]
        )
        extrinsics[3] = extrinsics[3] - angle

    return extrinsics


def localTransformPoints(localOrigin, localAngle, flag, Xin, Yin):


#World to Local
    if flag == 1:

        xp = Xin-localOrigin[0]
        yp = Yin-localOrigin[1]

        Xout = xp * np.cos(localAngle) + yp * np.sin(localAngle)
        Yout = yp * np.cos(localAngle) - xp * np.sin(localAngle)

        return Xout, Yout

#Local to World
    if flag == 0:

        Xout = Xin * np.cos(localAngle) - Yin * np.sin(localAngle)
        Yout = Yin * np.cos(localAngle) + Xin * np.sin(localAngle)

        Xout = Xout + localOrigin[0]
        Yout = Yout + localOrigin[1]

        return Xout, Yout


def getPixels(image, Ud, Vd, s):

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
                bounds_error=False,
                fill_value=np.nan,
            )
            ir[:, :, i] = rgi((Vd, Ud))
    else:
        ir = np.full((s[0], s[1], 1), np.nan)
        rgi = reg_interp(
            (np.arange(0, image.shape[0]), np.arange(0, image.shape[1])),
            image,
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


if __name__ == "__main__":
    # rgb images - non modified
    rgb_images = sorted(glob.glob('./drone/processed/subset20171791057/*.jpeg'), key=numerical_sort)[:282]

    #pull intrinsics
    drone_metadata = scipy.io.loadmat('./drone/DJI_3840x2160.mat')
    calib = drone_metadata['CopterCurrents_CamCalib']
    fc = calib['fc'] #focal length x and y
    fcx, fcy = fc[0][0][0][0], fc[0][0][1][0]

    cc = calib['cc'] #principal point x and y
    ccx, ccy = cc[0][0][0][0], cc[0][0][1][0]

    kc = calib['kc'] #distortion coefficients - k1, k2, k3, p1, p2 - radial and tangential distortion coefficients
    k1, k2, k3, p1, p2 = kc[0][0][0][0], kc[0][0][1][0], kc[0][0][2][0], kc[0][0][3][0], kc[0][0][4][0]

    alpha_c = calib['alpha_c'][0][0][0][0] #skew coefficient
    nx = calib['nx'][0][0][0][0] #image width
    ny = calib['ny'][0][0][0][0] #image height

    #create 1x11 intrinsics array from the calibration parameters
    intrinsics = np.array([[nx, ny, ccx, ccy, fcx, fcy, k1, k2, k3, p1, p2]])

    #create 1x6 extrinsics array x,y,z,a,t,s
    ext_metadata = np.load('./drone/1057metadata.npy')
    azimuth = float(ext_metadata[4][1]) * (np.pi / 180) #drone heading - equivalent to azimuth - convert to radians
    tilt = 0 * (np.pi / 180) #drone tile, nadir view, fixed value - assumed so in Alex's paper
    swing = 0 * (np.pi / 180) #drone swing, fixed value - assumed so in Alex's paper
    z = float(ext_metadata[1][1]) #altitude in meters - This is assumed absolute altitude above the ground as recorded by the DJI drone in the flight logs. Should be relative to MSL.
    x = float(ext_metadata[3][1]) #longitude in decimal degrees
    y = float(ext_metadata[2][1]) #latitude in decimal degrees

    #pull ground sampling distance for grid
    dx = float(ext_metadata[-2][1])  # dx and dy are the same value
    gsd = dx

    #convert x, y, z from wgs84 to UTM 18N for connecticut river - long island sound
    transformer = Transformer.from_crs("epsg:4326", "epsg:32618", always_xy=True)
    easting, northing = transformer.transform(x, y)

    #create 1x6 extrinsics array from the extrinsic parameters
    extrinsics = np.array([[easting, northing, z, azimuth, tilt, swing]])

    fov_x = 2 * np.arctan(intrinsics[0][1] / (2 * intrinsics[0][5]))  # horizontal FOV
    fov_y = 2 * np.arctan(intrinsics[0][0] / (2 * intrinsics[0][4]))  # vertical FOV

    half_width = z * np.tan(fov_x / 2)
    half_height = z * np.tan(fov_y / 2)

    margin = 1.6
    x_grid = np.arange(easting - half_width * margin, easting + half_width * margin, gsd) #resolution must be equivalent to the ground sampline distance
    y_grid = np.arange(northing - half_height * 1.2, northing + half_height * 1.2, gsd)

    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = np.zeros_like(xx)  # flat ground at z=0 MSL

    xyz_grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))  # shape (N, 3) where N is number of grid points

    for img_path in tqdm(rgb_images):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_w = img.shape[1]
        img_h = img.shape[0]
        Ud, Vd = xyz2DistUV(intrinsics[0], extrinsics[0], xyz_grid)

        # in_bounds = (Ud > 0) & (Ud < intrinsics[0][0]) & (Vd > 0) & (Vd < intrinsics[0][1])
        # print(f"Points in image bounds: {in_bounds.sum()} / {len(Ud)} ({in_bounds.mean() * 100:.1f}%)")
        #
        # plt.figure()
        # plt.scatter(Ud[in_bounds][::100], Vd[in_bounds][::100], s=1)
        # plt.xlim(0, intrinsics[0][0])
        # plt.ylim(intrinsics[0][1], 0)  # flip Y axis to match image coords
        # plt.title("UV projection coverage")
        # plt.xlabel("U")
        # plt.ylabel("V")
        # plt.show()

        s = xx.shape
        DU = Ud.reshape(s)
        DV = Vd.reshape(s)
        #get pixel intensities at each (Ud, Vd) coordinate
        ir = getPixels(img, DU, DV, s)  #Ud and Vd should be shaped to match the grid dimensions

        # Quick sanity plot
        plt.figure(figsize=(10, 10))
        if ir.shape[2] == 3:  #if the image is three channels
            plt.imshow(np.clip(ir.astype(np.uint8), 0, 255), extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
               origin='lower')
        else:  #if not three channels plot grayscale
            plt.imshow(ir[:, :, 0], cmap='gray', extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
               origin='lower')
        plt.title("Rectified image")
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.savefig('./drone/outputs/temp/rectified_sample.png')
        plt.show()
        plt.close('all')



    '''coastal imagelib code'''
    #create CameraData object
    # cams = ['1']
    # cameras = np.empty(len(cams), dtype=object)
    # for i in range(len(cams)):
    #     cameras[i] = cf.CameraData(intrinsics[i], extrinsics[i], mType="CIRN", coords="local", nc=3)
    #
    # #create XYZGrid object
    # #grid boundaries
    # xMin = easting - 200
    # xMax = easting + 200
    # yMin = northing - 200
    # yMax = northing + 200
    # # Image Resolution
    # dy = 1
    # dx = 1
    # # Estimated elevation of every point in the X,Y grid
    # z = 0
    #
    # #grid object
    # grid = cf.XYZGrid([xMin, xMax], [yMin, yMax], dx, dy, z)
    #
    # # rectify images
    # for frame in rgb_images:
    #     current_frame = [frame]
    #     rect_frame = cf.mergeRectify(current_frame, cameras, grid)
    #
    #     # plt.imshow(rect_frame)
    #     # plt.show()
    #     # plt.close('all')
    #
    # test_point = np.array([[easting, northing, 0.0]])  # directly below drone -
    # Ut, Vt = xyz2DistUV(intrinsics[0], extrinsics[0], test_point)







