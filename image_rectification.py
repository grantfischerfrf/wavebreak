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

# def rectifyImages(images, drone_intrinsics_path, drone_extrinsics_path, save_path, plot=True):
#
#     #create intrinsics and extrinsics arrays for rectification
#     #FIXME: Really sloppy code, need to change how the mat file is opened so a variable is not being indexed four times
#     '''INTRINSICS'''
#     drone_metadata = scipy.io.loadmat(drone_intrinsics_path)
#     calib = drone_metadata['CopterCurrents_CamCalib']
#     fc = calib['fc']  # focal length x and y
#     fcx, fcy = fc[0][0][0][0], fc[0][0][1][0]
#
#     cc = calib['cc']  # principal point x and y
#     ccx, ccy = cc[0][0][0][0], cc[0][0][1][0]
#
#     kc = calib['kc']  # distortion coefficients - k1, k2, k3, p1, p2 - radial and tangential distortion coefficients
#     k1, k2, k3, p1, p2 = kc[0][0][0][0], kc[0][0][1][0], kc[0][0][2][0], kc[0][0][3][0], kc[0][0][4][0]
#
#     alpha_c = calib['alpha_c'][0][0][0][0]  # skew coefficient
#     nx = calib['nx'][0][0][0][0]  # image width
#     ny = calib['ny'][0][0][0][0]  # image height
#
#     # create 1x11 intrinsics array from the calibration parameters
#     intrinsics = np.array([[nx, ny, ccx, ccy, fcx, fcy, k1, k2, k3, p1, p2]])
#
#     '''EXTRINSICS'''
#     # create 1x6 extrinsics array x,y,z,a,t,s
#     ext_metadata = np.load(drone_extrinsics_path)
#     azimuth = float(ext_metadata[4][1]) * (np.pi / 180)  # drone heading - equivalent to azimuth - convert to radians
#     tilt = 0 * (np.pi / 180)  # drone tile, nadir view, fixed value - assumed so in Alex's paper
#     swing = 0 * (np.pi / 180)  # drone swing, fixed value - assumed so in Alex's paper
#     z = float(ext_metadata[1][
#                   1])  # altitude in meters - This is assumed absolute altitude above the ground as recorded by the DJI drone in the flight logs. Should be relative to MSL.
#     x = float(ext_metadata[3][1])  # longitude in decimal degrees
#     y = float(ext_metadata[2][1])  # latitude in decimal degrees
#
#     # pull ground sampling distance for grid
#     dx = float(ext_metadata[-2][1])  # dx and dy are the same value
#     gsd = dx
#
#     # convert x, y, z from wgs84 to UTM 18N for connecticut river - long island sound
#     transformer = Transformer.from_crs("epsg:4326", "epsg:32618", always_xy=True)
#     easting, northing = transformer.transform(x, y)
#
#     # create 1x6 extrinsics array from the extrinsic parameters
#     extrinsics = np.array([[easting, northing, z, azimuth, tilt, swing]])
#
#     fov_x = 2 * np.arctan(intrinsics[0][1] / (2 * intrinsics[0][5]))  # horizontal FOV
#     fov_y = 2 * np.arctan(intrinsics[0][0] / (2 * intrinsics[0][4]))  # vertical FOV
#
#     half_width = z * np.tan(fov_x / 2)
#     half_height = z * np.tan(fov_y / 2)
#
#     margin = 1.6
#     x_grid = np.arange(easting - half_width * margin, easting + half_width * margin,  #TODO: fix this
#                        gsd)  # resolution must be equivalent to the ground sampline distance
#     y_grid = np.arange(northing - half_height * 1.2, northing + half_height * 1.2, gsd)
#
#
#     xx, yy = np.meshgrid(x_grid, y_grid)
#     zz = np.zeros_like(xx)  # flat ground at z=0 MSL
#
#     xyz_grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))  # shape (N, 3) where N is number of grid points
#
#     #set up memmap to save rectified arrays
#     N_frames = len(images)
#     H, W = images[0].shape
#     memmap_shape = (N_frames, H, W)
#     rectified_image_memmap = np.memmap(save_path, dtype='uint8', mode='w+', shape=memmap_shape) #can also save as float32 for more precise data
#
#     for n in tqdm(range(len(images))):
#         if type(images[n]) == 'str':
#             img = cv2.cvtColor(cv2.imread(images[n]), cv2.COLOR_BGR2RGB)
#
#         else:
#             img = images[n]
#
#         # img_w = img.shape[1]
#         # img_h = img.shape[0]
#         Ud, Vd = xyz2DistUV(intrinsics[0], extrinsics[0], xyz_grid)
#
#         s = xx.shape
#         DU = Ud.reshape(s)
#         DV = Vd.reshape(s)
#         # get pixel intensities at each (Ud, Vd) coordinate
#         ir = getPixels(img, DU, DV, s)  # Ud and Vd should be shaped to match the grid dimensions
#         #ir is of shape (4609, 3430, 3)
#
#         rectified_image_memmap[n] = ir
#
#
#
#         if plot:
#             # Quick sanity plot
#             plt.figure(figsize=(10, 10))
#             if ir.shape[2] == 3:  #if the image is three channels
#                 plt.imshow(np.clip(ir.astype(np.uint8), 0, 255), extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
#                    origin='lower')
#             else:  #if not three channels plot grayscale
#                 plt.imshow(ir[:, :, 0], cmap='gray', extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
#                    origin='lower')
#             plt.title("Rectified image")
#             plt.xlabel("Easting (m)")
#             plt.ylabel("Northing (m)")
#             # plt.savefig('./drone/outputs/temp/rectified_sample.png')
#             plt.show()
#             plt.close('all')
#
#         # periodically flush disk to avoid memory buildup
#         if n % 100 == 0:
#             rectified_image_memmap.flush()
#
#     #final flush
#     rectified_image_memmap.flush()


def rectifyImages(images, drone_intrinsics_path, drone_extrinsics_path, save_path, plot=True):

    '''INTRINSICS'''
    drone_metadata = scipy.io.loadmat(drone_intrinsics_path)
    calib = drone_metadata['CopterCurrents_CamCalib']
    fc = calib['fc']
    fcx, fcy = fc[0][0][0][0], fc[0][0][1][0]

    cc = calib['cc']
    ccx, ccy = cc[0][0][0][0], cc[0][0][1][0]

    kc = calib['kc']
    k1, k2, k3, p1, p2 = kc[0][0][0][0], kc[0][0][1][0], kc[0][0][2][0], kc[0][0][3][0], kc[0][0][4][0]

    alpha_c = calib['alpha_c'][0][0][0][0]
    nx = calib['nx'][0][0][0][0]
    ny = calib['ny'][0][0][0][0]

    intrinsics = np.array([[nx, ny, ccx, ccy, fcx, fcy, k1, k2, k3, p1, p2]])

    '''EXTRINSICS'''
    ext_metadata = np.load(drone_extrinsics_path)
    azimuth = float(ext_metadata[4][1]) * (np.pi / 180)
    tilt    = 0 * (np.pi / 180)
    swing   = 0 * (np.pi / 180)
    z = float(ext_metadata[1][1])
    x = float(ext_metadata[3][1])
    y = float(ext_metadata[2][1])

    dx  = float(ext_metadata[-2][1])
    gsd = dx

    transformer = Transformer.from_crs("epsg:4326", "epsg:32618", always_xy=True)
    easting, northing = transformer.transform(x, y)

    extrinsics = np.array([[easting, northing, z, azimuth, tilt, swing]])

    '''GRID FROM IMAGE CORNERS'''
    img_h, img_w = images[0].shape[:2]
    corners_uv = np.array([
        [0,     0    ],
        [img_w, 0    ],
        [0,     img_h],
        [img_w, img_h],
    ])

    corner_world = uv2XYZ(intrinsics[0], extrinsics[0], corners_uv, z_ground=0)

    east_min,  north_min = corner_world.min(axis=0)
    east_max,  north_max = corner_world.max(axis=0)

    buffer = 5
    x_grid = np.arange(east_min - buffer,  east_max + buffer,  gsd)
    y_grid = np.arange(north_min - buffer, north_max + buffer, gsd)

    xx, yy = np.meshgrid(x_grid, y_grid)
    zz     = np.zeros_like(xx)
    xyz_grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    '''MEMMAP — shape derived from the rectified grid'''
    N_frames   = len(images)
    img_h_rect = len(y_grid)   # rectified output rows
    img_w_rect = len(x_grid)   # rectified output cols

    # detect channels from first image
    first_img  = images[0] if not isinstance(images[0], str) \
                 else cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)
    n_channels = first_img.shape[2] if first_img.ndim == 3 else 1

    if n_channels > 1:
        memmap_shape = (N_frames, img_h_rect, img_w_rect, n_channels)
    else:
        memmap_shape = (N_frames, img_h_rect, img_w_rect)

    rectified_image_memmap = np.memmap(save_path, dtype='uint8', mode='w+', shape=memmap_shape)

    '''RECTIFICATION LOOP'''
    s = xx.shape
    Ud, Vd = xyz2DistUV(intrinsics[0], extrinsics[0], xyz_grid)  # compute once, reuse every frame
    DU = Ud.reshape(s)
    DV = Vd.reshape(s)

    for n in tqdm(range(N_frames)):
        img = images[n] if not isinstance(images[n], str) \
              else cv2.cvtColor(cv2.imread(images[n]), cv2.COLOR_BGR2RGB)

        ir = getPixels(img, DU, DV, s)  # (img_h_rect, img_w_rect, C)

        # replace NaNs with 0 before casting
        ir_clean = np.nan_to_num(ir, nan=0.0)

        if n_channels > 1:
            rectified_image_memmap[n] = np.clip(ir_clean, 0, 255).astype(np.uint8)
        else:
            rectified_image_memmap[n] = np.clip(ir_clean[:, :, 0], 0, 255).astype(np.uint8)

        if plot:
            plt.figure(figsize=(10, 10))
            if n_channels == 3:
                plt.imshow(np.clip(ir, 0, 255).astype(np.uint8),
                           extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
                           origin='lower')
            else:
                plt.imshow(ir[:, :, 0], cmap='gray',
                           extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
                           origin='lower')
            plt.title("Rectified image")
            plt.xlabel("Easting (m)")
            plt.ylabel("Northing (m)")
            plt.show()
            plt.close('all')

        if n % 100 == 0:
            rectified_image_memmap.flush()

    rectified_image_memmap.flush()

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
        # plt.savefig('./drone/outputs/temp/rectified_sample.png')
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







