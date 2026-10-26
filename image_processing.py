import numpy as np
import cv2
import glob
import os
import re
from tqdm import tqdm
import scipy
from pyproj import Transformer
import matplotlib.pyplot as plt
import json

import utils


def processVideo(file_path, save_path, start_frame=None, end_frame=None, decimate=False, new_fps=30):

    capture = cv2.VideoCapture(file_path)  #create video capture object

    original_fps = capture.get(cv2.CAP_PROP_FPS)  # get original fps of video
    print(f'FPS: {original_fps}')

    out_name = os.path.basename(file_path).split('.')[0] #create output name for frames
    # create output directory if it doesn't exist
    # if not os.path.exists(save_path + out_name):
    #     os.makedirs(save_path + out_name)

    print(f'Processing Video {out_name}...')

    frame_count = 0  #initialize a frame count

    if start_frame == None:  #if the user does not define a starting and ending frame - use the entire video
        start_frame = 0
        end_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        start_frame = start_frame  #user defined start frame
        end_frame = end_frame  #user defined end frame

    while capture.isOpened():  #while the capture is in use...

        ret, frame = capture.read()  #read a frame

        if not start_frame <= frame_count <= end_frame:  #if the read frame does not fall under the user defined bounds...
            frame_count += 1  #skip and add 1 to the frame count until the count reaches the user defined start
            continue

        if not ret:
            print("Can't recieve frame. Exiting...")
            break  #automatically breaks if no more frames to read

        if decimate:
            assert new_fps <= original_fps, "New fps must be less than or equal to original fps"

            frame_interval = int(original_fps/new_fps)  #define frame interval to create new fps

            if frame_count % frame_interval == 0:  #if the frame count is a multiple of the frame interval, save the frame

                cv2.imwrite(f'{save_path}{out_name}/{out_name}' + '_' + str(frame_count) + '.png', frame)  # save frame as PNG file

        else:
            # cv2.imwrite(f'{save_path}{out_name}/{out_name}' + '_' + str(frame_count) + '.png', frame)  # save all frames as PNG files
            cv2.imwrite(f'{save_path}/{out_name}' + '_' + str(frame_count) + '.png', frame)

        frame_count +=1  #add one to frame count

    capture.release()  #release the capture object
    print('Done')

    return original_fps


def get_mean_background(images):
    # normalize images based on background lighting - Kleiss and Melville 2011
    # use kleiss 2009 I' = (I/S(x, y)) * Sbar

    S = np.mean([cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in images], axis=0) / 255  # normalized mean background lighting, averaged over time dimension
    Sbar = np.mean(S, axis=(0, 1))  # average over x and y dimensions

    #TODO: may want to change this so that a mean background is iteratively updated as each chunk is processed. then pass one mean background into the process chunk function for all chunks

    return S, Sbar


def load_calibration(drone_intrinsics_path, drone_extrinsics_path):

    '''INTRINSICS'''
    drone_metadata = scipy.io.loadmat(drone_intrinsics_path)
    calib = drone_metadata['CopterCurrents_CamCalib']
    fc = calib['fc']
    fcx, fcy = fc[0][0][0][0], fc[0][0][1][0]  #focal length x and y

    cc = calib['cc']
    ccx, ccy = cc[0][0][0][0], cc[0][0][1][0]  #principal point x and y

    kc = calib['kc']  #distortion coefficients - k1, k2, k3, p1, p2 - radial and tangential distortion coefficients
    k1, k2, k3, p1, p2 = kc[0][0][0][0], kc[0][0][1][0], kc[0][0][2][0], kc[0][0][3][0], kc[0][0][4][0]

    alpha_c = calib['alpha_c'][0][0][0][0]  #skew coefficient
    nx = calib['nx'][0][0][0][0]  #image width
    ny = calib['ny'][0][0][0][0] # image height

    # create 1x11 intrinsics array from the calibration parameters
    intrinsics = np.array([[nx, ny, ccx, ccy, fcx, fcy, k1, k2, k3, p1, p2]])

    '''EXTRINSICS'''  #create 1x6 extrinsics array x,y,z,a,t,s
    ext_metadata = np.load(drone_extrinsics_path)
    azimuth = float(ext_metadata[4][1]) * (np.pi / 180)  #drone heading - equivalent to azimuth - convert to radians
    tilt = 0 * (np.pi / 180) #drone tile, nadir view, fixed value - assumed so in Alex's paper
    swing = 0 * (np.pi / 180) #drone swing, fixed value - assumed so in Alex's paper
    z = float(ext_metadata[1][1]) #altitude in meters - This is assumed absolute altitude above the ground as recorded by the DJI drone in the flight logs. Should be relative to MSL.
    x = float(ext_metadata[3][1]) #longitude in decimal degrees
    y = float(ext_metadata[2][1]) #latitude in decimal degrees

    # pull ground sampling distance for grid
    dx = float(ext_metadata[-2][1]) # dx and dy are the same value
    gsd = dx

    # convert x, y, z from wgs84 to UTM 18N for connecticut river - long island sound
    transformer = Transformer.from_crs("epsg:4326", "epsg:32618", always_xy=True)
    easting, northing = transformer.transform(x, y)

    extrinsics = np.array([[easting, northing, z, azimuth, tilt, swing]])  #create 1x6 extrinsics array from the extrinsic parameters

    return intrinsics, extrinsics, gsd


def build_rectification_grid(intrinsics, extrinsics, img_h, img_w, gsd, buffer=5):
    """Build the world-coordinate grid for projecting the rectification + distorted-UV lookup."""
    corners_uv = np.array([  #defines the four corner pixels of the source image in pixel space (u, v)
        [0, 0],
        [img_w, 0],
        [0, img_h],
        [img_w, img_h],
    ])

    #reproject the corners from pixel space to world space, assume they are flat on the z axis z=0
    corner_world = utils.uv2XYZ(intrinsics[0], extrinsics[0], corners_uv, z_ground=0)
    east_min, north_min = corner_world.min(axis=0) #define the bounding box min and max based on the world coordinates
    east_max, north_max = corner_world.max(axis=0)

    # build a 1d array of coordinates that span the bounding box - add a small buffer (in meters) so the image is not accidentally clipped
    x_grid = np.arange(east_min - buffer, east_max + buffer, gsd)
    y_grid = np.arange(north_min - buffer, north_max + buffer, gsd)

    xx, yy = np.meshgrid(x_grid, y_grid)  #create a 2D meshgrid from the 1d grids
    zz = np.zeros_like(xx) #projecting onto flat ground - every point has an elevation of 0
    xyz_grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))  #flatten the xx, yy, zz grids into a single (N, 3) array - needed for function input
    #ravel preserves order to allow reshaping later

    s = xx.shape #2-D image shape of rows and columns to reshape the flattened arrays
    Ud, Vd = utils.xyz2DistUV(intrinsics[0], extrinsics[0], xyz_grid) #for every world coordinate - reproject to image space
    DU = Ud.reshape(s)  #reshape the flat Ud and Vd arrays to the 2D shape of s
    DV = Vd.reshape(s)
    # The camera shot this scene at an angle, but we want a top-down view.
    # Each output pixel (i, j) = one real-world ground location (xx[i,j], yy[i,j]).
    # DU/DV tell us where that same location appears in the original angled photo,
    # so we can look it up and "flatten" the perspective.

    #or another way to put it:
    # Convert real-world ground coordinates (xx, yy) into distorted pixel
    # coordinates (DU, DV) in the original photo, so each real-world grid
    # point can be matched to where it actually appears in the raw image.

    return DU, DV, x_grid, y_grid


def rectify_chunk(norm_images, DU, DV, x_grid, y_grid, save_path, plot=False):
    """
    Rectify a chunk of already-normalized/clipped images
    and write them to their own memmap file.
    """
    s = DU.shape # (rows, cols) shape of the rectified output grid
    N_frames = len(norm_images) #number of normalized images
    img_h_rect, img_w_rect = s  #pull the height and width of the image

    first_img = norm_images[0]  #pull first image
    n_channels = first_img.shape[2] if first_img.ndim == 3 else 1  #check the number of channels - 1 (grayscale) or 3 (rgb)

    #define the memmap shape for saving the images
    memmap_shape = (
        (N_frames, img_h_rect, img_w_rect, n_channels)
        if n_channels > 1 else
        (N_frames, img_h_rect, img_w_rect)
    )

    #create the memmap
    rectified_memmap = np.memmap(save_path, dtype='uint8', mode='w+', shape=memmap_shape)

    for n in tqdm(range(N_frames), colour='yellow', leave=False): #loop through the frames
        img = norm_images[n]  #pull image
        ir = utils.getPixels(img, DU, DV, s, method='nearest')  #get the pixel values at each rectified location using DU and DV - linear was original
        ir_clean = np.nan_to_num(ir, nan=0.0)  #replace NaNs that are generated by out of bounds DU DV lookups

        if n_channels > 1:  #first check the number of channels - then save the cleaned pixel values to the memmap.
            rectified_memmap[n] = np.clip(ir_clean, 0, 255).astype(np.uint8) # clip to valid 0-255 range before casting to uint8, to avoid overflow/wraparound
        else:
            rectified_memmap[n] = np.clip(ir_clean[:, :, 0], 0, 255).astype(np.uint8)

        if plot:  #plot the rectified image as a sanity check if needed
            plt.figure(figsize=(10, 10))
            if n_channels == 3:
                plt.imshow(np.clip(ir, 0, 255).astype(np.uint8),
                           extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
                           origin='lower')
            else:
                plt.imshow(ir[:, :, 0], cmap='gray',
                           extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
                           origin='lower')
            plt.title("Rectified image")
            plt.xlabel("Easting (m)")
            plt.ylabel("Northing (m)")
            plt.show()
            plt.close('all')

        if n % 100 == 0:  #everytime the index of the loop is a multiple of 100, flush to disk so writes aren't lost if the process crashes mid-run
            rectified_memmap.flush()

    rectified_memmap.flush() # ensure all writes are committed to disk before returning


def save_chunk_metadata(save_dir, chunk_index, memmap_shape, dtype,
                         frame_numbers, core_frame_count,
                        x_grid, y_grid, gsd, overlap):
    """
    Save everything needed to reload and correctly interpret a chunk's memmap.
    """
    meta = {
        'chunk_index': chunk_index,    #index of a particular chunk
        'memmap_shape': list(memmap_shape),     # shape of the memmap - needed to reload it
        'dtype': str(dtype),                     # 'uint8'  - datatype
        'frame_numbers': [int(n) for n in frame_numbers], # original video frame numbers, core + buffer
        'global_frame_start': int(frame_numbers[0]), #video frame number start
        'global_frame_end': int(frame_numbers[-1]) + 1,     # video frame number end (exclusive)
        'core_frame_count': int(core_frame_count),     # how many are "non-overlap" output frames
        'overlap': overlap,  #number of frames included for overlap to deal with optical flow.
        'x_grid_min': float(x_grid.min()),  #bounds of the grid that the rectified images sit on
        'x_grid_max': float(x_grid.max()),
        'y_grid_min': float(y_grid.min()),
        'y_grid_max': float(y_grid.max()),
        'gsd': float(gsd), #ground sampling distance - what each pixel covers in meters
    }

    meta_path = os.path.join(save_dir, f'rectified_chunk_{chunk_index:04d}_meta.json')
    with open(meta_path, 'w') as f:  #write metadata to the json file
        json.dump(meta, f, indent=2)

    return meta_path


def frame_number(path):
    """Video frame number from a filename like '20171791057_1234.png' (last integer in the name)."""
    return int(re.findall(r'\d+', os.path.basename(path))[-1])


def build_chunks(frame_paths, chunk_size=500, overlap=5):
    """
    Split a sorted list of frame paths into chunks that never span a gap in
    frame numbers. Each chunk = up to `chunk_size` core frames + `overlap`
    buffer frames taken from the same contiguous segment.
    Returns a list of dicts with index bounds into frame_paths.
    """
    nums = np.array([frame_number(p) for p in frame_paths])
    breaks = np.where(np.diff(nums) != 1)[0] + 1 # new_segment -> where the index of each image is not spaced by 1. This means that there has been a jump in time and a new chunk must be created
    seg_starts = np.concatenate([[0], breaks])
    seg_ends = np.concatenate([breaks, [len(nums)]])    # exclusive

    chunks = []
    for s, e in zip(seg_starts, seg_ends):
        core_end = e - overlap                          # last `overlap` frames are buffer only
        if core_end <= s:
            print(f'Skipping segment of {e - s} frames ({nums[s]}-{nums[e - 1]}): shorter than overlap ({overlap})')
            continue
        for c0 in range(s, core_end, chunk_size):
            c1 = min(c0 + chunk_size, core_end)
            chunks.append({'start': c0, 'core_end': c1, 'end': c1 + overlap})

    print(f'{len(seg_starts)} contiguous segments -> {len(chunks)} chunks')
    for i, ch in enumerate(chunks):
        print(f"  chunk {i}: frames {nums[ch['start']]}-{nums[ch['end'] - 1]} "
              f"({ch['core_end'] - ch['start']} core + {overlap} buffer)")
    return chunks


def process_chunk(image_paths, save_dir, DU, DV, x_grid, y_grid, gsd, overlap,
                   chunk_index, x_min=None, x_max=None, y_min=700, y_max=1500, plot=False):
    """
    One chunk: background-normalize -> clip -> rectify -> save memmap.
    """
    frame_numbers = [frame_number(p) for p in image_paths]

    print(f'[chunk {chunk_index}] Getting mean background lighting for {len(image_paths)} frames ... This may take a while')
    S, Sbar = get_mean_background(image_paths)  #get the mean background lighting

    if x_min is None or x_max is None:  #check to make sure that x_min and x_max exist
        sample = cv2.imread(image_paths[0])
        x_min, x_max = 0, sample.shape[1]  #if they do not exist pull them from the image shape

    print(f'[chunk {chunk_index}] Normalizing and clipping...')
    norm_images = []  #empty list for normalize images
    for path in tqdm(image_paths, colour='yellow', leave=False):  #loop through all images in the chunk
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  #read the image as grayscale

        img_norm_01 = img / 255.0  #normalize to the 0-1 range
        img_norm = (img_norm_01 / S) * Sbar  #normalize through pixelwise division by mean background lighting. Paper finds this better than subtraction. kleiss 2009 I' = (I/S(x, y)) * Sbar

        clipped = img_norm[y_min:y_max, x_min:x_max] #clip the image to the area of interest

        # scale back to uint8 range for rectification input - 0-255 image
        clipped_uint8 = np.clip(clipped * 255, 0, 255).astype(np.uint8)
        norm_images.append(clipped_uint8) #append normalized uint8 image to the list

    print(f'[chunk {chunk_index}] Rectifying...')

    chunk_memmap_path = os.path.join(save_dir, f'rectified_chunk_{chunk_index:04d}.dat')  #create memmap save path

    rectify_chunk(norm_images, DU, DV, x_grid, y_grid, chunk_memmap_path, plot=plot)  #rectify each image in the chunk

    save_chunk_metadata(  #save the metadata for the chunk to reload later
        save_dir, chunk_index,
        memmap_shape=(len(norm_images), DU.shape[0], DU.shape[1]),
        dtype='uint8',
        frame_numbers=frame_numbers,
        core_frame_count=len(norm_images) - overlap,
        overlap=overlap,
        x_grid=x_grid, y_grid=y_grid, gsd=gsd,
    )
    return chunk_memmap_path


if __name__ == "__main__":

    '''Process Video'''
    video_path = os.path.join(os.getcwd(), "drone", "20171791057.MOV")
    save_path = '/mnt/d/Wave_Breaking/20171791057_raw_images'

    # processVideo(video_path, save_path)

    '''Image Processing'''
    all_frames = sorted(glob.glob(os.path.join(save_path, '*.png')), key=utils.numerical_sort)
    N_frames = len(all_frames) #number of total frames

    save_dir = '/mnt/d/Wave_Breaking/rectified_images'
    os.makedirs(save_dir, exist_ok=True) #save directory for the rectified images

    '''Rectification calibration and grid setup'''
    drone_intrinsics_path = './drone/DJI_3840x2160.mat'
    drone_extrinsics_path = './drone/1057metadata.npy'

    #load the camera intrinsics, extrinsics, and ground sampling distance used for image rectification
    intrinsics, extrinsics, gsd = load_calibration(drone_intrinsics_path, drone_extrinsics_path)

    sample_img = cv2.imread(all_frames[0]) #read sample image to get the original image shape
    img_h, img_w = sample_img.shape[:2]
    # NOTE: if you're clipping to y_min:y_max before rectifying, use the
    # clipped height here instead, since DU/DV must match the clipped frame size
    y_min, y_max = 700, 1500
    x_min, x_max = 0, img_w
    clipped_h, clipped_w = y_max - y_min, x_max - x_min #define the height and width of the clipped image

    #build a grid for the rectified images to populate
    DU, DV, x_grid, y_grid = build_rectification_grid(
        intrinsics, extrinsics, clipped_h, clipped_w, gsd
    )

    '''Process Images in Chunks'''
    chunk_size = 500  #number of frames to put in each chunk - The amount of RAM in the computer determines how large these chunks are able to be. More RAM can do larger chunks
    overlap = 5 #overlap between chunks to help keep optical flow simple

    chunks = build_chunks(all_frames, chunk_size, overlap)

    for i, ch in enumerate(chunks):  #loop through all frames in the chunk
        chunk_paths = all_frames[ch['start']:ch['end']]

        #process the images and save in a memmap
        memmap_path = process_chunk(
            chunk_paths, save_dir, DU, DV, x_grid, y_grid, gsd, overlap,
            chunk_index=i, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, plot=False
        )








