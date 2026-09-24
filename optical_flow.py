import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt

def opticalFlow(images, output_dir, chunk_idx=None, spacing=1, paths=True, plot=False, plot_every=100, plot_dir=None):

    # Get image dimensions from first frame
    if paths:
        first_frame = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
        plt.imshow(first_frame)
    else:
        first_frame = images[0]

    H, W = first_frame.shape
    n_frames = len(images) - spacing  # optical flow is between pairs so n-1 frames

    # Initialize memmaps
    vx_path = os.path.join(output_dir, f'vx_{chunk_idx:04d}.dat')
    vy_path = os.path.join(output_dir, f'vy_{chunk_idx:04d}.dat')

    vx_mm = np.memmap(vx_path, dtype='float32', mode='w+', shape=(n_frames, H, W))
    vy_mm = np.memmap(vy_path, dtype='float32', mode='w+', shape=(n_frames, H, W))

    for n in tqdm(range(n_frames), desc="Computing optical flow", colour='yellow'):

        if paths:
            frame1 = cv2.imread(images[n], cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread(images[n + spacing], cv2.IMREAD_GRAYSCALE)
        else:
            frame1 = images[n]
            frame2 = images[n + spacing]

        #check if frames are floating point or 8 bit images.
        if frame1.dtype == np.float64 or frame1.dtype == np.float32:
            frame1 = ((frame1 - frame1.min()) / (frame1.max() - frame1.min()) * 255).astype(np.uint8)  #min max normalization as to not saturate the brightest pixels
            frame2 = ((frame2 - frame2.min()) / (frame2.max() - frame2.min()) * 255).astype(np.uint8)
        else:
            frame1 = frame1.astype(np.uint8)
            frame2 = frame2.astype(np.uint8)

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

        if plot and n % plot_every == 0:
            visualize_opticalFlow(flow[..., 0], flow[..., 1], frame1, n, chunk_idx, plot_dir)

        # Flush to disk periodically to avoid memory buildup
        if n % 100 == 0:
            vx_mm.flush()
            vy_mm.flush()

    # Final flush
    vx_mm.flush()
    vy_mm.flush()

    # Save shape metadata - required for reloading
    np.save(os.path.join(output_dir, f'flow_shape_{chunk_idx:04d}.npy'), np.array([n_frames, H, W]))


def visualize_opticalFlow(u, v, image, frame_idx, chunk_idx, plot_dir):
    #image dimensions
    h, w = image.shape[0], image.shape[1]

    #visualize optical flow results
    step = 25  # downsample for clarity

    y, x = np.mgrid[0:h:step, 0:w:step]
    u_s = u[::step, ::step]
    v_s = v[::step, ::step]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    ax.quiver(x, y, u_s, v_s, color='red', angles='xy', scale_units='xy', scale=2)

    ax.set_title('Optical Flow (Farneback)')
    ax.invert_yaxis()
    plt.savefig(os.path.join(plot_dir, f'opticalFlow_chunk_{chunk_idx}_frame_{frame_idx}.png'))
    # plt.show()
    plt.close('all')



if __name__ == "__main__":


    rectified_data_dir = '/mnt/d/Wave_Breaking/rectified_images'  # directory of saved rectified images
    rectified_chunk_paths = sorted(glob.glob(os.path.join(rectified_data_dir, '*.dat')), key=utils.numerical_sort)

    save_dir = '/mnt/d/Wave_Breaking/optical_flow' #directory to save the optical flow outputs
    os.makedirs(save_dir, exist_ok=True) #ensure saving directory exists

    output_plot_dir = '/mnt/d/Wave_Breaking/figs/optical_flow'

    spacing = 5 #spacing between frames for optical flow - must be consistent with the buffer/overlap used in the image processing script

    'Compute optical flow for each chunk of rectified images'
    for i, chunk in enumerate(rectified_chunk_paths):

        print(f'Loading Chunk {i:04d}')
        images, meta = utils.load_chunk(rectified_data_dir, i) #load in each chunk of rectified imagery (N_frames, h, w)

        spacing = meta['overlap']  # buffer frames == flow spacing, by construction
        assert len(images) - spacing == meta['core_frame_count'], \
            f"chunk {i}: {len(images)} frames - {spacing} != core_frame_count {meta['core_frame_count']}"

        opticalFlow(images, save_dir, chunk_idx=i, spacing=spacing, paths=False, plot=True, plot_every=25, plot_dir=output_plot_dir)
