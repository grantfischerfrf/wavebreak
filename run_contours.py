import os
import glob
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_triangle

from contours import generate_contours_parallel, filter_contours_parallel, filter_contours, speed_thresh_parallel
import utils
from config import Config

def run_speed_thresh(contour_dir, optical_flow_dir, save_dir, plot_dir, config, n_cores=1):

    n_chunks = len(glob.glob(os.path.join(contour_dir, '*.pkl'))) #number of chunks in the unfiltered contours folder (data_dir)

    worker = partial(
        speed_thresh_parallel,
        contour_dir=contour_dir,
        optical_flow_dir=optical_flow_dir,
        save_dir=save_dir,
        plot_dir=plot_dir,
        config=config
    )

    print('Finding Speed Threshold...')
    with Pool(processes=n_cores) as pool:
        outputs = list(
            pool.imap_unordered(worker, range(n_chunks))
        )

    #combine speed histogram from all workers/cpu cores
    final_hist = np.sum([output[0] for output in outputs], axis=0) #pulls this histogram from each chunk and sums it
    total_count = final_hist.sum()

    c_bins = outputs[0][1] #bin edges

    # # determine speed threshold - x * np.std(hist) - outliers above 96% of the data will be removed
    c = 0.5 * (c_bins[:-1] + c_bins[1:]) #define the bin centers
    hist_mean = np.sum(final_hist * c) / total_count #weighted mean from the histogram count
    hist_variance = np.sum(final_hist * (c - hist_mean) ** 2) / total_count #variance of histogram
    hist_std = np.sqrt(hist_variance) #standard deviation of histogram
    speed_threshold = hist_mean + 3 * hist_std #define the speed threshold
    #
    # threshold_idx = utils.triangle_threshold_1d(final_hist)
    # speed_threshold = c[threshold_idx]
    np.save(os.path.join(save_dir, '3std_mean_speed_threshold.npy'), speed_threshold)

    # plot final histogram
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes.set_title('Distribution of Contour mean Speeds - All Frames')
    axes.set_xlabel('Mean Contour Speed (m/s)')
    axes.set_ylabel('Count')
    axes.stairs(final_hist, c_bins)  #axes.stairs plots precomputed histograms
    axes.axvline(speed_threshold, linestyle='--', color='red', label=f'Speed Threshold: {speed_threshold:.3f} (m/s)')
    axes.grid(axis='y')
    axes.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '3std_mean_speed_thresh_histogram.png'))
    plt.show()
    plt.close('all')

    return speed_threshold


#TODO: may need to do this pixel by pixel


if __name__ == "__main__":

    config = Config()  #load configuration parameters

    '''unfiltered contours'''
    # rectified_data_dir = '/mnt/d/Wave_Breaking/rectified_images'  # directory of saved rectified images
    # save_dir = '/mnt/d/Wave_Breaking/unfiltered_contours'  # directory to save contours in
    # os.makedirs(save_dir, exist_ok=True)    # check if save directory exists - create the directory if it doesn't exist
    #
    # # count how many chunks exist — used only to know how many indices to dispatch
    # n_chunks = len(glob.glob(os.path.join(rectified_data_dir, '*.dat')))
    #
    # worker = partial(
    #     generate_contours_parallel,
    #     data_dir=rectified_data_dir,
    #     save_dir=save_dir,
    #     sanity_plot_dir='/mnt/d/Wave_Breaking/figs/unfiltered_contours',
    #     VARIANCE_THRESH=200,
    #     GRADIENT_THRESH=0.5,
    #     plot_every=25,
    # )
    #
    # with Pool(processes=2) as pool:
    #     list(
    #         tqdm(
    #             pool.imap_unordered(worker, range(n_chunks)),
    #             total=n_chunks, colour='yellow', desc='Generating Contours'
    #         )
    #     )

    # '''speed_thresh'''
    # contour_dir = '/mnt/d/Wave_Breaking/unfiltered_contours'
    # optical_flow_dir = '/mnt/d/Wave_Breaking/optical_flow'
    # save_dir = '/mnt/d/Wave_Breaking/thresholds'
    # plot_dir = '/mnt/d/Wave_Breaking/figs'
    # run_speed_thresh(contour_dir, optical_flow_dir, save_dir, plot_dir, config, n_cores=6) #determine speed threshold between real waves and sun glint / bad tracking

    '''filtered contours'''
    unfiltered_contours_dir = '/mnt/d/Wave_Breaking/unfiltered_contours'
    filtered_contours_save_dir = '/mnt/d/Wave_Breaking/filtered_contours'
    optical_flow_dir = '/mnt/d/Wave_Breaking/optical_flow'
    plot_dir = '/mnt/d/Wave_Breaking/figs/filtered_contours'  # directory to save plots of filtered contours
    rectified_data_dir = '/mnt/d/Wave_Breaking/rectified_images'

    os.makedirs(filtered_contours_save_dir, exist_ok=True)

    #speed threshold
    speed_thresh = np.load('/mnt/d/Wave_Breaking/thresholds/3std_median_speed_threshold.npy')

    n_chunks = len(glob.glob(os.path.join(unfiltered_contours_dir, '*.pkl'))) #number of chunks to process

    #define worker for multiprocessing
    worker = partial(
        filter_contours_parallel,
        unfiltered_contours_dir=unfiltered_contours_dir,
        optical_flow_dir=optical_flow_dir,
        save_dir = filtered_contours_save_dir,
        area_thresh=config.area_thresh,
        speed_thresh=speed_thresh,
        variance_thresh=0.2,
        plot=True,
        plot_interval=25,
        rectified_data_dir=rectified_data_dir,
        plot_dir=plot_dir,
        config=config
    )

    print('Filtering Contours...')
    with Pool(processes=4) as pool:
        list(
            pool.imap_unordered(worker, range(n_chunks))
        )




