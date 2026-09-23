import os
import glob
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from contours import generate_contours_parallel, filter_contours
import utils


if __name__ == "__main__":


    rectified_data_dir = '/mnt/d/Wave_Breaking/rectified_images'  # directory of saved rectified images
    save_dir = '/mnt/d/Wave_Breaking/unfiltered_contours'  # directory to save contours in
    os.makedirs(save_dir, exist_ok=True)    # check if save directory exists - create the directory if it doesn't exist

    # count how many chunks exist — used only to know how many indices to dispatch
    n_chunks = len(glob.glob(os.path.join(rectified_data_dir, '*.dat')))

    worker = partial(
        generate_contours_parallel,
        data_dir=rectified_data_dir,
        save_dir=save_dir,
        sanity_plot_dir='/mnt/d/Wave_Breaking/figs/unfiltered_contours',
        VARIANCE_THRESH=200,
        GRADIENT_THRESH=0.5,
        plot_every=25,
    )

    with Pool(processes=4) as pool:
        list(
            tqdm(
                pool.imap_unordered(worker, range(n_chunks)),
                total=n_chunks, colour='yellow', desc='Generating Contours'
            )
        )

    unfiltered_contours_dir = '/mnt/d/Wave_Breaking/unfiltered_contours'
    filtered_contours_save_dir = '/mnt/d/Wave_Breaking/filtered_contours'
    optical_flow_dir = '/mnt/d/Wave_Breaking/optical_flow'

    plot_dir = '/mnt/d/Wave_Breaking/figs/filtered_contours'  # directory to save plots of filtered contours

    filter_contours(unfiltered_contours_dir, optical_flow_dir, filtered_contours_save_dir, variance_thresh=0.2, plot=True, plot_interval=25, rectified_data_dir=rectified_data_dir, plot_dir=plot_dir)



