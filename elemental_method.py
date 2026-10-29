import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os
import pickle
from config import Config

import utils

def filter_Speeds(ipc_raw, jpc_raw, CXi_raw, CYi_raw, dl_raw, maxSpeed, slowspeeds=1.0):

    #slowspeeds is filter in m/s

    speeds = np.sqrt(CXi_raw ** 2 + CYi_raw ** 2)
    index_speeds = (speeds >= slowspeeds) & (speeds <= maxSpeed)  #filter out slow velocities based on threshold
    ipc, jpc, CXi, CYi, dl = ipc_raw[index_speeds], jpc_raw[index_speeds], CXi_raw[index_speeds], CYi_raw[index_speeds], dl_raw[index_speeds]

    return ipc, jpc, CXi, CYi, dl


def whitecap_flow(filtered_contours_dir, optical_flow_dir, rectified_images_dir, save_dir, fps, gsd, spacing=5, plot=False, plot_every=50):

    #loop through each data chunk and compute the velocities of each element in a detected whitecap
    n_chunks = len(sorted(glob.glob(os.path.join(filtered_contours_dir, '*.pkl')), key=utils.numerical_sort)) #number of chunks to process

    global_img_index = 0
    for n in tqdm(range(n_chunks), colour='yellow'):
        tqdm.write(f'Processing Chunk {n:04d}')

        #load filtered contour data
        filtered_contours = utils.loadContours(os.path.join(filtered_contours_dir, f'Filtered_contours_chunk_{n:04d}.pkl'))

        #pull images for the corresponding frame
        images, meta = utils.load_chunk(rectified_images_dir, n)

        # get mask for buffer zone around the rectified image
        valid_mask = utils.get_mask(images[0])
        valid_image = np.where(valid_mask > 0, True, False)

        h, w = images[0].shape[0], images[0].shape[1]  #height and width of the image
        J, I = np.meshgrid(np.arange(w), np.arange(h)) #meshgrid to interpolate the contours onto

        # Lists will have shape (N frames, N contours in a frame, N points in a contour, (x, y)) --- Ex: (277, x, x, 2) --- gives nonhomogeneous shape as x's vary with frames
        all_CXi = []  #x velocities along the contour
        all_CYi = []  # y velocities along the contour
        all_ipc = [] # x coordinates of the contours
        all_jpc = [] # y coordinates of the contours
        all_dl = []  #infinitessimal length of breaking crests -> Euclidean distance between two points along a contour

        # load optical flow output -> this is often too large of an array to load into RAM - easier to load incrementally
        vx, vy, shape = utils.load_opticalFlow(optical_flow_dir, chunk_idx=n)  # optical flow x and y component velocites for each rectified frame -> shape (h, w)

        n_frames = meta['core_frame_count']
        for f in tqdm(range(n_frames), colour='blue', desc='Frame', leave=False): #loop through the contours in a frame

            # remove background current (e.g. estuarine front) before converting to physical speeds
            # vx_detrend = vx - np.median(vx[valid_image]) #TODO: fine for a first pass, may need a spatially varying detrend later if only certain patches are being removed
            # vy_detrend = vy - np.median(vy[valid_image])  #FIXME: is this necessary with no substantial camera drift? removing this may switch the direction of vectors making it difficult to isloate only the breaking face
            cx, cy = utils.relativeVel_2_metricVel(vx[f, :, :], vy[f, :, :], fps, gsd, spacing) # metric velocities of optical flow -> shape (h, w) of rectified image

            # Initialize empty lists to store interpolated values
            CXi_list = []
            CYi_list = []
            ipc_list = []
            jpc_list = []
            dl_list = []

            # Use linear interpolation for both cx and cy fields
            interp_cx = RegularGridInterpolator((I[:, 0], J[0, :]), cx, method='linear')
            interp_cy = RegularGridInterpolator((I[:, 0], J[0, :]), cy, method='linear')

            frame_contours = filtered_contours[f] #get the particular contours for a certain frame
            for contour in frame_contours:  #loop through individual (N_points, 2) contour arrays
                x_contour, y_contour = contour[:, 0], contour[:, 1]  #pull the x and y points from the contour

                # Stack as (N, 2) array of (y, x) pairs
                points = np.column_stack((y_contour, x_contour))

                # compute the length dl that each element covers
                dx = np.diff(np.r_[x_contour, x_contour[0]])
                dy = np.diff(np.r_[y_contour, y_contour[0]])

                dl = np.sqrt(dx ** 2 + dy ** 2) * gsd  # length of breaking crest in meters

                CXi_list.append(interp_cx(points))
                CYi_list.append(interp_cy(points))
                ipc_list.append(x_contour)
                jpc_list.append(y_contour)
                dl_list.append(dl)

            if not ipc_list: #protect against frames with no wave breaking - no contours in the list
                continue

            ipc_plot = np.concatenate(ipc_list)
            jpc_plot = np.concatenate(jpc_list)
            CXi_plot = np.concatenate(CXi_list)
            CYi_plot = np.concatenate(CYi_list)
            dl_plot = np.concatenate(dl_list)


            if plot and f % plot_every == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), sharex=True, sharey=True)

                ax1.imshow(images[f], cmap='gray')

                scatter1 = ax1.scatter(ipc_plot, jpc_plot, s=30, c=CYi_plot, cmap='RdBu_r', vmin=-4, vmax=4)
                cbar1 = plt.colorbar(scatter1, ax=ax1)
                cbar1.set_label("CY (m/s)")
                ax1.set_title("CY  along contours, frame: " + str(f), fontsize=20)
                ax1.set_aspect('equal')

                ax2.imshow(images[f], cmap='gray')
                scatter2 = ax2.scatter(ipc_plot, jpc_plot, s=30, c=CXi_plot, cmap='RdBu_r', vmin=-4, vmax=4)
                cbar2 = plt.colorbar(scatter2, ax=ax2)
                cbar2.set_label("CX (m/s)")
                ax2.set_title("CX along contours, frame:" + str(f), fontsize=20)
                ax2.set_aspect('equal')

                plt.tight_layout()
                plt.savefig(f'/mnt/d/Wave_Breaking/figs/cx_cy/{n}_cx_and_cy_{f}.png')
                # plt.show()
                plt.close('all')

            # Append this frame's data
            all_ipc.append(ipc_plot)
            all_jpc.append(jpc_plot)
            all_CXi.append(CXi_plot)
            all_CYi.append(CYi_plot)
            all_dl.append(dl_plot)

        # Save — use pickle since frames have different numbers of points
        save_path = os.path.join(save_dir, f'whitecap_flow_{n:04d}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({'ipc': all_ipc, 'jpc': all_jpc, 'CXi': all_CXi, 'CYi': all_CYi, 'dl': all_dl}, f)

def compute_Lambda(whitecap_chunk_dir, rectified_images_dir, lambda_save_dir, gsd, phase_speed, speed_bins=0.2, plot=False, plot_every=100):

    n_chunks = sorted(glob.glob(os.path.join(whitecap_chunk_dir, '*.pkl')), key=utils.numerical_sort)

    dc = speed_bins #speed bin size for lambda computation
    c_bins = np.arange(0, 30 + dc, dc) #speed bins

    total_area = 0.0 #total cumulative area over all chunks
    Lambda = np.zeros(len(c_bins) - 1)  # set up empty array for the lambda distribution for each chunk
    for n in tqdm(range(len(n_chunks)), colour='yellow'):
        tqdm.write(f'Processing Chunk {n:04d}')
        # n = 1

        #pull the whitecap flow data for the matching chunk
        flow_variables = np.load(os.path.join(whitecap_chunk_dir, f'whitecap_flow_{n:04d}.pkl'), allow_pickle=True)
        CXi_list = flow_variables['CXi']  # X velocities per frame per contour
        CYi_list = flow_variables['CYi']  # Y velocities per frame per contour
        ipc_list = flow_variables['ipc']  # X coordinates for each contour per frame per contour
        jpc_list = flow_variables['jpc']  # Y coordinates for each contour per frame per contour
        dl_list = flow_variables['dl']  # length of breaking crests

        #pull the rectified imagery for the matching chunk
        images, meta = utils.load_chunk(rectified_images_dir, n)
        # extent = (meta['x_grid_min'], meta['x_grid_max'], meta['y_grid_min'], meta['y_grid_max']) #extents of the rectified image in meters world coordinates

        #get mask for buffer zone around the rectified image
        valid_mask = utils.get_mask(images[0])
        valid_image = np.where(valid_mask > 0, True, False)
        valid_pixels = np.sum(valid_image)

        n_frames = meta['core_frame_count']
        A_tot_chunk = valid_pixels * (gsd ** 2) * n_frames  # Atot is the cumulative area over all frames
        total_area += A_tot_chunk #append the area from the chunk to the total area over all chunks

        amp = 5  # amplitude factor for quiver plot
        for f in range(len(CXi_list)):  # loop through frames - not using n_frames variable because not all frames have valid contours left on them
            # f = 400

            ipc_raw, jpc_raw, CXi_raw, CYi_raw, dl_raw = ipc_list[f], jpc_list[f], CXi_list[f], CYi_list[f], dl_list[f]  # Unpack contour coordinate locations and velocity values for a frame "n"

            # filter out slow velocities
            ipc, jpc, CXi, CYi, dl = filter_Speeds(ipc_raw, jpc_raw, CXi_raw, CYi_raw, dl_raw, maxSpeed=1.5*phase_speed, slowspeeds=0.0)  #TODO: find a good cutoff for this (maybe in literature) - 0.6 is ok for now

            dXipc = list()  # create empty lists for data
            dYjpc = list()

            # Check if jpc has elements
            if len(jpc) > 0:
                # Difference the y contour points
                dYjpc = jpc[1:len(jpc)] - jpc[0:len(jpc) - 1]
                tmpy = jpc[0] - jpc[len(jpc) - 1]  # close loop by differencing the first and last point
                dYjpc = np.append(dYjpc, tmpy)  # appending looped values during differentiation

            # Check if ipc has elements
            if len(ipc) > 0:
                # Difference the x contour points
                dXipc = ipc[1:len(ipc)] - ipc[0:len(ipc) - 1]
                tmpx = ipc[0] - ipc[len(ipc) - 1]  # close loop by differencing the first and last point
                dXipc = np.append(dXipc, tmpx)

            # print('Finding the outward vectors')

            # normal outward vector to the contours
            nCX = -1 * (dYjpc)  # if the tangent vector is obtained by the differencing, then rotating it 90 degrees will give you the outward pointing vector
            nCY = dXipc  # this is the 2D perpendicular projection to normal

            # take the dot product of the flow velocity vector (CX or CY) against the normal vector (nCX, nCY). If they are in the same direction: +, if opposite direction: -
            nxp = nCX * CXi + nCY * CYi  # project our velocities (as evaluated on the contours) onto the normal vectors
            # The outward vector direction depends on the winding direction for contour retrieval. CW winding: nxp > 0, CCW winding: nxp < 0
            # find the projection that is less than zero == outward vector
            outward = nxp > 0
            CXo = CXi[outward]
            CYo = CYi[outward]
            co = np.sqrt(CXo ** 2 + CYo ** 2)

            dl_out = dl[outward]

            for k in range(len(c_bins) - 1):  # loop through speed bins
                mask = (co >= c_bins[k]) & (co < c_bins[k + 1])
                Lambda[k] += np.sum(dl_out[mask])

            # plot outward facing vectors
            if plot and f % plot_every ==0:
                ipo = ipc[outward]
                jpo = jpc[outward]
                fig, axes = plt.subplots(1, 1, figsize=(10, 5))
                axes.imshow(images[f], cmap='gray')  #extents cannot be used here until the quiver arrows are converted to world coordinates.
                axes.invert_yaxis()
                axes.set_xticks([])
                axes.set_yticks([])
                for x, y, cx, cy in zip(ipo, jpo, CXo, CYo):
                    axes.arrow(
                        x, y,
                        cx * amp,
                        cy * amp,
                        length_includes_head=True,
                        head_length=0.5,
                        head_width=1,
                        ec='red',
                        facecolor='red'
                    )

                plt.savefig(os.path.join(lambda_save_dir, f'quiver_chunk{n}_frame{f}.png'), dpi=300, bbox_inches='tight')
                # plt.show()
                plt.close('all')

    Lambda /= (total_area * dc)

    c = 0.5 * (c_bins[:-1] + c_bins[1:])

    b = 7e-5
    rho = 1025
    g = 9.81
    fifth_moment = c ** 5 * Lambda * dc
    epsilon = b * rho * (1 / g) * np.sum(fifth_moment)
    print(f'Estimated Energy dissipation rate: {epsilon:.4e} W/m^2')

    # c**-6 reference line
    idx = 30  # pick a reasonable point away from noise
    c_ref = c[idx]
    Lambda_ref = Lambda[idx]
    Lambda_c6 = Lambda_ref * (c / c_ref) ** (-6)

    # plot lambda distribution - for the entire image and time period
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121)
    ax.loglog(c, Lambda)
    ax.loglog(c, Lambda_c6, '--', label=r'$c^{-6}$')
    ax.set_xlabel('Breaking Wave Speed (m/s)')
    ax.set_ylabel(r'$\Lambda(c)$ (s * m$^{-2}$)')  # units of m^-2, time is absorbed in the calculate lambda function
    ax.set_title('\u039B(c) Distribution of Breaking Waves')
    ax.set_ylim(1e-8, 1e0)
    ax.legend()
    ax.grid(axis='y')

    ax1 = fig.add_subplot(122)
    ax1.loglog(c, fifth_moment, color='salmon')
    ax1.set_xlabel('Breaking Wave Speed (m/s)')
    ax1.set_ylabel(r'c$^5$ * $\Lambda(c)$ * dc (m$^4$ s$^{-5}$)')
    ax1.set_title('Fifth Moment of the \u039B(c) Distribution')
    ax1.text(0.05, 0.95, s=f'ε: {epsilon:.4e} W/m²')
    ax1.set_ylim(1e-8, 1e0)
    ax1.grid(axis='y')
    plt.savefig(os.path.join(lambda_save_dir, 'elemental_Lambda(c).png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')

    # print(np.trapz(Lambda, c))  #should be around equal for and c bin size

    return Lambda







if __name__ == "__main__":

    config = Config() #load configuration parameters

    rectified_images_dir = '/mnt/d/Wave_Breaking/rectified_images'

    filtered_contours_dir = '/mnt/d/Wave_Breaking/filtered_contours'     #directory for filtered contour chunks

    optical_flow_dir = '/mnt/d/Wave_Breaking/optical_flow'  #directory for optical flow outputs

    peak_period = 3.88 #seconds pulled from the awac datatable for the corresponding day of the data collection
    phase_speed = 9.81 * peak_period / (2 * np.pi) # phase speed in m/s based on the deep water dispersion relation

    whitecap_save_dir = '/mnt/d/Wave_Breaking/20171791057_whitecap_flow'
    # whitecap_flow(filtered_contours_dir, optical_flow_dir, rectified_images_dir, whitecap_save_dir, config.fps, config.gsd, config.spacing, plot=True, plot_every=25)

    lambda_save_dir = '/mnt/d/Wave_Breaking/figs/quiver_all'
    compute_Lambda(whitecap_save_dir, rectified_images_dir, lambda_save_dir, config.gsd, phase_speed, speed_bins=config.speed_bin_width, plot=True, plot_every=1)


#TODO: rewrite and add comments to the image processing.py and all other important scripts.

#TODO: still some false positives on the edge of the screen here.




