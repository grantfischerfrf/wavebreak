import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.interpolate import RegularGridInterpolator
import re
import glob
from tqdm import tqdm
import os
import pickle


def numerical_sort(string):

    parts = re.split(r'(\d+)', string)
    return [int(part) if part.isdigit() else part for part in parts]

def clip_images(images, x_min, x_max, y_min, y_max):
    clipped = images[:, y_min:y_max, x_min:x_max]
    return clipped

def interpolate(X,Y,Z):
    '''
    creates an interpolator
    '''
    from scipy import interpolate
    Xl = list(X.flatten())
    Yl = list(Y.flatten())
    Zl = list(Z.flatten())
    interpolator=interpolate.LinearNDInterpolator(np.array([Xl,Yl]).T,Zl)
    return interpolator

def loadContours(path):
    with open(path, 'rb') as f:
        contour_list = pickle.load(f)
    return contour_list

def smooth_contour(contour, window_size):

    contour = contour.squeeze()  # (N, 1, 2) → (N, 2)  #if the input is cv2 generated contours

    # Smoothing function using Hann window
    hann_window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))
    hann_window /= hann_window.sum()  # Normalize the Hann window

    # Separate x and y coordinates from the contour
    x_vals = contour[:, 0]
    y_vals = contour[:, 1]

    # Pad the contour data to apply the moving window (to handle the edges)
    x_padded = np.pad(x_vals, (window_size // 2,), mode='wrap')
    y_padded = np.pad(y_vals, (window_size // 2,), mode='wrap')

    # Apply the Hann window using convolution
    x_smoothed = np.convolve(x_padded, hann_window, mode='valid')
    y_smoothed = np.convolve(y_padded, hann_window, mode='valid')

    # Combine the smoothed x and y values
    smoothed_contour = np.vstack((x_smoothed, y_smoothed)).T

    return smoothed_contour

def opticalFlow(images, output_dir, spacing=1, paths=True, plot=False):
    os.makedirs(output_dir, exist_ok=True)

    # Get image dimensions from first frame
    if paths:
        first_frame = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    else:
        first_frame = images[0]

    H, W = first_frame.shape
    n_frames = len(images) - spacing  # optical flow is between pairs so n-1 frames

    # Initialize memmaps
    vx_path = os.path.join(output_dir, 'vx.dat')
    vy_path = os.path.join(output_dir, 'vy.dat')

    vx_mm = np.memmap(vx_path, dtype='float32', mode='w+', shape=(n_frames, H, W))
    vy_mm = np.memmap(vy_path, dtype='float32', mode='w+', shape=(n_frames, H, W))

    for n in tqdm(range(n_frames), desc="Computing optical flow"):

        if paths:
            frame1 = cv2.imread(images[n], cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread(images[n + spacing], cv2.IMREAD_GRAYSCALE)
        else:
            frame1 = images[n]
            frame2 = images[n + spacing]

        #check if frames are floating point or 8 bit images.
        if frame1.dtype == np.float64 or frame1.dtype == np.float32:
            # frame1 = (frame1 * 255).astype(np.uint8)
            # frame2 = (frame2 * 255).astype(np.uint8)
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

        if plot:
            visualize_opticalFlow(flow[..., 0], flow[..., 1], frame1)

        # Flush to disk periodically to avoid memory buildup
        if n % 100 == 0:
            vx_mm.flush()
            vy_mm.flush()

    # Final flush
    vx_mm.flush()
    vy_mm.flush()

    # Save shape metadata - required for reloading
    np.save(os.path.join(output_dir, 'flow_shape.npy'), np.array([n_frames, H, W]))

    return vx_mm, vy_mm

def calculateVelocities(vx, vy, fps, spatial_res, spacing):
    '''convert relative optical flow output to metric units in m/s'''

    dt = spacing / fps

    cx = vx * spatial_res / dt
    cy = vy * spatial_res / dt

    cx = np.nan_to_num(cx, copy=True, nan=0.0, posinf=None, neginf=None)
    cy = np.nan_to_num(cy, copy=True, nan=0.0, posinf=None, neginf=None)

    return cx, cy


def filter_lowSpeeds(ipc_raw, jpc_raw, CXi_raw, CYi_raw, dl_raw, slowspeeds=1):

    #slowspeeds is filter in m/s

    speeds = np.sqrt(CXi_raw ** 2 + CYi_raw ** 2)
    index_speeds = np.where(speeds >= slowspeeds)  #filter out slow velocities based on threshold
    ipc, jpc, CXi, CYi, dl = ipc_raw[index_speeds], jpc_raw[index_speeds], CXi_raw[index_speeds], CYi_raw[index_speeds], dl_raw[index_speeds]

    return ipc, jpc, CXi, CYi, dl


def visualize_opticalFlow(u, v, image):
    #image dimensions
    h, w = image.shape[0], image.shape[1]

    #visualize optical flow results
    step = 10  # downsample for clarity

    y, x = np.mgrid[0:h:step, 0:w:step]
    u_s = u[::step, ::step]
    v_s = v[::step, ::step]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    ax.quiver(x, y, u_s, v_s, color='r', angles='xy', scale_units='xy', scale=1)

    ax.set_title('Optical Flow (Farneback)')
    ax.axis('off')
    # plt.savefig()
    plt.show()
    plt.close('all')

def whitecap_flow(cx, cy, images, contours, plot=True):

    h, w = images[0].shape[0], images[0].shape[1]

    J, I = np.meshgrid(np.arange(w), np.arange(h))

    #Lists will have shape (N frames, N contours in a frame, N points in a contour, 1, (x, y)) --- Ex: (277, x, x, 1, 2) --- gives nonhomogeneous shape as x's vary with frames
    all_CXi = []
    all_CYi = []
    all_ipc = []
    all_jpc = []
    all_dl = []

    frame_id = 0

    for n in range(len(cx)):  #loop through frames

        # smooth contours
        win = 5  # window size for Hann window
        smthd_contours = [smooth_contour(cont, win) for cont in contours[n]] #for each individual contour in an image "n" smooth the contours
        # smoothed_contours.append(smthd_contours)
        #
        # if plot:
        #     plt.figure(figsize=(12, 7))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(images[n], cmap='gray')
        #     for contour in contours[n]:
        #         #local variable contour is of shape: (N points in a contour, dummy, (x, y)) --- Ex: (22, 1, 2)
        #         plt.plot(contour.squeeze()[:, 0], contour.squeeze()[:, 1], 'r-', label='Original Contour')
        #     plt.title('Original Contours')
        #     # plt.axis('equal')
        #
        #     # Smoothed contours
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(images[n], cmap='gray')
        #     for smoothed_contour in smthd_contours:
        #         #local variable smoothed_contour is of shape: (N point in a contour, (x, y)) --- Ex: (22, 2)
        #         plt.plot(smoothed_contour[:, 0], smoothed_contour[:, 1], 'b-', label='Smoothed Contour')
        #     plt.title('Smoothed Contours')
        #     # plt.axis('equal')
        #
        #     # plt.tight_layout()
        #     plt.show()
        #     plt.close('all')


        # Initialize empty lists to store interpolated values
        CXi_list = []
        CYi_list = []
        ipc_list = []
        jpc_list = []
        dl_list = []

        # Use linear interpolation for both cx and cy fields
        interp_cx = RegularGridInterpolator((I[:, 0], J[0, :]), cx[n], method='linear')
        interp_cy = RegularGridInterpolator((I[:, 0], J[0, :]), cy[n], method='linear')

        # Iterate over smoothed contours and interpolate them onto cx and cy
        for contour in smthd_contours:
            x_contour, y_contour = contour[:, 0], contour[:, 1]

            # for x_point, y_point in zip(x_contour, y_contour):
            #     # Interpolate cx and cy at each point of the contour
            #     print(x_point, y_point)
            #     cx_val = interp_cx([[y_point, x_point]])[0]  # [0] extracts the scalar value
            #     cy_val = interp_cy([[y_point, x_point]])[0]  # [0] extracts the scalar value
            #
            #     # Store the interpolated values and coordinates
            #     CXi_list.append(cx_val)
            #     CYi_list.append(cy_val)
            #     ipc_list.append(x_point)  # x coordinates (columns)
            #     jpc_list.append(y_point)  # y coordinates (rows)

            # Stack as (N, 2) array of (y, x) pairs
            points = np.column_stack((y_contour, x_contour))

            #compute the length dl that each element covers
            dx = np.diff(np.r_[x_contour, x_contour[0]])
            dy = np.diff(np.r_[y_contour, y_contour[0]])

            dl = np.sqrt(dx ** 2 + dy ** 2) * gsd  #length of breaking crest in meters

            CXi_list.append(interp_cx(points))
            CYi_list.append(interp_cy(points))
            ipc_list.append(x_contour)
            jpc_list.append(y_contour)
            dl_list.append(dl)

        ipc_plot = np.concatenate(ipc_list)
        jpc_plot = np.concatenate(jpc_list)
        CXi_plot = np.concatenate(CXi_list)
        CYi_plot = np.concatenate(CYi_list)
        dl_plot = np.concatenate(dl_list)

        if plot:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), sharex=True, sharey=True)

            ax1.imshow(images[n], cmap='gray')

            scatter1 = ax1.scatter(ipc_plot, jpc_plot, s=30, c=CYi_plot, cmap='RdBu_r', vmin=-4, vmax=4)
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label("CY (m/s)")
            ax1.set_title("CY  along contours " + str(frame_id), fontsize=20)
            ax1.set_aspect('equal')

            ax2.imshow(images[n], cmap='gray')
            scatter2 = ax2.scatter(ipc_plot, jpc_plot, s=30, c=CXi_plot, cmap='RdBu_r', vmin=-4, vmax=4)
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label("CX (m/s)")
            ax2.set_title("CX along contours " + str(frame_id), fontsize=20)
            ax2.set_aspect('equal')

            plt.tight_layout()
            plt.savefig(f'./elemental/figures/cx_and_cy_{frame_id}.png')
            # plt.show()
            plt.close('all')

        frame_id += 1

        # Append this frame's data
        all_ipc.append(ipc_plot)
        all_jpc.append(jpc_plot)
        all_CXi.append(CXi_plot)
        all_CYi.append(CYi_plot)
        all_dl.append(dl_plot)

    # Save — use pickle since frames have different numbers of points
    with open('./elemental/whitecap_flow.pkl', 'wb') as f:
        pickle.dump({'ipc': all_ipc, 'jpc': all_jpc, 'CXi': all_CXi, 'CYi': all_CYi, 'dl':all_dl}, f)


def pull_outward_velocities(CXi_list, CYi_list, ipc_list, jpc_list, dl_list, images, gsd, fps, plot=True):

    #     #inspect the input for the outward velocity calculations
    # if plot:
    #     amp = 3
    #     for n in range(len(images)):
    #         ipc_raw, jpc_raw, CXi_raw, CYi_raw, dl_raw = ipc_list[n], jpc_list[n], CXi_list[n], CYi_list[n], dl_list[n]
    #         ipc, jpc, CXi, CYi, dl = filter_lowSpeeds(ipc_raw, jpc_raw, CXi_raw, CYi_raw, dl_raw, slowspeeds=0.5)
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(images[n], cmap='gray')
    #         for x_start, y_start, dx, dy in zip(ipc, jpc, CXi, CYi):
    #             plt.arrow(x_start, y_start, dx*amp, dy*amp,
    #               length_includes_head=True,
    #               head_length=0.5,
    #               head_width=1,
    #               ec='red',
    #               facecolor='red')
    #         #plt.savefig()
    #         plt.show()
    #         plt.close('all')

    dc = 0.2
    c_bins = np.arange(0, 5 + dc, dc)

    Lambda = np.zeros(len(c_bins) - 1)

    N_frames = len(images)
    A_tot = images[0].shape[0] * images[0].shape[1] * (gsd ** 2) * N_frames  #Atot is the cumulative area

    #find the outward vectors for the waves
    amp = 5 #amplitude factor for quiver plot
    for n in range(len(images)): #loop through images

        ipc_raw, jpc_raw, CXi_raw, CYi_raw, dl_raw = ipc_list[n], jpc_list[n], CXi_list[n], CYi_list[n], dl_list[n] #Unpack contour coordinate locations and velocity values for a frame "n"

        #filter out slow velocities
        ipc, jpc, CXi, CYi, dl = filter_lowSpeeds(ipc_raw, jpc_raw, CXi_raw, CYi_raw, dl_raw, slowspeeds=0.0)  #TODO: can use a histogram knee point to find the ideal speed to set the threshold at - 0.6 is good starting point

        dXipc = list() #create empty lists for data
        dYjpc = list()

        # Check if jpc has elements
        if len(jpc) > 0:
            # Difference the y contour points
            dYjpc = jpc[1:len(jpc)] - jpc[0:len(jpc) - 1]
            tmpy = jpc[0] - jpc[len(jpc) - 1] #close loop by differencing the first and last point
            dYjpc = np.append(dYjpc, tmpy)  # appending looped values during differentiation

        # Check if ipc has elements
        if len(ipc) > 0:
            # Difference the x contour points
            dXipc = ipc[1:len(ipc)] - ipc[0:len(ipc) - 1]
            tmpx = ipc[0] - ipc[len(ipc) - 1] #close loop by differencing the first and last point
            dXipc = np.append(dXipc, tmpx)

        # print('Finding the outward vectors')

        # normal outward vector to the contours
        nCX = -1 * (dYjpc)  #if the tangent vector is obtained by the differencing, then rotating it 90 degrees will give you the outward pointing vector
        nCY = dXipc  #this is the 2D perpendicular projection to normal

        #take the dot product of the flow velocity vector (CX or CY) against the normal vector (nCX, nCY). If they are in the same direction: +, if opposite direction: -
        nxp = nCX * CXi + nCY * CYi  # project our velocities (as evaluated on the contours) onto the normal vectors
        #The outward vector direction depends on the winding direction for contour retrieval. CW winding: nxp > 0, CCW winding: nxp < 0
        # find the projection that is less than zero == outward vector
        outward = nxp > 0
        CXo = CXi[outward]
        CYo = CYi[outward]
        co = np.sqrt(CXo ** 2 + CYo ** 2)

        dl_out = dl[outward]

        for k in range(len(c_bins) - 1):
            mask = (co >= c_bins[k]) & (co < c_bins[k + 1])

            # Lambda[k] += np.sum(dl_out[mask]) * 5/fps  #TODO: do I need a factor of time here?
            Lambda[k] += np.sum(dl_out[mask])

        #plot outward facing vectors
        if plot:
            ipo = ipc[outward]
            jpo = jpc[outward]

            plt.figure(figsize=(10, 5))
            plt.imshow(images[n], cmap='gray')
            for x, y, cx, cy in zip(ipo, jpo, CXo, CYo):
                plt.arrow(
                    x, y,
                    cx * amp,
                    cy * amp,
                    length_includes_head=True,
                    head_length=0.5,
                    head_width=1,
                    ec='red',
                    facecolor='red'
                )

            # plt.savefig()
            plt.show()
            plt.close('all')


    Total_time_sec = len(images) / fps  #total time in seconds

    # Lambda /= (A_tot * Total_time_sec * dc)  #TODO: Is total time needed in the normalization?
    Lambda /= (A_tot * dc)

    c = 0.5 * (c_bins[:-1] + c_bins[1:])

    b = 7e-5
    rho = 1025
    g = 9.81
    fifth_moment = c ** 5 * Lambda * dc
    epsilon = b * rho * (1 / g) * np.sum(fifth_moment)
    print(f'Estimated Energy dissipation rate: {epsilon:.4e} W/m^2')

    # c**-6 reference line
    idx = 20  # pick a reasonable point away from noise
    c_ref = c[idx]
    Lambda_ref = Lambda[idx]
    Lambda_c6 = Lambda_ref * (c / c_ref) ** (-6)

    # plot lambda distribution - for the entire image and time period
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121)
    ax.loglog(c, Lambda)
    ax.loglog(c, Lambda_c6, '--', label=r'$c^{-6}$')
    ax.set_xlabel('Wave Speed (m/s)')
    ax.set_ylabel(r'$\Lambda(c)$ (s * m$^{-2}$)')  # units of m^-2, time is absorbed in the calculate lambda function
    ax.set_title('\u039B(c) Distribution of Breaking Waves')
    ax.set_ylim(1e-8, 1e0)
    ax.legend()
    ax.grid(axis='y')

    ax1 = fig.add_subplot(122)
    ax1.loglog(c, fifth_moment, color='salmon')
    ax1.set_xlabel('Wave Speed (m/s)')
    ax1.set_ylabel(r'c$^5$ * $\Lambda(c)$ * dc (m$^4$ s$^{-5}$)')
    ax1.set_title('Fifth Moment of the \u039B(c) Distribution')
    ax1.grid(axis='y')
    plt.savefig('./elemental/figures/elemental_Lambda(c).png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')

    # print(np.trapz(Lambda, c))  #should be around equal for and c bin size

    return Lambda





if __name__ == "__main__":

    #import corrected images
    images = np.load('./drone/processed/1057subset_images.npy', mmap_mode='r')[:282]

    #clip images to the area of interest
    x_min, x_max = 0, images[0].shape[1]
    y_min, y_max = 700, 1500
    clipped_images = clip_images(images, x_min, x_max, y_min, y_max)

    #calculate optical flow using farneback method
    output_dir = './elemental/opticalFlow/'
    spacing = 5
    # opticalFlow(clipped_images, output_dir, spacing=spacing, paths=False, plot=True)

    # load optical flow output
    shape = tuple(np.load(os.path.join(output_dir, 'flow_shape.npy')))
    vx_path = './elemental/opticalFlow/vx.dat'
    vy_path = './elemental/opticalFlow/vy.dat'
    vx = np.memmap(vx_path, dtype='float32', mode='r', shape=shape) #in relative units of pixels/frame
    vy = np.memmap(vy_path, dtype='float32', mode='r', shape=shape)

    # fps
    fps = 29.97

    # spatial resolution in meters
    drone = np.load('./drone/1057metadata.npy')
    dx = float(drone[-2][1])  # dx and dy are the same value
    gsd = dx  # ground sampling distance

    #load in contours from image processing - of shape (num_Images, num_contours, points per contour, dummy, tuple of (x, y))
    # filtered_contours = [loadContours(file) for file in sorted(glob.glob(f'./drone/processed/subset20171791057_contours_filtered/*.pkl'), key=numerical_sort)]#[::spacing]

    #calculate velocities
    # cx, cy = calculateVelocities(vx, vy, fps, gsd, spacing)
    #cx is positive for motion to the right(increasing column index) and negative for motion to the left
    #cy is positive for motion downward (increasing row index) and negative for motion upward

    #visualize the results
    # images_downsampled = images[::spacing] #downsample to match spacing used for optical flow

    # whitecap_flow(cx, cy, clipped_images, filtered_contours, plot=False)

    #load flow data
    flow_variables = np.load('./elemental/whitecap_flow.pkl', allow_pickle=True) #For each variable (N Frames, N contours in each frame, N points in each contour, 1, (x, y))
    CXi = flow_variables['CXi'] #X velocities per frame per contour
    CYi = flow_variables['CYi'] #Y velocities per frame per contour
    ipc = flow_variables['ipc'] # X coordinates for each contour per frame per contour
    jpc = flow_variables['jpc'] # Y coordinates for each contour per frame per contour
    dl = flow_variables['dl'] # length of breaking crests

    Lambda = pull_outward_velocities(CXi, CYi, ipc, jpc, dl, clipped_images[:277], gsd, fps, plot=False)



    #calculate the lambda c distribution - dcx, dcy speed bins are 0.2m/s from akawaase paper
    # the distribution seems to have a lack of higher speed data.

    #TODO: make sure the speed filters are equivalent between the two methods - this way the comparison between the distributions can be done
    #TODO: next up is to first reorganize the code. Then get the energy dissipation per unit area



