import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import glob
import os
from scipy.optimize import linear_sum_assignment  #shows up as an error sometimes but shouldn't raise one. Maybe deprecation of some sort
from scipy.spatial.distance import cdist
from tqdm import tqdm
from collections import deque

from utils import numerical_sort, load_chunk, loadContours, get_mask



class WaveTracker:
    def __init__(self, fps, image_shape, gsd, phase_speed, min_age=5,
                 max_area_change=0.25, baseline=5):
        self.fps = fps
        self.gsd = gsd
        self.image_shape = image_shape
        self.max_speed = 1.5 * phase_speed  # m/s, physical cap
        self.max_distance = self.max_speed / (fps * gsd)  # px/frame implied by the cap
        self.min_age = min_age  # keep >= baseline. How many frames a track must exist to count towards the lambda distribution
        self.max_area_change = max_area_change  #25% change larger or smaller is allowed
        self.baseline = baseline  # frames used for the speed estimate
        self.prev = []  # [{'hist', 'area', 'age'}]
        self.samples = []  # (speed m/s, crest length m)

    def _fit_ellipses(self, contours, image_shape):
        detections = []
        h, w = image_shape
        for cont in contours:
            cont = np.asarray(cont, dtype=np.float32)

            if len(cont) < 5:
                continue
            if cv2.contourArea(cont) < 50:
                continue

            hull = cv2.convexHull(cont)
            if len(hull) < 8:
                continue
            hull_area = cv2.contourArea(hull)
            if hull_area < 1e-3:
                continue

            ellipse = cv2.fitEllipse(cont)
            x, y = ellipse[0]
            MA, ma = ellipse[1]

            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if not (0 <= x <= w and 0 <= y <= h):
                continue
            # if max(MA, ma) > w * 0.6:
            #     continue
            if ma < 1e-3:
                continue

            # ellipse_area = np.pi * (MA / 2) * (ma / 2)
            # if ellipse_area < 200:
            #     continue
            # if ellipse_area / hull_area > 2.0:   # ellipse much bigger than its hull
            #     continue

            detections.append((np.array([x, y]), ellipse))
        return detections

    def update(self, filtered_contours):
        detections = self._fit_ellipses(filtered_contours, self.image_shape)
        det_area = np.array([np.pi * (e[1][0] / 2) * (e[1][1] / 2) for _, e in detections])
        match = {}

        if self.prev and detections:
            prev_c = np.array([p['hist'][-1] for p in self.prev])
            prev_a = np.array([p['area'] for p in self.prev])
            det_c = np.array([d[0] for d in detections])

            cost = cdist(prev_c, det_c)
            ratio = det_area[None, :] / prev_a[:, None]  # new area / previous area
            bad = (cost > self.max_distance) | (np.abs(ratio - 1) > self.max_area_change)
            cost[bad] = 1e6  # gate before assignment
            rows, cols = linear_sum_assignment(cost)
            match = {j: i for i, j in zip(rows, cols) if cost[i, j] < 1e6}

        curr, out = [], []
        for j, (centroid, ellipse) in enumerate(detections):
            speed = None
            if j in match:
                p = self.prev[match[j]]
                age = p['age'] + 1
                hist = (p['hist'] + [centroid])[-(self.baseline + 1):]
                k = len(hist) - 1  # frames spanned by the history
                speed = np.linalg.norm(hist[-1] - hist[0]) / k * self.fps * self.gsd
                if age >= self.min_age:
                    self.samples.append((speed, max(ellipse[1]) * self.gsd))
            else:
                age = 0
                hist = [centroid]
            curr.append({'hist': hist, 'area': det_area[j], 'age': age})
            out.append((ellipse, age, speed))
        self.prev = curr
        return out

    def sum_dl(self, speed_bins):
        if not self.samples:
            return np.zeros(len(speed_bins) - 1)
        s, l = np.array(self.samples).T
        return np.histogram(s, bins=speed_bins, weights=l)[0]


def plot_tracks(image, detections, frame_idx, chunk, min_age, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')

    for k, (ellipse, age, speed) in enumerate(detections):
        (cx, cy), (MA, ma), angle = ellipse
        color = plt.get_cmap('tab20')(k % 20)
        mature = age >= min_age

        if mature:
            ax.add_patch(Ellipse((cx, cy), MA, ma, angle=angle, fill=False,
                             edgecolor=color, linewidth=1.5,
                             linestyle='-' if mature else ':'))
            if speed is not None:
                ax.text(cx, cy, f'{speed:.1f} m/s', color=color, fontsize=7,
                        ha='center', va='center')

    ax.set_title(f'Chunk {chunk:04d}, frame {frame_idx}  |  {len(detections)} detections')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def compute_lambda(rectified_images_dir, filtered_contours_dir, lambda_save_dir, gsd, fps, phase_speed, speed_bins=0.2, plot=False, plot_every=50, plot_dir=None):

    #determine the number of chunks to iterate through
    n_chunks = len(glob.glob(os.path.join(rectified_images_dir, '*.dat')))

    dc = speed_bins  # speed bin size for lambda computation
    c_bins = np.arange(0, 30 + dc, dc)  # speed bins

    if plot:
        os.makedirs(plot_dir, exist_ok=True)

    Lambda = np.zeros(len(c_bins) - 1)  # set up empty array for the lambda distribution for all chunk
    Area_total = 0.0 #total area for all chunks
    for chunk in tqdm(range(n_chunks), colour='yellow'):
        tqdm.write(f'Processing Chunk: {chunk:04d}')

        # define the image shape of the rectified images
        # sample_chunk_path = os.path.join(rectified_images_dir, os.listdir(rectified_images_dir)[0])  # get the first chunk in the rectified images directory
        chunk_images, chunk_metadata = load_chunk(rectified_images_dir, chunk)  #load the metadata and data for the current chunk
        _, h, w = chunk_metadata['memmap_shape']  #pull the image shape from the metadata
        n_frames = chunk_metadata['core_frame_count'] #number of core frames in the array (no buffer frames that were used for optical flow)

        # get mask for buffer zone around the rectified image
        valid_mask = get_mask(chunk_images[0])
        valid_image = np.where(valid_mask > 0, True, False)
        valid_pixels = np.sum(valid_image)

        A_tot_chunk = valid_pixels * (gsd ** 2) * n_frames  # Atot is the cumulative area over all frames
        Area_total += A_tot_chunk  #add the total area for this chunk to the overall total area

        # instantiate the wave tracking class
        tracker = WaveTracker(fps=fps, gsd=gsd, image_shape=(h, w), phase_speed=phase_speed,
                              min_age=int(fps * (2/3)), max_area_change=0.25, baseline=1)  #int(fps * (2/3))
        #load in the contours for each chunk
        chunk_contours = loadContours(os.path.join(filtered_contours_dir, f'Filtered_contours_chunk_{chunk:04d}.pkl')) #list of shape (n_frames, n_contours_in_frame, n_points_in_conours, (x, y))

        for frame in tqdm(range(n_frames), colour='blue', leave=False, desc='Frame'):

            #load in the contours for each frame
            frame_contours = chunk_contours[frame]  #get the contours for the current frame. shape: (n_contours_in_frame, n_points_in_conours, (x, y))

            detections = tracker.update(frame_contours)

            if plot and frame % plot_every == 0:
                plot_tracks(chunk_images[frame], detections, frame, chunk, tracker.min_age,
                            os.path.join(plot_dir, f'tracks_chunk{chunk:04d}_frame{frame:05d}.png'))

        Lambda += tracker.sum_dl(speed_bins=c_bins)  #lambda distribution per chunk

    Lambda /= (Area_total * dc) #normalize by the total area and bin width to get the final lambda distribution

    c = 0.5 * (c_bins[:-1] + c_bins[1:]) #compute the center of each speed bin for plotting

    b = 7e-5
    rho = 1025 #density of seawater
    g = 9.81 #gravitational acceleration
    fifth_moment = c ** 5 * Lambda * dc #5th moment of the lambda distribution, multiplied by the bin width to account for the discrete nature of the distribution
    epsilon = b * rho * (1 / g) * np.sum(fifth_moment)  #compute the energy dissipation rate using the formula from Phillips (1985)
    print(f'Estimated Energy dissipation rate: {epsilon:.4e} W/m^2')

    # c**-6 reference line
    idx = 20  # pick a reasonable point away from noise
    c_ref = c[idx] # reference bin for the c**-6 line
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
    ax1.set_ylim(1e-8, 1e0)
    ax1.grid(axis='y')
    plt.savefig(os.path.join(lambda_save_dir, 'ellipse_Lambda(c).png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')

    # print(np.trapz(Lambda, c))  #should be around equal for and c bin size

    return Lambda




if __name__ == "__main__":

    rectified_images_dir = '/mnt/d/Wave_Breaking/rectified_images'

    filtered_contours_dir = '/mnt/d/Wave_Breaking/filtered_contours'  # directory for filtered contour chunks

    optical_flow_dir = '/mnt/d/Wave_Breaking/optical_flow'  # directory for optical flow outputs

    fps = 29.97

    # spatial resolution in meters
    drone = np.load('./drone/1057metadata.npy')
    dx = float(drone[-2][1])  # dx and dy are the same value
    gsd = dx  # ground sampling distance

    lambda_save_dir = '/mnt/d/Wave_Breaking/figs'

    plot_dir = '/mnt/d/Wave_Breaking/figs/ellipse'

    peak_period = 3.88 #seconds pulled from the awac datatable for the corresponding day of the data collection
    phase_speed = 9.81 * peak_period / (2 * np.pi) # phase speed in m/s based on the deep water dispersion relation

    Lambda = compute_lambda(rectified_images_dir, filtered_contours_dir, lambda_save_dir, gsd, fps, phase_speed, plot=True, plot_every=25, plot_dir=plot_dir)


#TODO: add kalman filter to help with wave tracking and prediction

