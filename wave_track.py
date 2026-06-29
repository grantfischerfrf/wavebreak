import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import glob
import os
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import re
# import torch
# from resnet import UNet
from tqdm import tqdm
from collections import deque
import pickle



def numerical_sort(string):

    parts = re.split(r'(\d+)', string)
    return [int(part) if part.isdigit() else part for part in parts]

def loadContours(path):
    with open(path, 'rb') as f:
        contour_list = pickle.load(f)
    return contour_list

def consolodate_images(path, savepath):

    #save all images as a file to allow memory mapping
    files = sorted(glob.glob(path + '/*.npy'), key=numerical_sort)

    # Load one file to get shape/dtype
    sample = np.load(files[0])
    shape = (len(files),) + sample.shape
    dtype = sample.dtype

    # Create output file on disk - not RAM
    print('saving to disk...')
    stack = np.lib.format.open_memmap(
        f'{savepath}',
        mode='w+',
        dtype=dtype,
        shape=shape
    )

    # Fill it incrementally
    for i, f in enumerate(tqdm(files)):
        stack[i] = np.load(f)  # loads one file at a time

    # Flush
    stack.flush()
    del stack

    #this will take a while to write
    print('saved')

def clipContours(all_frames, x_min, x_max, y_min, y_max):

    clipped_all = [
        [cont[(cont[:, 0, 0] >= x_min) & (cont[:, 0, 0] < x_max) &
              (cont[:, 0, 1] >= y_min) & (cont[:, 0, 1] < y_max)].copy()
         for cont in frame if np.any(
            (cont[:, 0, 0] >= x_min) & (cont[:, 0, 0] < x_max) &
            (cont[:, 0, 1] >= y_min) & (cont[:, 0, 1] < y_max)
        )]
        for frame in all_frames
    ]

    return clipped_all


class WaveTrack:
    _id_counter = 0

    def __init__(self, centroid, ellipse, frame_idx, fps, gsd):
        WaveTrack._id_counter += 1
        self.id = WaveTrack._id_counter
        self.fps = fps
        self.gsd = gsd
        self.dt = 1 / fps

        self.centroids = deque(maxlen=15) #x, y coordinates of the centroid of the wave in each frame. creates a moving window of the last 15 frames
        self.ellipses = deque(maxlen=15) #parameters of the fitted ellipse in each frame. creates a moving window of the last 15 frames
        self.areas = deque(maxlen=15)  # track ellipse areas
        self.birth_frame = frame_idx #the frame index where the wave was first detected
        self.missed = 0 #track number of frames where the wave was not detected to allow for temporary occlusions or missed detections

        self.centroids.append(centroid) #add the initial centroid to the deque
        self.ellipses.append(ellipse) #add the initial ellipse parameters to the deque
        self.areas.append(self._ellipse_area(ellipse))

    def update(self, centroid, ellipse):
        self.centroids.append(centroid)
        self.ellipses.append(ellipse)
        self.areas.append(self._ellipse_area(ellipse))
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    def _ellipse_area(self, ellipse):
        #Calculate ellipse area
        (cx, cy), (MA, ma), angle = ellipse
        return np.pi * (MA / 2) * (ma / 2)

    def overlaps_with(self, new_ellipse, overlap_threshold=0.0):
        """
        Check if new_ellipse spatially overlaps with the most recent ellipse.
        Uses a conservative bounding-circle overlap: two ellipses are considered
        overlapping if the distance between their centers is less than the sum
        of their semi-major axes scaled by overlap_threshold.

        overlap_threshold=0.0  → centroids must be within touching distance
        overlap_threshold=-0.5 → require 50% overlap (stricter)
        overlap_threshold=0.5  → allow up to 50% gap (looser)
        """
        if not self.ellipses:
            return True  # no history yet, always accept

        (cx0, cy0), (MA0, ma0), _ = self.ellipses[-1]
        (cx1, cy1), (MA1, ma1), _ = new_ellipse

        dist = np.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)
        # semi-major radii (MA is the full axis length in cv2)
        r0 = max(MA0, ma0) / 2
        r1 = max(MA1, ma1) / 2

        return dist < (r0 + r1) * (1 + overlap_threshold)

    def area_is_consistent(self, new_ellipse, max_area_ratio=1.25):
        """
        Reject matches where the new ellipse area differs too much from
        the tracked ellipse area. Prevents small noise from being stitched
        together into a plausible-looking track. Tracks area between last frame and next frame

        max_area_ratio=1.25 → new ellipse can be at most 1.25x larger or smaller
        """
        if not self.areas:
            return True

        old_area = self.areas[-1]
        (_, (MA1, ma1), _) = new_ellipse
        new_area = np.pi * (MA1 / 2) * (ma1 / 2)

        if old_area < 1e-3:
            return False

        ratio = max(new_area / old_area, old_area / new_area)
        return ratio < max_area_ratio

    @property
    def area_stability(self):
        #Check if area is relatively stable (low coefficient of variation) - across entire deque
        if len(self.areas) < 3:
            return True  # not enough data yet
        areas_array = np.array(self.areas)
        cv = np.std(areas_array) / np.mean(areas_array)  # coefficient of variation
        return cv < 0.1  # area shouldn't change more than 10%

    @property
    def crest_length(self):
        #major axis of current ellipse
        (cx, cy), (MA, ma), angle = self.ellipses[-1] #selects the most recent ellipse parameters - back of the deque
        return max(MA, ma) * self.gsd

    @property
    def speed(self):
        #define the speed of the wave - difference in centroid position over time (last two positions)
        if len(self.centroids) < 2:
            return None #not enough data to calculate speed
        dp = np.array(self.centroids[-1]) - np.array(self.centroids[-2]) #difference in position between the last two centroids
        speed_per_frame = np.linalg.norm(dp) #euclidian distance between the two centroids - pixels per frame
        return speed_per_frame * self.fps * self.gsd #*ground sampling distance to convert to real world units meters per second

    @property
    def smooth_speed(self):
        #speed from linear regression over entire centroid history - reduces the noise from instantaneous speed
        if len(self.centroids) < 3:
            return self.speed
        pts = np.array(self.centroids)
        t = np.arange(len(pts)) * self.dt
        vx = np.polyfit(t, pts[:, 0], 1)[0] #slope of the linear fit for x coordinates - 1st degree polynomial
        vy = np.polyfit(t, pts[:, 1], 1)[0] #slope of the linear fit for y coordinates
        speed_px_per_sec = np.sqrt(vx**2 + vy**2) #euclidian speed from the linear fit - pixels per frame per second
        return speed_px_per_sec * self.gsd

    def lambda_contribution(self):
        #returns the speed and crest length to help build the distribution
        #call once per frame while wave tracking is active
        s = self.smooth_speed
        if s is None:
            return None
        l = self.crest_length
        return (s, l)


#when building a class - write the attributes first: what data do I need to store?
#then write the methods: what do I need to do with that data? how do I want to interact with it?
#an overall update function can run the methods in a complete and clean sequence
class WaveTracker:

    def __init__(self, fps, image_shape, gsd, max_distance=50, max_missed=5, min_track_length=5, overlap_threshold=0.5, max_area_ratio=1.25, track_overlap_fraction=1.0):
        self.tracks = {} #dictionary of active waves
        self.fps = fps
        self.gsd = gsd #ground sampling distance - pixel scale to convert to meters
        self.max_distance = max_distance #maximum distance in pixels to consider a detection as part of an existing track
        self.max_missed = max_missed #maximum number of consecutive missed detections before a track is terminated
        self.min_track_length = min_track_length  #minimum number of frames a track must exist to be considered valid
        self.overlap_threshold = overlap_threshold
        self.max_area_ratio = max_area_ratio
        self.track_overlap_fraction = track_overlap_fraction
        self.frame_idx = 0
        self.image_shape = image_shape

        self.lambda_accumulator = [] #list to accumulate speed and crest length pairs for building the distribution

    def update(self, filtered_contours):

        #pass already filtered contours for the frame, returns active tracking
        detections = self._fit_ellipses(filtered_contours, image_shape) #fit ellipses to the contours and get centroids and ellipse parameters
        self._match_and_update(detections)
        self._suppress_overlapping_tracks(overlap_fraction=self.track_overlap_fraction)
        self._accumulate_lambda()
        self._prune()
        self.frame_idx += 1
        return self.tracks

    def _fit_ellipses(self, contours, image_shape):
        detections = []
        h, w = image_shape
        for cont in contours:
            if len(cont) < 5:
                continue

            area = cv2.contourArea(cont)
            if area < 10:  # reject small noise contours  #FIXME: play with this number - originally 500 - 150 works okay too
                continue

            ellipse = cv2.fitEllipse(cont) #fit an ellipse to the contour - returns (center (x, y), (major axis, minor axis), angle of rotation)
            x, y = ellipse[0] #centroid location of the ellipse
            MA, ma = ellipse[1] #major and minor axis lengths of the ellipse

            # guard against NaN/inf in ellipse center
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            if not (0 <= x <= w and 0 <= y <= h): #toss ellipses if they are somehow outside the image bounds
                continue

            # reject ellipses that are unrealistically large
            max_axis = max(MA, ma)
            if max_axis > w * 0.6:  # reject if major axis > 60% of image width
                continue

            # reject degenerate ellipses (nearly a line)
            if ma < 1e-3:
                continue

            # Minimum area threshold for ellipse
            ellipse_area = np.pi * (MA / 2) * (ma / 2)
            if ellipse_area < 200:  #FIXME: minimum ellipse area in pixels - important tuning parameter - originally 200
                continue

            if min(MA, ma) / max(MA, ma) > 0.8:  # reject near-circular ellipses
                continue  #FIXME: consider removing this

            detections.append((np.array([x, y]), ellipse))
        return detections

    def _match_and_update(self, detections):
        #use hungarian algorithm to match detections to existing tracks based on distance between centroids, then update matched tracks and create new tracks for unmatched detections

        active_ids = list(self.tracks.keys()) #get the IDs of currently active tracks - ordered list to allow for indexing

        if not active_ids:  #if there are no active tracks, initialize new tracks for all detections
            for centroid, ellipse in detections:  #loop over all detections (centroid and ellipse)
                t = WaveTrack(centroid, ellipse, self.frame_idx, self.fps, self.gsd) #create new tracking object
                self.tracks[t.id] = t  #store the new track using a unique ID
            return

        if not detections: # if there are no detections, mark all active tracks as missed and return
            for tid in active_ids:
                self.tracks[tid].mark_missed() #for each active track, increase the missed counter if track is not detected
            return

        track_centroids = np.array([self.tracks[tid].centroids[-1] for tid in active_ids]) #get the most recent centroid for each active track
        det_centroids = np.array([d[0] for d in detections]) #get the centroids from the current detections
        cost = cdist(track_centroids, det_centroids) #compute pair wise distances between tracked and detected centroids cost[i, j] = distance(track_centroids[i], det_centroids[j])
        row_idx, col_idx = linear_sum_assignment(cost) # find the best matches between tracks and detections - given a cost matrix, find the optimal assignments that minimize the total cost
        #Tries to find the best matches between existing tracks and new detections - smallest distance between tracks and detections

        matched_tracks = set() #create unique elements to track which tracks and detections have been matched to avoid double counting
        matched_dets = set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < self.max_distance:
                track = self.tracks[active_ids[r]]
                _, new_ellipse = detections[c]

                # Reject match if ellipses don't overlap spatially
                if (not track.overlaps_with(new_ellipse, overlap_threshold=self.overlap_threshold) or
                        not track.area_is_consistent(new_ellipse, max_area_ratio=self.max_area_ratio)):
                    self.tracks[active_ids[r]].mark_missed()
                    matched_dets.add(c)
                    t = WaveTrack(*detections[c], self.frame_idx, self.fps, self.gsd)
                    self.tracks[t.id] = t
                    continue

                track.update(*detections[c])
                matched_tracks.add(r)
                matched_dets.add(c)

        for r, tid in enumerate(active_ids):  #handle the unmatched trakcs - if a track was not matched to any detection, mark it as missed
            if r not in matched_tracks:
                self.tracks[tid].mark_missed()

        for c, det in enumerate(detections): #handle unmatched detections - if a detection was not matched to any track, initialize a new track for it
            if c not in matched_dets:
                t = WaveTrack(*det, self.frame_idx, self.fps, self.gsd)
                self.tracks[t.id] = t

    def _suppress_overlapping_tracks(self, overlap_fraction=1.0):
        '''Plugs the centroid of one ellipse into the ellipse equation of the other to determine if they overlap.
        If the value is less than 1, the point is inside the ellipse. If the value is greater than 1, the point is outside the ellipse. If the value is equal to 1, the point is on the boundary of the ellipse.
        The overlap fraction can scale the threshold for determining if two tracks are considered overlapping. For example, an overlap_fraction of 0.5 would consider two tracks to be overlapping if the centroid
        of one track is within half the distance of the ellipse boundary of the other track. An overlap_fraction of 1.0 would require the centroid to be within the full distance of the ellipse boundary for the
        tracks to be considered overlapping. Adjusting this parameter allows for more or less strict criteria for suppressing overlapping tracks based on their spatial proximity.

        The ellipse equation is as follows:

        (x / MajorAxis)^2 + (y / MinorAxis)^2 <= overlap_fraction (1.0)

        The rotation matrix for an ellipse with angle theta is:

        [x_rot, y_rot] = [[cos(angle_a), -sin(angle_a)]; [sin(angle_a), cos(angle_a)]] * [dx, dy]

        where (dx, dy) are the gap between the centroid of the point and the center of the ellipse.
        x_rot and y_rot are the location of the point being evaluated inside the ellipse reference frame. This is what is plugged into the ellipse equation.'''

        ids = list(self.tracks.keys())  #grab all track ids into a list
        to_delete = set()  #empty set to collect ids for removal

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):  #iterate over every unique pair of tracks
                tid_a, tid_b = ids[i], ids[j]
                if tid_a in to_delete or tid_b in to_delete:  #skip the pair if it is already marked for deletion
                    continue

                track_a = self.tracks[tid_a]  #define track a and track b
                track_b = self.tracks[tid_b]

                (cx_a, cy_a), (MA_a, ma_a), angle_a = track_a.ellipses[-1]  #unpack the most recent ellipse [-1] in the deque for track a and track b
                (cx_b, cy_b), (MA_b, ma_b), angle_b = track_b.ellipses[-1]  # unpacks the centroid location, major and minor axis lengths, and orientation angle

                # check if centroid of b is inside ellipse of a, and vice versa
                def point_in_ellipse(px, py, cx, cy, MA, ma, angle_deg):
                    '''Asks the question: is a point inside an ellipse - scaled by a factor?'''
                    angle_rad = np.deg2rad(angle_deg)  #convert ellipse rotation angle from degrees to radians
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)  #precompute trig values
                    dx, dy = px - cx, py - cy  #translates the point to the ellipse center
                    # rotate point into ellipse frame
                    x_rot = cos_a * dx + sin_a * dy  #rotate the point into the ellipses frame of reference - basically unrotates the ellipse
                    y_rot = -sin_a * dx + cos_a * dy  #this is necessary because the point must be aligned with the ellipse axis for the ellipse equation to work correctly
                    return (x_rot / (MA / 2)) ** 2 + (y_rot / (ma / 2)) ** 2 <= overlap_fraction  #return True if the point is inside the ellipse defined by the major and minor axes, scaled by the overlap fraction (1.0 is the exact boundary, <1.0 shrinks it (softer filter), >1.0 expands it (stricter filter))

                b_in_a = point_in_ellipse(cx_b, cy_b, cx_a, cy_a, MA_a, ma_a, angle_a)  #check if either centroid is inside the other track's ellipse
                a_in_b = point_in_ellipse(cx_a, cy_a, cx_b, cy_b, MA_b, ma_b, angle_b)

                if b_in_a or a_in_b:
                    area_a = track_a.areas[-1]  #get the most recent area for track a and track b
                    area_b = track_b.areas[-1]
                    to_delete.add(tid_a if area_a < area_b else tid_b)  #mark the track with the smaller area for deletion - this removes smaller tracks that are likely noise inside bigger tracks - often happens in large foamy waves

        for tid in to_delete:  #if the track id is inside the to_delete set, remove it from the tracks dictionary
            del self.tracks[tid]

    def _accumulate_lambda(self):
        #each active track contributes on (speed, crest length) pair per frame to build the distribution
        #Only accumulate from tracks that have persisted long enough
        for track in self.tracks.values():
            # Only count waves that have been tracked for min_track_length frames
            track_age = self.frame_idx - track.birth_frame
            if track_age >= self.min_track_length:
                contribution = track.lambda_contribution()
                if contribution is not None:
                    self.lambda_accumulator.append(contribution)

    def _prune(self):
        to_delete = [tid for tid, t in self.tracks.items()
                     if t.missed > self.max_missed or not t.area_stability]
        for tid in to_delete:
            del self.tracks[tid]

    def compute_lambda(self, speed_bins, total_area, dt, total_time_sec):

        if not self.lambda_accumulator:
            return None

        speeds = np.array([s for s, _ in self.lambda_accumulator]) #extract speeds from the accumulated contributions
        lengths = np.array([l for _, l in self.lambda_accumulator]) #extract crest lengths from the accumulated contributions

        dc = speed_bins[1] - speed_bins[0] #bin width for speed
        n_bins = len(speed_bins) - 1  # number of intervals between edges
        Lambda = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (speeds >= speed_bins[i]) & (speeds < speed_bins[i + 1]) #select contributions that fall within the current speed bin
            Lambda[i] = lengths[mask].sum() #* dt #sum the total crest lengths - dt was added in to fix the units

        # Lambda /= (total_area * total_time_sec * dc) #normalize by the area to get a rate per unit area per unit time per speed bin #units of s/m^2
        Lambda /= (total_area * dc)

        return Lambda

def plot_tracked_ellipses(images, filt_conts, gsd, n_frames=6):
    """
        Re-runs the tracker over n_frames and plots ellipses colour-coded by track ID.
        images: list of original image paths
        """
    # generate a colour per track ID — consistent across frames
    np.random.seed(42)
    colour_map = {}  # tid → RGB colour

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    image_shape = (
    cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)[700:1500, ...].shape[0], cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)[700:1500, ...].shape[1])
    w, h = image_shape[1], image_shape[0]
    x_max = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)[700:1500, ...].shape[1]

    # reset tracker state for clean replay
    temp_tracker = WaveTracker(fps=29.97, gsd=gsd, image_shape=image_shape)

    for frame_idx in range(min(n_frames, len(filt_conts))):

        temp_tracker.update(filt_conts[frame_idx])

        img = cv2.cvtColor(cv2.imread(images[frame_idx]), cv2.COLOR_BGR2RGB)[700:1500, ...]
        ax = axes[frame_idx]
        ax.imshow(img)
        ax.set_title(f'Frame {frame_idx}')
        ax.axis('off')
        ax.set_xlim(3000, 3840)

        for tid, track in temp_tracker.tracks.items():

            # assign colour per track ID
            if tid not in colour_map:
                colour_map[tid] = np.random.rand(3)
            colour = colour_map[tid]

            # draw ellipse
            ellipse = track.ellipses[-1]
            (cx, cy), (MA, ma), angle = ellipse

            e = Ellipse(xy=(cx, cy),
                        width=MA, height=ma,
                        angle=angle,
                        edgecolor=colour,
                        facecolor='none',
                        linewidth=2)
            ax.add_patch(e)

            # draw centroid
            ax.plot(cx, cy, '.', color=colour, markersize=6)

            # label track ID
            # ax.annotate(f'T{tid}', xy=(cx, cy),
            #             color=colour, fontsize=8,
            #             xytext=(5, 5), textcoords='offset points')

            # draw centroid trail
            if len(track.centroids) > 1:
                trail = np.array([[c[0], c[1]] for c in track.centroids])
                # only plot trail points within image bounds
                valid = (trail[:, 0] >= 0) & (trail[:, 0] <= w) & \
                        (trail[:, 1] >= 0) & (trail[:, 1] <= h)
                trail = trail[valid]
                if len(trail) > 1:
                    ax.plot(trail[:, 0], trail[:, 1], '-',
                            color=colour, linewidth=1, alpha=0.6)

    plt.tight_layout()
    plt.savefig('./drone/outputs/temp/tracked_ellipses.png', dpi=300)
    plt.show()
    plt.close('all')

def contour_area_dist(contourList, sigma=7):

    # plot distribution of contours
    cont_areas = [cv2.contourArea(c) for conts in contourList for c in conts]

    # use scotts rule to determine number of bins the histogram should have
    n = len(cont_areas)
    cont_areas_arr = np.array(cont_areas)
    bin_width_scott = 3.5 * np.std(cont_areas_arr) / n ** (1 / 3)
    n_bins_scott = int(np.ceil((cont_areas_arr.max() - cont_areas_arr.min()) / bin_width_scott))

    # calculate histogram
    hist, bins = np.histogram(cont_areas, bins=n_bins_scott, range=(0, 256))
    knee, _ = find_knee(hist, sigma=sigma)
    # plt.hist(hist, bins)
    # plt.xscale('log')
    # plt.show()
    # plt.close('all')

    return knee


def find_knee(hist, sigma=5):
    # find where histogram drops off from the spike
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


if __name__ == "__main__":

    #Get Ground Sampling Distance to convert pixels to meters
    drone = np.load('./drone/1057metadata.npy')
    dx = float(drone[-2][1])  #dx and dy are the same value
    gsd = dx #ground sampling distance

    # rgb images - non modified
    rgb_images = sorted(glob.glob('./drone/processed/subset20171791057/*.jpeg'), key=numerical_sort)[:282]
    image_shape = (cv2.imread(rgb_images[0], cv2.IMREAD_GRAYSCALE)[700:1500, ...].shape[0],
                   cv2.imread(rgb_images[0], cv2.IMREAD_GRAYSCALE)[700:1500, ...].shape[1])  # height, width of the images

    #load saved contours:
    filtered_contours = [loadContours(file) for file in sorted(glob.glob(f'./drone/processed/subset20171791057_contours_filtered/*.pkl'), key=numerical_sort)]

    # knee = contour_area_dist(conts, sigma=7) #TODO: can use this for other histograms for filtering - instead of guessing integers

    #define observation area in pixels
    #clip image to area of interest (where waves are visible) to calculate area in meters squared
    y_min, y_max = 700, 1500
    full_area_pixels = cv2.imread(rgb_images[0], cv2.IMREAD_GRAYSCALE)[700:1500, ...].shape[0] * cv2.imread(rgb_images[0], cv2.IMREAD_GRAYSCALE)[700:1500, ...].shape[1]

    N_frames = len(rgb_images)
    full_area_meters = full_area_pixels * (gsd**2) * N_frames

    #define total observation time in seconds
    fps = 29.97
    total_time_sec = len(rgb_images) / fps

    tracker = WaveTracker(fps=fps, gsd=gsd, image_shape=image_shape, max_distance=20, max_missed=3, min_track_length=7, overlap_threshold=0.1, max_area_ratio=1.25, track_overlap_fraction=1.0)

    for frame_conts in tqdm(filtered_contours):
        tracker.update(frame_conts)

    # speed_bins = np.linspace(0, 1000, 68)  # pixels/second
    speed_bins = np.linspace(0, 20, 68) # meters/second
    Lambda = tracker.compute_lambda(speed_bins, full_area_meters, 1/fps, total_time_sec)

    # phillips 1985 dissipation
    # b = 1e-3
    b = 7e-5
    rho = 1025
    g = 9.81
    dc = speed_bins[1] - speed_bins[0]
    c = 0.5 * (speed_bins[:-1] + speed_bins[1:])  # bin centers
    fifth_moment = c ** 5 * Lambda * dc
    epsilon = b * rho * (1 / g) * np.sum(fifth_moment)  # TODO:check for Nan's in here and in other parts of the script
    print(f'Estimated Energy dissipation rate: {epsilon:.4e} W/m^2')  # watt is a joule per second. so really this is joules per second per square meter. the energy dissipated by the breaking waves per unit area of the ocean surface per unit time.
    # this quantity is representative of the time frame that you input into the tracker. Only represents the energy dissipation rate for the frames that I put in.

    plotting_images = rgb_images[-6:]  # last 6 frames of the original rgb images, clipped to the area of interest
    plot_tracked_ellipses(plotting_images, filtered_contours[-6:], gsd,  n_frames=6)
    # TODO: some problems with tracking ellipses - small noise will often be near other small noise: then the tracker believes that it is a wave that is moving.
    # TODO: remember than the lambda distribution is integrated over all wave directions - account for this at some point.
    # TODO: Currently the accumulator is in temporal style: per frame accumulation of waves

    #c**-6 reference line
    idx = 20  # pick a reasonable point away from noise
    c_ref = c[idx]
    Lambda_ref = Lambda[idx]
    Lambda_c6 = Lambda_ref * (c / c_ref) ** (-6)
    # c_line = np.logspace(np.log10(c.min()), np.log10(c.max()), 500)
    # Lambda_c6 = Lambda_ref * (c_line / c_ref) ** (-6)

    # plot lambda distribution - for the entire image and time period
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121)
    # ax.bar(c, Lambda, width=dc, align='center', edgecolor='k')
    ax.loglog(c, Lambda)
    ax.loglog(c, Lambda_c6, '--', label=r'$c^{-6}$')
    ax.set_xlabel('Wave Speed (m/s)')
    ax.set_ylabel(r'$\Lambda(c)$ (s * m$^{-2}$)') #units of m^-2, time is absorbed in the calculate lambda function
    ax.set_title('\u039B(c) Distribution of Breaking Waves')
    ax.set_ylim(1e-8, 1e0)
    ax.legend()
    ax.grid(axis='y')

    ax1 = fig.add_subplot(122)
    # ax1.bar(c, fifth_moment, width=0.8 * dc, color='salmon', edgecolor='k')
    ax1.loglog(c, fifth_moment, color='salmon')
    ax1.set_xlabel('Wave Speed (m/s)')
    ax1.set_ylabel(r'c$^5$ * $\Lambda(c)$ * dc (m$^4$ s$^{-5}$)')
    ax1.set_title('Fifth Moment of the \u039B(c) Distribution')
    ax1.grid(axis='y')
    plt.savefig('./drone/outputs/temp/distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')

    # plot clipped contours for a single frame to visually check the contour filtering and clipping
    plotting_contours = filtered_contours[0]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(cv2.imread(rgb_images[0])[700:1500, ...])
    c = cv2.drawContours(cv2.imread(rgb_images[0])[700:1500, ...], plotting_contours, -1, color=(255, 255, 0), thickness=2)
    ax.imshow(c)
    ax.set_title('Clipped Contours Overlay')
    # plt.savefig('./drone/outputs/temp/clipped_contours.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')

#TODO: check the result of the distribution - first check if the observed wave spectrum for the estuary follows the decay
# shape of the phillips 1958 equilibrium spectrum f^-5 - must follow/abide by all assumptions too.

#TODO: MORE ROBUST WAVE TRACKING - use ellipse equation to make sure the the following waves is inside the ellipse of the previous wave. This is consistent with overlapping contour methods from other literature.
#TODO: smooth the contours using the technique used by Akaawase et al.

#TODO: ensure ellipse area remains similar between frames cannot drop immensly

#TODO: make sure that speeds are filtered the same as in the elemental method - start with tossing everything under 0.6 m/s



