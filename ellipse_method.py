import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import glob
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm
from collections import deque
import pickle

from utils import numerical_sort, load_chunk, loadContours, get_mask


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

        overlap_threshold=0.0  -> centroids must be within touching distance
        overlap_threshold=-0.5 -> require 50% overlap (stricter)
        overlap_threshold=0.5  -> allow up to 50% gap (looser)
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

        max_area_ratio=1.25 -> new ellipse can be at most 1.25x larger or smaller
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
        self.fps = fps #frames per second
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
        detections = self._fit_ellipses(filtered_contours, self.image_shape) #fit ellipses to the contours and get centroids and ellipse parameters
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
            cont = np.asarray(cont, dtype=np.float32) #change the type to float32 to avoid errors in cv2

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

    def compute_lambda(self, speed_bins, total_area):

        if not self.lambda_accumulator:
            return None

        speeds = np.array([s for s, _ in self.lambda_accumulator]) #extract speeds from the accumulated contributions
        lengths = np.array([l for _, l in self.lambda_accumulator]) #extract crest lengths from the accumulated contributions

        dc = speed_bins[1] - speed_bins[0] #bin width for speed
        n_bins = len(speed_bins) - 1  # number of intervals between edges
        Lambda = np.zeros(n_bins)  #initialize empty array to fill later

        for i in range(n_bins):
            mask = (speeds >= speed_bins[i]) & (speeds < speed_bins[i + 1]) #select contributions that fall within the current speed bin
            Lambda[i] = lengths[mask].sum() #* dt #sum the total crest lengths - dt was added in to fix the units

        Lambda /= (total_area * dc)  #normalize by the total area and bin width to get the final lambda distribution

        return Lambda

    def sum_dl(self, speed_bins):

        if not self.lambda_accumulator:
            return None  #if there are no contributions, return None

        speeds = np.array([s for s, _ in self.lambda_accumulator])  # extract speeds from the accumulated contributions
        lengths = np.array([l for _, l in self.lambda_accumulator])  # extract crest lengths from the accumulated contributions

        dc = speed_bins[1] - speed_bins[0]  # bin width for speed
        n_bins = len(speed_bins) - 1  # number of intervals between edges
        Lambda = np.zeros(n_bins)  # initialize empty array to fill later

        for i in range(n_bins):
            mask = (speeds >= speed_bins[i]) & (speeds < speed_bins[i + 1])  # select contributions that fall within the current speed bin
            Lambda[i] = lengths[mask].sum()  # * dt #sum the total crest lengths - dt was added in to fix the units

        return Lambda




def compute_lambda(rectified_images_dir, filtered_contours_dir, lambda_save_dir, gsd, fps, speed_bins=0.2):

    #determine the number of chunks to iterate through
    n_chunks = len(glob.glob(os.path.join(rectified_images_dir, '*.dat')))

    dc = speed_bins  # speed bin size for lambda computation
    c_bins = np.arange(0, 20 + dc, dc)  # speed bins

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
        tracker = WaveTracker(fps=fps, gsd=gsd, image_shape=(h, w), max_distance=20, max_missed=3, min_track_length=7, overlap_threshold=0.1, max_area_ratio=1.25, track_overlap_fraction=1.0)

        #load in the contours for each chunk
        chunk_contours = loadContours(os.path.join(filtered_contours_dir, f'Filtered_contours_chunk_{chunk:04d}.pkl')) #list of shape (n_frames, n_contours_in_frame, n_points_in_conours, (x, y))

        for frame in tqdm(range(n_frames), colour='blue', leave=False, desc='Frame'):

            #load in the contours for each frame
            frame_contours = chunk_contours[frame]  #get the contours for the current frame. shape: (n_contours_in_frame, n_points_in_conours, (x, y))

            tracker.update(frame_contours) #update the wave tracking class for each frame

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

    Lambda = compute_lambda(rectified_images_dir, filtered_contours_dir, lambda_save_dir, gsd, fps)

    #TODO: really need to go back into the image processing and check for noise. Only the large waves with velocities over 1 m/s are actually significant here.

    #TODO: convex hull from contours?

    #TODO: we want to plot some sample images as well here.