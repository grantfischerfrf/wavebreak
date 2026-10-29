import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from config import Config
from utils import load_chunk, loadContours, get_mask
from collections import Counter

CHI2_GATE = 9.21  # 99% chi-square threshold, 2 degrees of freedom
BIG = 1e6  # cost for pairs that fail a gate (Hungarian can never pick these)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def polygon_iou(a, b):
    """IoU of two polygons given as (N, 2) float arrays, rasterized in a shared bounding box."""
    pts = np.vstack([a, b])   #given two polygons of shape (N, 2) stack them vertically (row wise)
    x0, y0 = np.floor(pts.min(0)).astype(int) # return the minimum x, y point in the array
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 1  #return the maximum x, y point in the array + 1
    ma = np.zeros((y1 - y0, x1 - x0), np.uint8) #create a bounding box the encloses the maximum pixel boundary that encloses both polygons
    mb = np.zeros_like(ma) #second mask that is a copy of mask a
    cv2.fillPoly(ma, [np.round(a - [x0, y0]).astype(np.int32)], 1) #draw a filled polygon - shift the polygon to the minimum point - fits the new mask
    cv2.fillPoly(mb, [np.round(b - [x0, y0]).astype(np.int32)], 1)  #rounding each point such that it fits the new mask grid - then fill the area of the polygon with 1's
    union = np.logical_or(ma, mb).sum()  #count the number of pixels covered by at least one of the two shapes
    return np.logical_and(ma, mb).sum() / union if union else 0.0  #count the number of pixels covered by both shapes - divide by union. Intersection over union


# ----------------------------------------------------------------------------
# One Track = one wave = one Kalman filter
# ----------------------------------------------------------------------------
class Track:
    """State is [cx, cy, vx, vy] in pixels and pixels/frame. cv2.KalmanFilter does the math."""
    #kalman filter class - recursive filter depending on prediction and correction with measurements. Only stores the current state and covariance, not the history.
    # in the wave tracker: a wave that is tracked every frame alternates predict and correct, one that goes untracked predicts and predicts until it is matched again or dies when > max misses
    _next_id = 0

    def __init__(self, center, area, contour, flow_vel, q_acc, r_pos, r_vel):
        kf = cv2.KalmanFilter(4, 4, 0, cv2.CV_64F) #define the kalman filter. 4 state variables, 4 measured variables, 0 control inputs, float 64 precision
        kf.transitionMatrix = np.array([[1, 0, 1, 0],   #define the transition matrix (F): cx' = cx + vx, cy' = cy + vy, vx' = vx, vy' = vy
                                        [0, 1, 0, 1],           #constant velocity assumption - time step is one frame
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], float)
        kf.measurementMatrix = np.eye(4)  #measurement matrix (H). I means measurement vector is the initial state itself [cx, cy, vx, vy].   cx is centroidx, etc...
        G = np.array([[.5, 0], [0, .5], [1, 0], [0, 1]])  # white-noise acceleration model
        kf.processNoiseCov = (G @ G.T) * q_acc ** 2  #Process noise Q: assumes constant velocity model is imperfect bc waves accelerate. This is modeled as a random acceleration
        # a with standard deviation q_acc (px/frame^2) acting during one frame. It changes velocity by a and position by 0.5a, so each axis’s noise vector is [0.5, 1]·a.
        # G stacks those vectors for x and y, and G G.T q^2 is the covariance.

        #Initial state and covariance - the estimate before seeing the waves measurements - written by predict()
        v0 = flow_vel if flow_vel is not None else (0.0, 0.0)  #if the initial velocity is not given, set to 0, 0
        vel_var = r_vel ** 2 if flow_vel is not None else 25.0  #velocity variance if flow gave a starting velocity, otherwise give 25 -> very uncertain
        kf.statePost = np.array([[center[0]], [center[1]], [v0[0]], [v0[1]]], float) #initial state vector [x, y, vx, vy]. the covariance matrix says how uncertain this vector estimate is
        kf.errorCovPost = np.diag([r_pos ** 2, r_pos ** 2, vel_var, vel_var])  #covariance error matrix same row and column as the state (4x4 diagonal matrix here). contains position and velocity
        #variances, which are uncertainty squared.

        self.kf, self.r_pos, self.r_vel = kf, r_pos, r_vel  #filter object kf, and the two noise settings r_pos and r_vel
        self.id = Track._next_id
        Track._next_id += 1
        self.area = area  #last matched ellipse area
        self.contour = contour  # last matched contour (N, 2)
        self.contour_center = np.asarray(center, float)  # detected centroid that contour belongs to
        self.hits, self.misses = 1, 0  # hits = total matched frames (gaps allowed) - starts at one bc birth counts as an observation, misses is number of consecutive unmatched frames
        self.path = {}  # {frame index :cx, cy, crest length in px}, one entry per matched frame
        self.birth = 0  # frame index the track was born on

    @property  #allows t.x instead of t.x()
    def x(self):
        return self.kf.statePost.ravel()  # returns the post state vector [cx, cy, vx, vy]. This holds the predicted position after predict and before update, and the corrected position after update

    def predict(self):
        self.kf.predict()  #predict the location of the new track using a kalman filter (.predict is a function of cv2.kalman)
        #.predict() computes -> statePre = F·statePost and errorCovPre = F·errorCovPost·F.T + Q, then copies both into the Post slots
        #for a coasting track nothing calls .correct() so the .predict() starts from the previous prediction

    def gate_cov(self, floor_px):
        """Builds the Position innovation covariance (2x2) with a minimum gate size so it never gets too tight."""
        return self.kf.errorCovPre[:2, :2] + np.eye(2) * (self.r_pos ** 2 + floor_px ** 2)
        #errorCovPre[:2, :2] position block of filter prediction uncertainty. floor_px is a fixed minimum that stops gate from collapsing once the filter gets confident

    def predicted_contour(self):
        """Last matched contour shifted to where the filter now thinks the wave is."""
        return self.contour + (self.x[:2] - self.contour_center) #self.x[:2] is a position block of predicted position

    def update(self, center, contour, area, flow_vel=None):
        #when this function is called .predict() has already produced a prior for this frame - statePre (where the wave is expected) and errorCovPre(how uncertain the estimate is)
        #.correct() reads those inputs and produces a posterior statePost and errorCovPost

        # cv2 fixes the measurement size, so always send 4 numbers [x, y, vx, vy]; with no flow the
        # velocity entries get a huge variance and are effectively ignored.
        rv = self.r_vel ** 2 if flow_vel is not None else 1e6  #with no flow the variance is set extremely high
        fv = flow_vel if flow_vel is not None else (0.0, 0.0)
        self.kf.measurementNoiseCov = np.diag([self.r_pos ** 2, self.r_pos ** 2, rv, rv])  #diagonal noise covariance matrix for the position and velocity of the measured ellipse
        self.kf.correct(np.array([[center[0]], [center[1]], [fv[0]], [fv[1]]], float)) #builds the measurement vector - raw ellipse from the detection - results of .correct() are stored
        # in statePost and errorCovPost -> this is what t.x reads
        self.contour, self.contour_center, self.area = contour, np.asarray(center, float), area # self.contour_center store the raw contour centroid. self.area is overwritten to always be current frame
        self.hits += 1 #bumps the match count up by one each time the track is matched to a detection
        self.misses = 0  #allows for a coasting track to return. If A detection is matched again this is set to 0.


# ----------------------------------------------------------------------------
# Tracker: ellipse fitting + Hungarian assignment + track lifecycle
# ----------------------------------------------------------------------------
class WaveTracker:
    def __init__(self, fps, image_shape, gsd, phase_speed, spacing=5,
                 min_age=19, max_misses=5, max_area_change=2.0,
                 q_acc=0.05, r_pos=2.0, r_vel=0.5,
                 gate_floor=3.0, w_area=0.5, w_iou=1.0, flow_gate=2.0, half_window=5):
        """
        :parameter
            min_age:         matched frames a track needs before it counts toward Lambda (gaps allowed)
            max_misses:      consecutive frames a track may coast without a detection before it is dropped
            max_area_change: allowed fractional area change per frame (0.5 -> ratio in [1/1.5, 1.5])
            q_acc:           process noise: how much a wave's velocity can change per frame (px/frame^2)
            r_pos:           centroid jitter (px)
            r_vel:           noise on the flow velocity (px/frame)
            gate_floor:      minimum gate radius (px) so a confident track cannot reject a real wave
            w_area, w_iou:   weights of the soft area and IoU terms in the assignment cost
            flow_gate:       ignore a flow velocity that differs from the filter's velocity by more than
                            this many px/frame (once the track has a few hits)
            half_window:     speed at frame f = centroid displacement between f-h and f+h divided by the time
                            between them (h = half_window frames). Bigger h = smoother speed, less time resolution
        """
        self.fps, self.gsd, self.image_shape, self.spacing = fps, gsd, image_shape, spacing
        self.max_speed = 1.5 * phase_speed  # m/s, physical cap on accepted speed samples
        self.min_age, self.max_misses = min_age, max_misses
        self.max_area_change = max_area_change
        self.kf_params = dict(q_acc=q_acc, r_pos=r_pos, r_vel=r_vel)
        self.gate_floor, self.w_area, self.w_iou, self.flow_gate = gate_floor, w_area, w_iou, flow_gate
        self.tracks = []
        self.samples = []  # (speed m/s, crest length m, track age in frames, track id, total track length in frames, frames from nearest track end, path straightness)
        self.track_lengths = []  # hits of every finished track, for tuning
        self.diag = Counter()  # per track-frame outcome counts, for tuning
        self.half_window, self.frame = half_window, 0  # self.frame = index of the frame being processed

    # -- detections ----------------------------------------------------------
    def _fit_ellipses(self, contours, image_shape):
        detections = []   #empty list to hold detected ellipses - becomes a list of tuples
        h, w = image_shape  #define image shape, height and width
        for cont in contours:
            cont = np.asarray(cont, dtype=np.float32).reshape(-1, 2)  #for a selected contour get rid of the dummy dimension (N_points, 1, 2) -> (N_points, 2)
            if len(cont) < 5:  #if the number of points is less than 5 skip it
                continue
            if cv2.contourArea(cont) < 50:  #if the area of the contour is less than 50 skip it. small ellipses are unreliable for crest length calculations, TODO: change me??
                continue
            # hull = cv2.convexHull(cont)
            # if len(hull) < 8 or cv2.contourArea(hull) < 1e-3:
            #     continue

            ellipse = cv2.fitEllipse(cont)  #fit an ellipse to the contour
            x, y = ellipse[0] #x, y location of the ellipse centroid
            MA, ma = ellipse[1]  #semi major and semi minor ais of the ellipse
            if not np.isfinite(x) or not np.isfinite(y):  # ensure the centroid is a real number
                continue
            if not (0 <= x <= w and 0 <= y <= h):  #ensure the centroid is inside the image height and width
                continue
            if ma < 1e-3: #if the minor axis is incredibly small skip the ellipse
                continue

            area = np.pi * (MA / 2) * (ma / 2)  #area of the ellipse
            detections.append((np.array([x, y]), ellipse, cont, area))  #append the centroid location, ellipse, contour, and area to detections list
        return detections

    # -- flow ----------------------------------------------------------------
    def _flow_velocity(self, cont, vx, vy):
        """Median flow (px/frame) inside a contour, or None."""
        if vx is None:  #check is vx holds data
            return None
        h, w = vx.shape  # pull the shape of the optical flow data, (height, width)
        pts = np.round(cont).astype(np.int32).reshape(-1, 1, 2) #convert contour to integer pixel coordinates (no floats for pixels) (np.round and dtype int32) - reshapes the contour (n_points, 1, 2)
        x, y, bw, bh = cv2.boundingRect(pts) #draw a bounding rectangle around the points (x, y, base, height)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h) # clip the rectangle to the dimensions of the optical flow array
        if x1 <= x0 or y1 <= y0: #check to make sure rectangle has a positive area
            return None
        roi = np.zeros((bh, bw), np.uint8)  #define a black mask the size of height and width of the bounding box
        cv2.drawContours(roi, [pts], -1, 255, -1, offset=(-x, -y))  #draw contour (pts) inside the ROI. fill the contour - shift to the local coords of the rectangle (offset (-x, -y))
        sel = roi[y0 - y:y1 - y, x0 - x:x1 - x] > 0  #create boolean mask for points that fall inside the contour = True.
        if not sel.any():
            return None
        dx = np.median(vx[y0:y1, x0:x1][sel]) / self.spacing  #take the median x velocities inside the bounding rectangle, inside the contour in px/frame  #TODO: when m/s?
        dy = np.median(vy[y0:y1, x0:x1][sel]) / self.spacing  #y velocities
        if not (np.isfinite(dx) and np.isfinite(dy)):
            return None
        return np.array([dx, dy])

    def _checked_flow(self, track, fv):
        """Drop a flow velocity that disagrees wildly with an established track."""
        if fv is None or track is None or track.hits < 3:  #return the flow if something doesnt exist of the number of hits, matches < 3
            return fv
        return fv if np.hypot(*(fv - track.x[2:4])) <= self.flow_gate else None  # if the median velocity in px/frame differs from the tracks current velocity by more than flow_gate return None
    # -- main step -------------------------------------------------------------
    def update(self, filtered_contours, vx=None, vy=None):
        """vx, vy: flow for this frame, shape (h, w), px per `spacing` frames (optional).
        Returns a list of (ellipse, hits, speed m/s or None, track_id) for this frame's detections."""
        dets = self._fit_ellipses(filtered_contours, self.image_shape) #returns a list of tuples holding centroid location, ellipse, contour, and ellipse area

        # 1. Predict future location of the ellipse - empty on the first iteration of the loop bc tracks have not been made yet
        for t in self.tracks:
            t.predict()

        # 2. Cost Matrix: Mahalanobis (gated) + soft area + soft IoU against the shifted last contour
        match = {}  # track index -> detection index
        cat = ['no_dets'] * len(self.tracks)  # why each track went unmatched (diagnostics) - initialize as no detections in the frame as the default
        if self.tracks and dets:  #if there is at least one track and one detection
            cost = np.full((len(self.tracks), len(dets)), BIG)  #build a cost matrix the size of (n_tracks, n_detections) fill with huge to make the cost very high as default
            hi, lo = 1 + self.max_area_change, 1 / (1 + self.max_area_change)  #define the allowed area bounds. self.max_area_change takes a percentage. 200% allowed means a contour can be [1/3, 3]x the size
            for i, t in enumerate(self.tracks):  #for index and track in self.tracks list
                S_inv = np.linalg.inv(t.gate_cov(self.gate_floor)) #inverse 2x2 position covariance (filter uncertainty + floor) used for Mahalanobis distance
                pred_cont = t.predicted_contour() #last contour shifted to the predicted location (via the kalman filter) - used for intersection over union calcualtion

                # diagnostics: what happens between this track and its nearest detection?
                d_all = np.array([np.hypot(*(d[0] - t.x[:2])) for d in dets]) #find the pixel Euclidean distance from the original ellipse to the new ellipse
                j0 = int(np.argmin(d_all)) #find the nearest detection
                r0 = dets[j0][0] - t.x[:2] # determine the residual vector between nearest detection and the track
                if d_all[j0] > 25:  #if the nearest detection is greater than 25 pixels away
                    cat[i] = 'nearest_det_far_(>25px)'
                elif r0 @ S_inv @ r0 > CHI2_GATE:
                    cat[i] = 'nearest_fails_distance_gate'  #fails the mahalanobis gate
                elif not (lo <= dets[j0][3] / t.area <= hi):
                    cat[i] = 'nearest_fails_area_gate'  #fails the area gate
                else:
                    cat[i] = 'nearest_ok_but_lost_assignment' # passes all other gates, so any miss must be from losing its assignemnt to another track

                #cost matrix loop - score one track against every detection in the frame
                for j, (c, _, cont, area) in enumerate(dets):  # for index and tuple of (centroid, ellipse (_), contour, area):
                    r = c - t.x[:2] #residual vector between the centroid and the predicted position in pixels
                    m2 = r @ S_inv @ r #mahalanobis distance squared. How many standard deviations away is this detection given filter uncertainty
                    if m2 > CHI2_GATE: #If the residual is Gaussian with covariance S, m2 follows a chi-square distribution with 2 degrees of freedom,
                        ## and 99% of true matches fall below 9.21. Anything above that is rejected, and the cell stays at BIG
                        continue
                    ratio = area / t.area  #area ratio between the new detected ellipse and the old ellipse
                    if ratio > hi or ratio < lo:  #if the ratio is above or below the threshold skip the detection
                        continue
                    u = polygon_iou(pred_cont, cont) #conpute the intersection over union from the previous contour and the new contour
                    cost[i, j] = m2 / CHI2_GATE + self.w_area * abs(np.log(ratio)) + self.w_iou * (1 - u)  #compute the cost matrix for the track [i] and detection [j]
                    # normalizes mahalanobis distance to 0-1, log makes symmetric penalty for area - 0~0.55, IOU 0-1, so cost goes from 0~2.55.
            rows, cols = linear_sum_assignment(cost) #use the hungarian algorithm to minimize the total cost across all pairings of tracks and detections
            match = {i: j for i, j in zip(rows, cols) if cost[i, j] < BIG}  #log the track [i] and detection [j] in match dict

        # diagnostics tally after assignment. t.misses is the number of consecutive misses before this frame
        for i, t in enumerate(self.tracks):  #for index track in tracks
            if i in match: #if the track is in the match dictionary
                self.diag['matched'] += 1  #add one to the matched tally
                if t.misses > 0:  #if the number of misses is >0
                    self.diag['_recovered_after_coast'] += 1  # subset of matched; underscore keeps it out of the table. matches recovered after losing the wave
            elif t.misses == 0:
                self.diag['first_miss:' + cat[i]] += 1  # the frame a track is first lost, and why
            else:
                self.diag['coasting_still_missing'] += 1  #if the wave is not found it is still missing
        self.diag['detections'] += len(dets)  #number of detections
        self.diag['frames'] += 1  #number of frames

        # 3. update matched tracks, birth unmatched detections, coast/death unmatched tracks
        n_old = len(self.tracks)  #number of old already detected tracks - before this frames new births
        matched_dets = set(match.values())  #set of detection indices that have been claimed on this frame
        out = []  #empty list for ploting

        #update matched tracks
        for i, j in match.items():  #for track and detection index i, j inside match dict
            t = self.tracks[i]  #pull the track ellipse
            c, ellipse, cont, area = dets[j]  #pull the centroid, ellipse, contour, area of the detection
            fv = self._checked_flow(t, self._flow_velocity(cont, vx, vy))  #returns an array of median optical flow velocities inside
            t.update(c, cont, area, fv) #apply the kalman correction - .update is in the Track class
            speed = np.hypot(t.x[2], t.x[3]) * self.fps * self.gsd  # filtered speed (np.sqrt(vx, vy)), m/s - plot label only, Lambda speeds come from _commit_samples
            t.path[self.frame] = (c[0], c[1], max(ellipse[1]))  # raw detected centroid + crest length for this frame in dict path{} keyed by the frame index {i, c0, c1, L}. builds to {i: info, i+n, info, ...}
            out.append((ellipse, t.hits, speed, t.id)) #append ellipse, number of hits, speed, track id. Used for plotting only. hits is compared to min age for dotted or solid lines

        #birth unmatched detections
        for j, (c, ellipse, cont, area) in enumerate(dets):  #for detection index j, and tuple(centroid, ellipse, contour, area)
            if j in matched_dets:  #if the detection is already matched, skip it
                continue
            fv = self._flow_velocity(cont, vx, vy)  #returns an array of median optical flow velocities inside
            t = Track(c, area, cont, fv, **self.kf_params) #instantiate the track class, builds the kalman filter
            t.birth = self.frame  #frame index that the track is born on
            t.path[self.frame] = (c[0], c[1], max(ellipse[1])) # # raw detected centroid + crest length for this frame in dict path{} keyed by the frame index {i, c0, c1, L}
            self.tracks.append(t) # append the new track to tracks
            out.append((ellipse, t.hits, None, t.id)) #append ellipse, number of hits, no speed yet because just born track, and the track id.

        for i in range(n_old):  #loop through the already detected tracks
            if i not in match:  #if the track id [i] is not in match
                self.tracks[i].misses += 1  #give the track a miss tally

        #coast or kill the unmatched tracks
        alive = []  #empty list for living tracks
        for t in self.tracks:  #for each track [i]
            if t.misses > self.max_misses: #if the misses tally is > max misses
                self.track_lengths.append(t.hits) # append the number of hits/matches
                self._commit_samples(t)  # track is finished: compute its speeds from the full path
            else:
                alive.append(t)  #else: keep the track alive
        self.tracks = alive  #feed the track back into tracks
        self.frame += 1  #increase the frame index by one
        return out  #return the out: list of tuples for plotting

    def _commit_samples(self, t):
        """Speed samples for a finished track from a windowed centroid displacement.
        Speed at frame f = distance between the matched centroids at the start and end of the window
        [f-h, f+h], divided by the time between them. Uses real frame numbers (gap safe) and becomes
        one-sided near the track ends, so no frames are thrown away."""
        if t.hits < self.min_age:  # min_age decides which tracks count
            return
        h = self.half_window
        f = np.array(sorted(t.path))  # matched frame numbers
        P = np.array([t.path[k] for k in f], float)  # columns: cx, cy, crest length (px)
        j0 = np.searchsorted(f, f - h)  # first matched frame inside each window
        j1 = np.searchsorted(f, f + h, side='right') - 1  # last matched frame inside each window
        dt = f[j1] - f[j0]  # frames between the window's end points
        speed = np.hypot(*(P[j1, :2] - P[j0, :2]).T) / np.maximum(dt, 1) * self.fps * self.gsd  # m/s
        edge = np.minimum(f - f[0], f[-1] - f)  # frames from the nearest end of the track
        stride = max(1, min(h, (len(f) - 1) // 2))  # h frames, shrunk for short tracks so there are always 3+ points
        Q = P[::stride, :2]  # centroids every `stride` matched frames: steps are large compared with jitter
        steps = np.hypot(*np.diff(Q, axis=0).T)
        # straightness = net displacement / path length: near 1 for a steady crest, near 0 for a jittering blob
        straight = np.hypot(*(Q[-1] - Q[0])) / steps.sum() if len(Q) >= 3 and steps.sum() > 0 else np.nan
        for k in np.flatnonzero((dt >= h) & (speed <= self.max_speed)):  # need at least h frames of baseline
            self.samples.append((speed[k], P[k, 2] * self.gsd, f[k] - t.birth, t.id, t.hits, edge[k], straight))

    def finalize(self):
        """Flush tracks still alive at the end so their lengths and speeds are counted."""
        for t in self.tracks:  #for track in the track list
            self.track_lengths.append(t.hits)  #append the number of matches to the list holding every finished track
            self._commit_samples(t)  #append the data to the samples list which holds the speed, crest length etc of all the detected waves
        self.tracks = []  #create a new empty list of tracks for the next chunk

    def sum_dl(self, speed_bins):
        '''build the numerator of the lambda c distribution - sum crest lengths in each speed bin'''
        if not self.samples:
            return np.zeros(len(speed_bins) - 1)  #return nothing in the bins is the samples list holds nothing
        arr = np.array(self.samples)  # convert list of tuples to numpy array, columns: speed, crest length, age, track id
        s, l = arr[:, 0], arr[:, 1] #pull the speed and crest length arrays
        return np.histogram(s, bins=speed_bins, weights=l)[0]  #return the histogram - each speed bin receives a crest length contribution using weights=l, instead of a count


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_tracks(image, detections, frame_idx, chunk, min_age, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')

    for ellipse, age, speed, tid in detections:
        if speed is not None and speed >= 1.0:
            (cx, cy), (MA, ma), angle = ellipse
            color = plt.get_cmap('tab20')(tid % 20)  # colour = track id, so continuity is visible
            mature = age >= min_age
            ax.add_patch(Ellipse((cx, cy), MA, ma, angle=angle, fill=False, edgecolor=color,
                                 linewidth=1.5 if mature else 0.8,
                                 linestyle='-' if mature else ':'))
            # if mature and speed is not None:
            #     ax.text(cx, cy, f'#{tid} {speed:.1f} m/s', color=color, fontsize=7,
            #             ha='center', va='center')

    # ax.set_title(f'Chunk {chunk:04d}, frame {frame_idx}  |  {len(detections)} detections')
    # ax.set_xlabel('x (px)')
    # ax.set_ylabel('y (px)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Lambda(c)
# ----------------------------------------------------------------------------
def epsilon_by_track_length(S, c_bins, Area_total, b, rho, g, half_window=5, edges=(0, 10, 19, 30, 60, np.inf)):
    """Split the dissipation estimate by the total length of the track each sample came from.
    S columns: speed (m/s), crest length (m), age, track id, track length (frames),
    frames from the nearest track end, path straightness of the track.
    Each sample's contribution uses the same speed binning as Lambda, so the rows add up to the printed epsilon."""
    c = 0.5 * (c_bins[:-1] + c_bins[1:])
    k = np.clip(np.digitize(S[:, 0], c_bins) - 1, 0, len(c) - 1)
    eps = b * rho / g * c[k] ** 5 * S[:, 1] / Area_total  # contribution of each sample to epsilon (W/m^2)
    tot = eps.sum()
    print('Dissipation by total track length (c = speed in m/s, L = crest length in m):')
    print('  frames      tracks   samples  samples%   eps (W/m^2)   eps%   eps/sample   med c   p90 c   mean L')
    rows = [(f'{a0}-{a1 - 1}' if np.isfinite(a1) else f'{a0}+', (S[:, 4] >= a0) & (S[:, 4] < a1))
            for a0, a1 in zip(edges[:-1], edges[1:])]
    rows.append(('total', np.ones(len(S), bool)))
    for label, m in rows:
        if m.any():
            print(f'  {label:9s} {len(np.unique(S[m, 3])):8d} {m.sum():9d} {m.mean() * 100:8.1f}% '
                  f'{eps[m].sum():12.3e} {eps[m].sum() / tot * 100:6.1f}% {eps[m].mean():12.2e} '
                  f'{np.median(S[m, 0]):7.2f} {np.percentile(S[m, 0], 90):7.2f} {S[m, 1].mean():8.2f}')

    # end effects and coherence: are the fast short tracks still fast away from their ends, and do they move steadily?
    print(f'Interior samples (>= {half_window} frames from both track ends) and path straightness:')
    print('  frames        all samples: n   med c   p90 c  |  interior: n   med c   p90 c  |  straightness (median over tracks)')
    for label, m in rows:
        if not m.any():
            continue
        mi = m & (S[:, 5] >= half_window)
        _, first = np.unique(S[m, 3], return_index=True)
        st = S[m, 6][first]  # one value per track
        line = f'  {label:9s} {m.sum():15d} {np.median(S[m, 0]):7.2f} {np.percentile(S[m, 0], 90):7.2f}  |'
        line += (f' {mi.sum():12d} {np.median(S[mi, 0]):7.2f} {np.percentile(S[mi, 0], 90):7.2f}  |'
                 if mi.any() else '         (none)                 |')
        print(line + (f'  {np.nanmedian(st):.2f}' if np.isfinite(st).any() else '  n/a'))


def compute_lambda(rectified_images_dir, filtered_contours_dir, optical_flow_dir, lambda_save_dir,
                   config, phase_speed, speed_bins=0.2, use_flow=True, tracker_kwargs=None,
                   plot=False, plot_every=50, plot_dir=None):

    fps, gsd, spacing = config.fps, config.gsd, config.spacing  #pull the fps, ground sampling distance, and optical flow frame spacing from the configuration
    n_chunks = len(glob.glob(os.path.join(rectified_images_dir, '*.dat'))) #number of data chunks
    # chunks=[1]

    dc = speed_bins # speed bin width in m/s
    c_bins = np.arange(0, 30 + dc, dc) #speed bins

    if plot:
        os.makedirs(plot_dir, exist_ok=True) #if the plotting directory already exists, continue

    tracker_kwargs = dict(tracker_kwargs or {}) #create empty dict for holding keyword arguments
    tracker_kwargs.setdefault('min_age', int(fps * (2 / 3))) #set the default argument, {min_age: 19sec} from kleiss and melville rules for wave tracking.

    min_age = tracker_kwargs['min_age'] # define the minimum age (N_frames) that a wave must survive in order to be tracked and contribute to lambda distribution
    Lambda = np.zeros(len(c_bins) - 1)  # summed over chunks
    all_lengths, n_samples = [], 0  # track diagnostics summed over chunks
    all_samples = []  # per-chunk sample arrays (speed, crest length, age, track id) for the age-vs-speed check
    Area_total = 0.0  #Total area: area of analysis * number of frames -> this is summed over all chunks
    diag = Counter()

    for chunk in tqdm(range(n_chunks), colour='yellow'):
        tqdm.write(f'Processing Chunk: {chunk:04d}')  #iterate through data chunks

        chunk_images, chunk_metadata = load_chunk(rectified_images_dir, chunk)  #load the chunk image data and metadata
        _, h, w = chunk_metadata['memmap_shape']  #shape of the image, height and width
        n_frames = chunk_metadata['core_frame_count']  #number of non overlap frames in the chunk

        valid_pixels = np.sum(get_mask(chunk_images[0]) > 0)  #mask out black areas around the rectified image
        Area_total += valid_pixels * (gsd ** 2) * n_frames  #calculate the area in the image in m**2

        #instantiate a wave tracker per chunk. Chunks can have temporal jumps so no tracks can transition between chunks
        tracker = WaveTracker(fps=fps, gsd=gsd, image_shape=(h, w), phase_speed=phase_speed,
                              spacing=spacing, **tracker_kwargs)

        chunk_contours = loadContours(os.path.join(filtered_contours_dir,
                                                   f'Filtered_contours_chunk_{chunk:04d}.pkl'))   #load the filtered contours from the selected chunk

        # handle the optical flow
        vx_all = vy_all = None
        if use_flow:  #if use_flow is True
            vx_path = os.path.join(optical_flow_dir, f'vx_{chunk:04d}.dat')  #pull the paths to the vx and vy optical flow memmaps for the selected chunk
            vy_path = os.path.join(optical_flow_dir, f'vy_{chunk:04d}.dat')
            shape_path = os.path.join(optical_flow_dir, f'flow_shape_{chunk:04d}.npy')  #path to the shape of the memmap
            if os.path.exists(vx_path) and os.path.exists(vy_path) and os.path.exists(shape_path):
                flow_shape = tuple(np.load(shape_path))
                vx_all = np.memmap(vx_path, dtype='float32', mode='r', shape=flow_shape)  #if all exist, load the data
                vy_all = np.memmap(vy_path, dtype='float32', mode='r', shape=flow_shape)
            else:
                tqdm.write(f'No optical flow found for chunk {chunk:04d}; position-only tracking') #if one does not exist, do not use optical flow

        for frame in tqdm(range(min(n_frames, len(chunk_contours))), colour='blue', leave=False, desc='Frame'):  #for each frame in the contour list
            if vx_all is not None and frame < vx_all.shape[0]:
                vx, vy = vx_all[frame], vy_all[frame]  #load the x and y velocities for the selected frame
            else:
                vx = vy = None  #if the do not exist load nothing

            detections = tracker.update(chunk_contours[frame], vx, vy)  #update the wave tracker with the contours and velocities from the selected frame

            if plot and frame % plot_every == 0:  #if the plot is True
                plot_tracks(chunk_images[frame], detections, frame, chunk, tracker.min_age,   #plot the tracked ellipses
                            os.path.join(plot_dir, f'tracks_chunk{chunk:04d}_frame{frame:05d}.png'))

        # end of chunk: close open tracks and bank this chunk's results
        tracker.finalize()  #flush tracks at the end of the chunk so the crest lengths and speeds are counted
        Lambda += tracker.sum_dl(speed_bins=c_bins)
        all_lengths.extend(tracker.track_lengths)
        n_samples += len(tracker.samples)
        if tracker.samples:
            all_samples.append(np.array(tracker.samples))
        diag.update(tracker.diag)

    lengths = np.array(all_lengths)
    if lengths.size:
        print(f'Tracks: {lengths.size} | median length {np.median(lengths):.0f} frames | '
              f'{np.mean(lengths >= min_age) * 100:.1f}% reach min_age ({min_age}) | '
              f'{n_samples} speed samples')
        print('Track length breakdown: '
              f'1 frame {np.mean(lengths == 1) * 100:.1f}% | 2-4 {np.mean((lengths >= 2) & (lengths < 5)) * 100:.1f}% | '
              f'5-19 {np.mean((lengths >= 5) & (lengths < 20)) * 100:.1f}% | 20+ {np.mean(lengths >= 20) * 100:.1f}%')
    if diag['frames']:
        print(f"Detections per frame: {diag['detections'] / diag['frames']:.1f}")
        skip = ('detections', 'frames')
        shown = {k: v for k, v in diag.items() if k not in skip and not k.startswith('_')}
        total = sum(shown.values())
        print('Track-frame outcomes (share of all track-frames):')
        for k, v in sorted(shown.items(), key=lambda kv: -kv[1]):
            print(f'  {k:34s} {int(v):10d}  {v / total * 100:5.1f}%')
        print(f"Recovered after coasting: {int(diag['_recovered_after_coast'])}")
        if diag['_fv_dot_fv'] > 0:
            slope = diag['_fv_dot_disp'] / diag['_fv_dot_fv']
            print(f'Centroid displacement per frame / flow velocity per frame (slope): {slope:.2f}  (about 1 if units agree)')

    # speed vs age for the SAME tracks: if early speeds are higher, jitter / filter convergence is inflating them
    if all_samples:
        S = np.vstack(all_samples)  # columns: speed, crest length, age, track id
        ids, inv = np.unique(S[:, 3], return_inverse=True)
        max_age = np.zeros(len(ids))
        np.maximum.at(max_age, inv, S[:, 2])
        long_track = max_age[inv] >= 30  # only tracks that live at least 30 frames
        early = long_track & (S[:, 2] >= 5) & (S[:, 2] <= 10)
        late = long_track & (S[:, 2] >= 20) & (S[:, 2] <= 30)
        if early.any() and late.any():
            print(f'Speed vs age for {len(np.unique(S[long_track, 3]))} tracks with 30+ frames (m/s):')
            for name, m in (('age  5-10', early), ('age 20-30', late)):
                print(f'  {name}: median {np.median(S[m, 0]):.2f} | mean {np.mean(S[m, 0]):.2f} | '
                      f'90th pct {np.percentile(S[m, 0], 90):.2f}')

    Lambda /= (Area_total * dc)

    c = 0.5 * (c_bins[:-1] + c_bins[1:])

    b = 7e-5
    rho = 1025
    g = 9.81
    fifth_moment = c ** 5 * Lambda * dc
    epsilon = b * rho * (1 / g) * np.sum(fifth_moment)  # Phillips (1985)
    print(f'Estimated Energy dissipation rate: {epsilon:.4e} W/m^2')
    if all_samples:
        epsilon_by_track_length(np.vstack(all_samples), c_bins, Area_total, b, rho, g,
                                half_window=tracker_kwargs.get('half_window', 5))

    # c**-6 reference line
    idx = 15
    Lambda_c6 = Lambda[idx] * (c / c[idx]) ** (-6)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121)
    ax.loglog(c, Lambda)
    ax.loglog(c, Lambda_c6, '--', label=r'$c^{-6}$')
    ax.set_xlabel('Breaking Wave Speed (m/s)')
    ax.set_ylabel(r'$\Lambda(c)$ (s * m$^{-2}$)')
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

    return Lambda


if __name__ == "__main__":
    config = Config() #initialize configuration parameters

    #paths for directories and saving
    rectified_images_dir = '/mnt/d/Wave_Breaking/rectified_images'
    filtered_contours_dir = '/mnt/d/Wave_Breaking/filtered_contours'
    optical_flow_dir = '/mnt/d/Wave_Breaking/optical_flow'
    lambda_save_dir = '/mnt/d/Wave_Breaking/figs'
    plot_dir = '/mnt/d/Wave_Breaking/figs/ellipse'

    peak_period = 3.88  # s, from the AWAC table for the collection day
    phase_speed = 9.81 * peak_period / (2 * np.pi)  # deep-water dispersion relation

    Lambda = compute_lambda(rectified_images_dir, filtered_contours_dir, optical_flow_dir,
                            lambda_save_dir, config, phase_speed,
                            speed_bins=config.speed_bin_width,
                            use_flow=True, #False = position only kalman filter
                            tracker_kwargs=dict(min_age=10, max_misses=6, max_area_change=2.0, gate_floor=6,
                                                q_acc=0.15, r_pos=2.0, r_vel=0.5, half_window=5),
                            plot=True, plot_every=1, plot_dir=plot_dir)