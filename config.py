from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=False)
class Config():

    '''
    This class holds all necessary hardcoded variables and paths needed in the various scripts of this repository
    frozen=True makes the dataclass immutable, so that the values cannot be changed after initialization.
    The User should input their own specified values below:
    '''

    # --- CAMERA PARAMETERS ---
    fps:float = 29.97 #frames per second

    # spatial resolution of a pixel in meters
    drone = np.load('./drone/1057metadata.npy')  #FIXME: change this as needed
    dx = float(drone[-2][1])  # dx and dy are the same value
    gsd = dx  # ground sampling distance

    # --- OPTICAL FLOW PARAMETERS ---
    spacing:int = 5 # number of frames to calculate optical flow over. Ex: a = frame0, b = frame4. spacing -> b-a = 5. This helps stabilize the optical flow computation given a high fps.

    # --- PLOTTING PARAMETERS ---
    speed_bin_width:float = 0.2 #speed bin width in m/s

    # --- PHYSICAL CONSTANTS
    g = 9.81 # gravitational acceleration in m/s**2
    rho = 1025 #density of seawater in kg/m**3
    b = 7e-5 # breaking parameter - unit conversion from statistical values to energy dissipation rate

    # --- CONTOUR FILTERING PARAMETERS ---
    area_thresh = 10 #minimum area in pixels that a contour must be
    #TODO: add the other thresholds here

    # --- DIRECTORIES ---
    root:Path = Path('.') #current working directory

    def __post_init__(self):
        self.root = Path(self.root)



