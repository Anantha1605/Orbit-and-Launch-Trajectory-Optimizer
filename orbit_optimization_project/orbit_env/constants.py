import math

import numpy as np

#Earth data
EARTH_RADIUS = 6378.137
EARTH_MU = 398600.4418
EARTH_J2 = 1.08263e-3

#time
SECONDS_PER_DAY = 86400.0
CONVERT_TO_RAD = 180.0 / np.pi

# Orbit related constants
TRUE_ANOMALY = np.pi/2                   # some fixed constant -> position of satellite in orbit is not important
LEO_E_MAX = 0.1                            # Max e = 0.25, >0.25 -> highly elliptical -> MEO

# Mission parameters
TARGET_GROUND_COORDINATES = (12.3, 34.6)   # Target ground coordinates -> (lat, lon) {in degrees}
GROUND_TARGET_VARIANCE_THRESHOLD = 5.0     # The maximum distance an orbits subpoint can be from the target (in kms)
REVISIT_FREQ_RANGE = (1, 12)               # passes/day
SAFETY_BUFFER_DISTANCE = 50.0              # km
MISSION_DURATION = 10.0                    # in years
MISSION_COVERAGE_RANGE = (700.0, 1500.0 + SAFETY_BUFFER_DISTANCE)   # km


# Reward function priority weights      {modify as per requirement}
SAFETY_DISTANCE_WEIGHT = 1
COVERAGE_ERROR_WEIGHT = 1
TARGET_VALIDITY_WEIGHT = 1
MAX_STEPS = 500                            # Maximum steps per episode



#TLE data source
TLE_SOURCE = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'

MAX_RENDER = 1000 # set a max number of satellites to render to avoid visual clutter