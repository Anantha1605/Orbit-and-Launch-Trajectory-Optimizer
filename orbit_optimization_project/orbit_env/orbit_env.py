#This file contains all the code for the custom orbit environment

from gymnasium.spaces import Discrete, Box, Dict
from gymnasium import Env
import requests
import numpy as np
from .active_satellites_orbit_plot import plot_tle
from .constants import EARTH_RADIUS, MISSION_COVERAGE_RANGE, TLE_SOURCE, TARGET_GROUND_COORDINATES, LEO_E_MAX
import random
import logging

logging.basicConfig(
    filename='training.log',       # File to write logs to
    filemode='a',                  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO             # Minimum level to log (INFO, WARNING, ERROR)
)

"""
Observation space
    Keplerian elements (orbital elements)
        a — Semi-major axis
        e — Eccentricity
        i —  inclination
        Ω — RAAN 
        ω — Argument of Perigee 
        v — True anomaly (start) 
    Ground_target_validation -> bool
    Coverage_error -> bool
    Safety_buffer_distance (in Kms) -> bool 
    

Action Space
    Keplerian elements (orbital elements)
    a — Semi-major axis
    e — Eccentricity
    i —  inclination
    Ω — RAAN 
    ω — Argument of Perigee 
    V {True anomaly (start)} — The true anomaly is excluded from the action space, as the model aims to compute the orbit (satisfying applied constraints), and not the placement or position of an object (or satellite) in that orbit.

Reward Function
    Inclination → Ensures latitude coverage.
    RAAN → Aligns the orbit with the longitude of the target.
    -> Positive reward if at any time of the day, the orbit is directly overhead the target_ground_coordinates\ 

"""

class OrbitEnv(Env):
    """
    Custom Gym Environment for designing satellite orbits using Keplerian elements.
    The environment aims to align a satellite's orbit to ground targets while avoiding conflicts.
    """

    def __init__(self):
        # TLE Data from URL
        TLE_URL = TLE_SOURCE
        response = requests.get(TLE_URL)
        if response.status_code != 200: raise RuntimeError("Failed to fetch TLE data.")

        lines = response.text.strip().split('\n')

        self.steps = 0

        # Group lines into sets of 3 (name, line1, line2)
        self.tle_sets = []

        self.coverage_error_range = MISSION_COVERAGE_RANGE  # Mission parameter altitude
        self.tle_sets, self.fig, self.ax = plot_tle(self.tle_sets, self.coverage_error_range, lines)

        # Orbital elements (Keplerian elements) Initialization
        a = random.uniform(EARTH_RADIUS + self.coverage_error_range[0],
                   EARTH_RADIUS + self.coverage_error_range[1])
        e = random.uniform(0.0, 0.01)
        i = random.uniform(0, np.pi / 2)
        raan = random.uniform(0, 2*np.pi)
        arg_of_perigee = random.uniform(0, 2*np.pi)

        self.current_orbit = np.array([a, e, i, raan, arg_of_perigee], dtype=np.float32)
        self.ground_target = TARGET_GROUND_COORDINATES

        # Action space -> (a, e, i, raan, arg_of_perigee) {orbital elements}
        self.action_space : Box = Box(
            low = np.array(
                [
                    float(EARTH_RADIUS + self.coverage_error_range[0]),   # a
                    0.0,                                                  # e
                    0.0,                                                  # i
                    0.0,                                                  # raan
                    0.0,                                                  # argument of perigee
                ], dtype=np.float32
            ),
            high = np.array(
                [
                    float(EARTH_RADIUS + self.coverage_error_range[1] + 100),      # a
                    LEO_E_MAX,                                                     # e
                    np.pi,                                                         # i
                    2 * np.pi,                                                     # raan
                    2 * np.pi,                                                     # argument of perigee
                ], dtype=np.float32
            ),
            dtype=np.float32
        )

        # Observation space
        self.observation_space : Dict = Dict(
            {
                'orbital_elements' : Box(
                                            low = np.array(
                                                [
                                                    float(EARTH_RADIUS + self.coverage_error_range[0]),   # a
                                                    0.0,                                                  # e
                                                    0.0,                                                  # i
                                                    0.0,                                                  # raan
                                                    0.0,                                                  # argument of perigee
                                                ], dtype=np.float32
                                            ),
                                            high = np.array(
                                                [
                                                    float(EARTH_RADIUS + self.coverage_error_range[1] + 100),      # a
                                                    LEO_E_MAX,                                                     # e
                                                    np.pi,                                                         # i
                                                    2 * np.pi,                                                     # raan
                                                    2 * np.pi,                                                     # argument of perigee
                                                ], dtype=np.float32
                                            ),
                                            dtype=np.float32
                                        ),

                'Ground_target_valid' : Discrete(2),    # returns 0 if orbit doesn't pass over the target ground coordinates, else 1
                'Coverage_error' : Discrete(2),         # returns 0 if altitude of orbit not in mission specified range
                'safety_buffer_distance' : Discrete(2), # returns 0 if safety distance is not maintained.
            }
        )


    def step(self, action):
        from .constants import MAX_STEPS
        self.steps += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Action -> [a, e, i, raan, arg_of_perigee]

        self.current_orbit = np.array(action, dtype=np.float32)

        # Reward
        reward, info = self.reward(self.current_orbit)

        # New observation
        coverage_valid = int(self.check_coverage_error(self.current_orbit))
        safety_valid = int(self.check_safety_buffer_distance(self.current_orbit))
        _, target_valid = self.check_ground_target_validity(self.current_orbit)

        observation = {
            "orbital_elements": np.array(self.current_orbit, dtype=np.float32),
            "Ground_target_valid": int(target_valid),
            "Coverage_error": coverage_valid,
            "safety_buffer_distance": safety_valid,
        }

        done = info.get('all_objectives_met', False)
        truncated = self.steps >= MAX_STEPS


        return observation, reward, done, truncated, info

    def reset(self, *, seed = None, options=None):
        a = random.uniform(EARTH_RADIUS + self.coverage_error_range[0],
                           EARTH_RADIUS + self.coverage_error_range[1])
        e = random.uniform(0.0, 0.01)
        i = random.uniform(0, np.pi / 2)
        raan = random.uniform(0, 2 * np.pi)
        arg_of_perigee = random.uniform(0, 2 * np.pi)

        self.current_orbit = np.array([a, e, i, raan, arg_of_perigee], dtype=np.float32)
        self.steps = 0

        # constraint validation
        safety_valid = self.check_safety_buffer_distance(self.current_orbit)
        coverage_valid = self.check_coverage_error(self.current_orbit)
        _, target_valid = self.check_ground_target_validity(self.current_orbit)

        observation = {
            "orbital_elements": np.array(self.current_orbit, dtype=np.float32),  # True Anomaly is irrelevant
            "Ground_target_valid": int(target_valid),
            "Coverage_error": coverage_valid,
            "safety_buffer_distance": safety_valid,
        }

        info = {}

        return observation, info

    def render(self):
        pass

    def check_coverage_error(self, orbit):
        """
        Check if the orbit's altitude range (perigee to apogee) falls within mission parameters.
        For LEO satellites, check both perigee and apogee altitudes.
        """
        a = orbit[0]  # Semi-major axis (in meters)
        e = orbit[1]  # Eccentricity

        # Calculate perigee and apogee distances from Earth center
        perigee_distance = a * (1 - e)  # Closest approach
        apogee_distance = a * (1 + e)  # Farthest distance

        # Convert to altitudes above Earth surface
        perigee_altitude = perigee_distance - EARTH_RADIUS
        apogee_altitude = apogee_distance - EARTH_RADIUS

        min_alt = self.coverage_error_range[0]
        max_alt = self.coverage_error_range[1]

        # Both perigee and apogee must be within mission altitude range
        perigee_valid = min_alt <= perigee_altitude <= max_alt
        apogee_valid = min_alt <= apogee_altitude <= max_alt

        return perigee_valid and apogee_valid
    def check_safety_buffer_distance(self, orbit):
        """
        We are only considering the Low Earth orbits. LEO have a very small eccentricity value, and can thus
        be considered as circular orbit.

        Thus, the semi-major axis becomes the radius. And the safety buffer distance validation can be done
        by comparing the semi-major axis (a) of 2 orbits.

        The equivalent length of the radius or the semi-major axis along the equatorial plane is not needed in
        the comparison. Since it is considered as an almost circular orbit, both the orbits are considered as
        concentric circles rotated along the vertical axis, horizontal axis, and/or the vernal equinox.

        Inclination (i) and RAAN represent the orbital plane orientation. Even if two orbits have different radii,
        if their planes are very different (large differences in i or raan), the orbits might never come close to
        each other in space. Conversely, if i and raan are very similar, and the radii are close, orbits might be
        dangerously near.
        """
        from .constants import EARTH_MU, SAFETY_BUFFER_DISTANCE

        a1, _, i1, raan1, _ = orbit

        # parsing the orbital TLE data set
        for name, line1, line2 in self.tle_sets:
            try:
                n = float(line2[43:51])  # Mean motion
                n_rad = 2.0 * np.pi * n / 86400
                a2 = (EARTH_MU / np.power(n_rad, 2)) ** (1 / 3)  # semi-major axis -> radius of 'circular' orbit

                # inclination and RAAN in degrees, converted to radians
                i2 = np.radians(float(line2[8:16].strip()))
                raan2 = np.radians(float(line2[17:25].strip()))

                '''
                a1, a2 -> radii (semi-major axis)
                i1, i2 -> inclination (in radians)
                raan1, raan2 -> angle of ascending node with the vernal equinox (in radians)
                cos_theta = angle for safety distance
                d_min = minimum distance between the two orbits
                '''

                cos_theta = np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(raan1 - raan2)
                # Minimum distance between two circular orbits
                if cos_theta >= 0:  # Planes intersect at acute angle
                    d_min = abs(a1 - a2)
                else:  # Planes intersect at obtuse angle
                    d_min = np.sqrt(a1 ** 2 + a2 ** 2 - 2 * a1 * a2 * abs(cos_theta))

                # Buffer distance Check
                if d_min < SAFETY_BUFFER_DISTANCE: return False # orbit too close -> unsafe

            except Exception as e:
                logging.error("Error: %s", str(e))
                continue
        return True # Safety distance maintained w.r.t all orbits

    @staticmethod
    def validate_orbital_elements(orbit):
        """Validate orbital elements for physical constraints"""
        a, e, i, raan, arg_perigee = orbit

        # Semi-major axis must be greater than Earth radius
        if a <= EARTH_RADIUS:
            logging.error(f"Semi-major axis {a} must be > Earth radius {EARTH_RADIUS}")
            return False

        # Eccentricity constraints for LEO
        if not (0 <= e < 1):
            logging.error(f"Eccentricity {e} must be in range [0, 1)")
            return False

        # Inclination constraints
        if not (0 <= i <= np.pi):
            logging.error(f"Inclination {i} must be in range [0, π]")
            return False

        # RAAN constraints
        if not (0 <= raan <= 2 * np.pi):
            logging.error(f"RAAN {raan} must be in range [0, 2π]")
            return False

        # Argument of perigee constraints
        if not (0 <= arg_perigee <= 2 * np.pi):
            logging.error(f"Argument of perigee {arg_perigee} must be in range [0, 2π]")
            return False

        return True

    @staticmethod
    def check_ground_target_validity(orbit):
        """
        Using Satellite Sub point approach to determine ground target validity
        """
        from .constants import GROUND_TARGET_VARIANCE_THRESHOLD, EARTH_RADIUS
        a, e, i_rad, raan_rad, arg_perigee = orbit

        lat_rad = np.radians(TARGET_GROUND_COORDINATES[0])
        lon_rad = np.radians(TARGET_GROUND_COORDINATES[1])

        x = EARTH_RADIUS * np.cos(lat_rad) * np.cos(lon_rad)
        y = EARTH_RADIUS * np.cos(lat_rad) * np.sin(lon_rad)
        z = EARTH_RADIUS * np.sin(lat_rad)
        target_ECI = np.array([x, y, z], dtype = np.float32)     # [x, y, z] -> ECI coordinates for ground target



        nx = np.sin(i_rad) * np.sin(raan_rad)
        ny = -np.sin(i_rad) * np.cos(raan_rad)
        nz = np.cos(i_rad)
        orbital_normal = np.array([nx, ny, nz], dtype = np.float32)  # [nx, ny, nz] -> ECI coordinates of Normal to orbital plane

        # Distance between target and normal
        distance_to_plane = abs(np.dot(target_ECI, orbital_normal))

        return distance_to_plane, distance_to_plane <= GROUND_TARGET_VARIANCE_THRESHOLD

    def reward(self, orbit):
        """
        Refined reward function for LEO orbit optimization.
        Applies soft penalties and scaled rewards for better convergence.
        """
        from .constants import SAFETY_DISTANCE_WEIGHT, TARGET_VALIDITY_WEIGHT, COVERAGE_ERROR_WEIGHT, EARTH_RADIUS, GROUND_TARGET_VARIANCE_THRESHOLD

        if not self.validate_orbital_elements(orbit):
            return -1000, {'error': 'Invalid orbital elements', 'reward': -1000}


        a, e, i, raan, arg_perigee = orbit
        mean_alt = a - EARTH_RADIUS

        # --- Coverage (Altitude) Reward ---
        min_alt, max_alt = self.coverage_error_range

        if min_alt <= mean_alt <= max_alt:
            coverage_reward = 100
            coverage_penalty = 0
        else:
            coverage_reward = 0
            coverage_penalty = min(300, abs(mean_alt - np.clip(mean_alt, min_alt, max_alt)) / 100 * 25)

        # --- Safety Distance Reward ---
        if self.check_safety_buffer_distance(orbit):
            safety_reward = 100
            safety_penalty = 0
        else:
            safety_reward = 0
            safety_penalty = 300

        # --- Ground Target Validity Reward ---
        try:
            distance_to_target, target_validity = self.check_ground_target_validity(orbit)

            if target_validity:
                target_reward = max(0, 100 - (distance_to_target / GROUND_TARGET_VARIANCE_THRESHOLD) * 50)
                target_penalty = 0
            else:
                target_reward = 0
                target_penalty = min(300, distance_to_target / 100 * 25)

        except Exception as e:
            logging.error(f"Ground target error: {e}")
            distance_to_target = float('inf')
            target_reward = 0
            target_penalty = 300

        # --- Total Reward Calculation ---
        total_reward = (
                COVERAGE_ERROR_WEIGHT * coverage_reward +
                SAFETY_DISTANCE_WEIGHT * safety_reward +
                TARGET_VALIDITY_WEIGHT * target_reward
        )

        total_penalty = (
                COVERAGE_ERROR_WEIGHT * coverage_penalty +
                SAFETY_DISTANCE_WEIGHT * safety_penalty +
                TARGET_VALIDITY_WEIGHT * target_penalty
        )

        final_reward = total_reward - total_penalty
        objectives_met = False

        # Bonus for meeting all 3 objectives
        if coverage_reward > 0 and safety_reward > 0 and target_reward > 0:
            final_reward += 100
            objectives_met = True

        diagnostics = {
            'reward': final_reward,
            'mean_altitude_km': mean_alt,
            'coverage_reward': coverage_reward,
            'coverage_penalty': coverage_penalty,
            'safety_reward': safety_reward,
            'safety_penalty': safety_penalty,
            'target_reward': target_reward,
            'target_penalty': target_penalty,
            'distance_to_target_km': distance_to_target,
            'all_objectives_met': objectives_met
        }

        return final_reward, diagnostics








