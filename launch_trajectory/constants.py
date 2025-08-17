# constants.py

import numpy as np

# Orbital parameters
ORBITAL_ELEMENTS = [7078.1372, 0.1, 3.1415915, 0.011398683, 6.2831011]  # [a, e, i, RAAN, omega]
ORBITAL_ELEMENTS_UNITS = ['km', 'unitless', 'rad', 'rad/s', 'rad']

a_km = ORBITAL_ELEMENTS[0]
a = a_km * 1000  # convert to meters
e = ORBITAL_ELEMENTS[1]
i = ORBITAL_ELEMENTS[2]
raan = ORBITAL_ELEMENTS[3]
arg_of_perigee = ORBITAL_ELEMENTS[4]


# Earth and gravitational constants
G = 6.67430e-11          # m^3/kg/s^2
M = 5.972e24             # kg
mu = G * M               # m^3/s^2


R_km = 6378.137          # Earth radius -> in km
R = R_km * 1000          # Earth radius -> in meters

# Rocket
rocket_mass = 75e4                # Initial mass including fuel -> in kg

# Standard gravity
g = 9.80665             # m/s^2

# Useful derived values
F_g = G * M * rocket_mass / R**2  # Gravitational force at Earth's surface
escape_velocity = np.sqrt(2 * G * M / a)  # m/s
orbital_velocity = np.sqrt(G * M / a)

"""
Assumption:- The orbit is LEO, thus the eccentricity is ~0.025, a nearly circular orbit -> semi major axis (a) = radius (r)
"""

# Ground Launch Coordinates
coordinates = (1.3, 31.3)   # latitude and longitude of ground launch coordinates -> in degrees
launch_lat = np.radians(coordinates[0])
launch_lon = np.radians(coordinates[1])
launch_direction = np.array([0.0, 0.0, 1.0])  # Simplified vertical launch


# Launch azimuth limits for prograde launch
# Works if inclination > latitude (for LEO)
theta = np.clip(np.cos(i) / np.cos(launch_lat), -1, 1)
azimuth_min = np.arcsin(theta)  # radians

# Rotation of earth
earth_rotation_rate = 7.2921159e-5  # rad/s


# Thrust and burn parameters for rocket propulsion
T = 3.5e6         # Thrust in Newtons (adjust per case)
Isp = 300         # seconds (typical for chemical rockets)
mdot = T / (Isp * g) # Mass depletion rate in kg/s
burn_time = rocket_mass / mdot  # seconds (approx)


# Weights [physics, initial, terminal, fuel efficiency]
weights = {
    "phys": 5.0,
    "init": 10.0,
    "term": 10.0,
    "fuel": 5.0
}
loss_threshold = 100 # absolute maximum value of loss function for early stopping the model.