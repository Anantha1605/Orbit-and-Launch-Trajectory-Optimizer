# constants.py

import numpy as np
import torch

# Orbital parameters
ORBITAL_ELEMENTS = [6778137, 0.01, np.radians(51.6), np.radians(45), np.radians(0)]  # [a, e, i, RAAN, omega]
ORBITAL_ELEMENTS_UNITS = ['km', 'unitless', 'rad', 'rad', 'rad']

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
rocket_mass = 54e4                # Initial mass including fuel -> in kg
fuel_mass_fraction = 0.85

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
coordinates = (28.5721, -80.6480)   # latitude and longitude of ground launch coordinates -> in degrees
launch_lat = np.radians(coordinates[0])
launch_lon = np.radians(coordinates[1])
launch_direction = np.array([0.0, 0.0, 1.0])  # Simplified vertical launch


# Launch azimuth limits for prograde launch
# Works if inclination > latitude (for LEO)
theta = np.clip(np.cos(i) / np.cos(launch_lat), -1, 1)
azimuth_min = np.arcsin(theta)  # radians

# Rotation of earth
earth_rotation_rate = 7.2921159e-5  # rad/s
v_ground = earth_rotation_rate*R*np.cos(launch_lat)


# Thrust and burn parameters for rocket propulsion
T = 7.6e6         # Thrust in Newtons (adjust per case)
Isp = 311         # seconds (typical for chemical rockets)
mdot = T / (Isp * g) # Mass depletion rate in kg/s

fuel_mass = rocket_mass * fuel_mass_fraction  # ~467,000 kg fuel
burn_time = fuel_mass / mdot   # seconds (approx)


# Initial ground velocity in ECI coordinates due to Earth's rotation
# ECI frame: Z is Earth's rotation axis, X points toward vernal equinox
# Launch site: defined by latitude and longitude
# Earth's surface rotates eastward => initial velocity is tangential to surface (east direction)

# Convert launch lat/lon to ECI position
x_eci = R * np.cos(launch_lat) * np.cos(launch_lon)
y_eci = R * np.cos(launch_lat) * np.sin(launch_lon)
z_eci = R * np.sin(launch_lat)
r0_eci = np.array([x_eci, y_eci, z_eci])  # ECI position in meters

# Earth's angular velocity vector (rad/s) — Z-axis
omega_earth_vec = np.array([0.0, 0.0, earth_rotation_rate])

# Velocity in ECI from cross product ω × r
v0_eci = np.cross(omega_earth_vec, r0_eci)

# Convert both to PyTorch tensors with shape [1, 3]
r0_eci_tensor = torch.tensor(r0_eci, dtype=torch.float32).view(1, 3)
v0_eci_tensor = torch.tensor(v0_eci, dtype=torch.float32).view(1, 3)




# Weights [physics, initial, terminal, fuel efficiency]
weights = {
    "phys": 1.5,    # slightly higher to enforce physics
    "init": 1.0,    # initial conditions are important but less than physics
    "term": 1.2,    # terminal must converge, slightly higher
    "fuel": 1.0     # fuel optional, keep reasonable
}

# Normalization values
loss_threshold = 10
individual_loss_threshold = 1e-3

# Scaling factor
phys_scale = 1.0
init_scale = 1.0
term_scale = 1.0
fuel_scale = 1.0


print("\nLoss threshold: ", loss_threshold)
print("\n")