import numpy as np
from constants import R as earth_radius, earth_rotation_rate

def convert_to_eci(lat, lon, t=0):
    """
    Convert launch site to ECI coordinates at time t. t-> time since epoch
    t in seconds from reference epoch (e.g., t=0 = launch)
    """

    # Earth radius is in metres
    # Enter latitude and longitude in radians

    theta = lon + earth_rotation_rate * t
    x = earth_radius * np.cos(lat) * np.cos(theta)
    y = earth_radius * np.cos(lat) * np.sin(theta)
    z = earth_radius * np.sin(lat)

    return np.array([x, y, z], dtype=np.float64)

def rotation_matrix(i, raan, arg_of_perigee):
    # 3 - 1 - 3 Euler rotation matrix -> sequence of rotations about the z-axis, then the x-axis, then again the z-axis.

    R3_W =np.array(
        [
            [np.cos(-raan), -np.sin(-raan), 0],
            [np.sin(-raan), np.cos(-raan), 0],
            [0, 0, 1]
        ]
    )

    R1_i = np.array(
        [
            [1, 0, 0],
            [0, np.cos(-i), -np.sin(-i)],
            [0, np.cos(-i), np.sin(-i)]
        ]
    )

    R3_w = np.array(
        [
            [np.cos(-arg_of_perigee), -np.sin(-arg_of_perigee), 0],
            [np.sin(-arg_of_perigee), np.cos(-arg_of_perigee), 0],
            [0, 0, 1]
        ]
    )

    return R3_W @ R1_i @ R3_w

def position_on_orbit(a, e, true_anomaly):
    # to return -> position in perifocal coordinates

    """
    What is Perifocal coordinates?

    Perifocal coordinates are a special 3D coordinate system used to describe a spacecraft’s position and motion in its orbital plane.

        - The origin is at the center of the planet or star.
        - The x-axis points to the closest point in the orbit (called periapsis).
        - The y-axis points in the direction the spacecraft moves at periapsis.
        - The z-axis points up from the orbital plane (perpendicular to it).

    This system makes orbit math easier, especially for elliptical orbits.
    """

    r = a * (1 + np.power(np.e, 2)) / (1 + np.e * np.cos(true_anomaly))

    return np.array([
        r * np.cos(true_anomaly),
        r * np.sin(true_anomaly),
        0
    ])

def compute_optimal_true_anomaly(orbital_elements, launch_lat, launch_lon, t_launch):
    # assuming latitude and longitude are in radians
    a = orbital_elements[0]
    e = orbital_elements[1]
    i = orbital_elements[2]
    raan = orbital_elements[3]
    arg_of_perigee = orbital_elements[4]

    # ECI conversion of launch coordinates
    launch_eci = convert_to_eci(launch_lat, launch_lon, t_launch)
    launch_dir = launch_eci / np.linalg.norm(launch_eci)

    # converting perifocal to ECI using rotation matrix
    Q_p_to_i = rotation_matrix(i, raan, arg_of_perigee)

    # Trialing true anomalies
    best_angle = np.inf
    best_nu = 0  # Trialing and validating true anomalies

    for nu in np.linspace(0, 2*np.pi, 50): # Trialling 1000 values, add more for fine tuning
        r_perifocal = position_on_orbit(a, e, nu)
        r_eci = Q_p_to_i @ r_perifocal
        r_dir = r_eci / np.linalg.norm(r_eci)

        angle = np.arccos(np.clip(np.dot(r_dir, launch_dir), -1, 1))
        if angle < best_angle:
            best_angle = angle
            best_nu = nu

    return best_nu


# Trial code -> remove at last
"""
if __name__ == "__main__":
    from constants import ORBITAL_ELEMENTS, launch_lon, launch_lat
    from datetime import datetime
    from astropy.time import Time

    # Get current time (or time of launch)
    t = datetime.now()

    # Convert to seconds since J2000 using astropy
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch
    time_difference = t - j2000_epoch  # Calculate time difference
    t_launch_seconds = time_difference.total_seconds()  # Convert to seconds

    # Now pass the numeric value (t_launch_seconds) to your function
    nu = compute_optimal_true_anomaly(ORBITAL_ELEMENTS, launch_lat, launch_lon, t_launch_seconds)

    print("Optimal true anomaly:", nu)
"""


