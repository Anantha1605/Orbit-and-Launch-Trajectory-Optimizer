def plot_tle(tle, error_range, lines):
    import numpy as np
    from datetime import datetime, timezone
    import matplotlib.pyplot as plt
    from skyfield.api import EarthSatellite, load
    from orbit_optimization_project.orbit_env.constants import EARTH_MU, EARTH_RADIUS


    # returns true if either the semi-major axis, or semi-minor axis distance is in the range of coverage_error
    def accepted_range(tle_line2, coverage_error_range):
        # EARTH_MU = 398600.4418 -> earth gravitational parameter

        n = float(tle_line2[43:51])  # Mean motion
        e_str = tle_line2[26:33]  # e.g., "0005578"
        eccentricity = float("0." + e_str)

        n_rad = 2.0 * np.pi * n / 86400
        a = (EARTH_MU / np.power(n_rad, 2)) ** (1 / 3)

        if coverage_error_range[0] <= a <= coverage_error_range[1]:
            return True
        else:
            b = a * np.sqrt(1 - np.power(eccentricity, 2))

            if coverage_error_range[0] <= b <= coverage_error_range[1]:
                return True

        return False

    # start of screening
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()  # line2[43:51] -> mean motion, line2[26:33] -> eccentricity
            if accepted_range(line2, error_range):
                tle.append((name, line1, line2))
    # tle now contains all the satellites whose orbits are of concern
    # I.e. the length of semi-major or semi-minor axis is within the given range of altitude (mission constraint)

    # Load timescale
    ts = load.timescale()

    # Current time in UTC (non-deprecated)
    now = datetime.now(timezone.utc)
    time = ts.from_datetime(now)

    # Prepare 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Satellite Orbits Around 3D Earth", fontsize=14)

    # Plot the Earth
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x = EARTH_RADIUS * np.cos(u) * np.sin(v)
    y = EARTH_RADIUS * np.sin(u) * np.sin(v)
    z = EARTH_RADIUS * np.cos(v)
    ax.plot_surface(x, y, z, cmap='Blues', alpha=0.5)

    # Step 2: Propagate and Plot Satellite Orbits
    for name, line1, line2 in tle[:3]:
        try:
            satellite = EarthSatellite(line1, line2, name, ts)
            # Sample points across one orbit
            t = ts.utc(now.year, now.month, now.day, np.linspace(0, 24, 2800))
            geocentric = satellite.at(t)
            pos = geocentric.position.km  # (x, y, z)
            ax.plot(pos[0], pos[1], pos[2], label=name[:10], linewidth=1.5)
        except Exception as e:
            print(f"Error with satellite {name}: {e}")

    # Axis configuration
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_box_aspect([1, 1, 1])

    # Only show legend if there are satellites to display
    if len(tle) > 0:
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()

    print(f"Plotted {len(tle)} satellites within altitude range {error_range}\n")

    return tle, fig, ax

