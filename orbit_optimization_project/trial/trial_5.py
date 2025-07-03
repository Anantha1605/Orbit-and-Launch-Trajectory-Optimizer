from trial_4 import plot_tle
from orbit_optimization_project.orbit_env.constants import MISSION_COVERAGE_RANGE

with open("tle_data.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

tle_sets = []
tle_sets, fig, ax = plot_tle(tle_sets, MISSION_COVERAGE_RANGE, lines)
