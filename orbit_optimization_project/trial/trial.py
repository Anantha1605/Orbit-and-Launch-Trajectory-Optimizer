import requests
from trial_4 import plot_tle
from orbit_optimization_project.orbit_env.constants import MISSION_COVERAGE_RANGE

file = open('TLE_data_test.txt', 'r')
# --- Step 1: Load TLE data from URL ---
"""
TLE_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'  # Replace with your actual URL
response = requests.get(TLE_URL)
lines = response.text.strip().split('\n')
"""

lines = str(file.read())

# Group lines into sets of 3 (name, line1, line2)
tle_sets = []

#print(lines)

plot_tle(tle_sets, MISSION_COVERAGE_RANGE, lines)

