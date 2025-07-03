import requests

file = open('TLE_data_test.txt', 'w')

# --- Step 1: Load TLE data from URL ---
TLE_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'  # Replace with your actual URL
response = requests.get(TLE_URL)
lines = response.text.strip().split('\n')

# Group lines into sets of 3 (name, line1, line2)
tle_sets = []

file.write(str(lines))