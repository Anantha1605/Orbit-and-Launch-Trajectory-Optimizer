from tensorboard.backend.event_processing import event_accumulator

files = [
    r"C:\Users\Manan\OneDrive\Documents\orbit_project\orbit_optimization_project\logs\events.out.tfevents.1760451459.AchuAnan1615.15560.0",
    r"C:\Users\Manan\OneDrive\Documents\orbit_project\orbit_optimization_project\logs\events.out.tfevents.1760458729.AchuAnan1615.5236.0"
]

for f in files:
    print(f"\n=== Checking file: {f} ===")
    ea = event_accumulator.EventAccumulator(f)
    ea.Reload()
    print("Available tags:", ea.Tags()['scalars'])  # Print first 15 scalar tags
