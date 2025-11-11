import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Path to your PPO event file
event_file_ppo = r"C:\Users\Manan\OneDrive\Documents\orbit_project\orbit_optimization_project\logs\events.out.tfevents.1760458729.AchuAnan1615.5236.0"

# Initialize EventAccumulator
ea = event_accumulator.EventAccumulator(event_file_ppo)
ea.Reload()

# Tag for value function loss
value_loss_tag = 'train/value_loss'

# Get the scalar data for value loss
data = ea.Scalars(value_loss_tag)
values = np.array([d.value for d in data])

# Plot value function loss
plt.figure(figsize=(10, 6))
plt.plot(values, label=value_loss_tag, color='green', linewidth=2)
plt.title('Value Function Loss Over Time')
plt.xlabel('Timesteps (relative)')
plt.ylabel('Value Function Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
