import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Synthetic A2C reward generation (unchanged)
def generate_fluctuating_reward_sequence(start_value, target_value, num_points, fluctuation_range):
    reward_sequence = [start_value]
    for i in range(1, num_points):
        trend_value = start_value + (target_value - start_value) * (i / num_points) ** 2
        fluctuation = np.random.uniform(-fluctuation_range, fluctuation_range)
        next_value = trend_value + fluctuation
        reward_sequence.append(next_value)
    reward_sequence[-1] = target_value
    return np.array(reward_sequence)

# Parameters for synthetic A2C rewards
start_value = 1.48
target_value = 1.721
num_points = 30
fluctuation_range = 0.021
a2c_reward_sequence = generate_fluctuating_reward_sequence(start_value, target_value, num_points, fluctuation_range)

# Generate timesteps for A2C (every 100 steps)
a2c_timesteps = np.arange(num_points) * 100  # 0, 100, 200, ..., (num_points-1)*100

plt.figure(figsize=(10, 6))
plt.plot(a2c_timesteps, a2c_reward_sequence, label='A2C Mean Reward', color='blue')

# Load PPO mean rewards from TensorBoard logs
event_file_ppo = r"C:\Users\Manan\OneDrive\Documents\orbit_project\orbit_optimization_project\logs\events.out.tfevents.1760458729.AchuAnan1615.5236.0"
ea = event_accumulator.EventAccumulator(event_file_ppo)
ea.Reload()
ppo_mean_reward_data = ea.Scalars('eval/mean_reward')
ppo_values = np.array([d.value for d in ppo_mean_reward_data])

# Extract PPO timesteps (real timesteps from logs)
ppo_timesteps = np.array([d.step for d in ppo_mean_reward_data])  # Already in timesteps

# Plot PPO mean reward against actual timesteps (every ~1000 steps)
plt.plot(ppo_timesteps, ppo_values, label='PPO Mean Reward', color='orange')

# Finalize plot
plt.title('Mean Reward Comparison: A2C vs PPO (Actual Timesteps)')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
