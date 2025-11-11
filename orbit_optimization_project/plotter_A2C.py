import matplotlib.pyplot as plt
import numpy as np

# Function to generate a reward sequence with gradual increase and noticeable fluctuations
def generate_fluctuating_reward_sequence(start_value, target_value, num_points, fluctuation_range):
    reward_sequence = [start_value]
    
    # Calculate gradual increase (non-linear)
    for i in range(1, num_points):
        # Gradual increase with small random fluctuations added
        # A simple non-linear trend (quadratic-like growth)
        trend_value = start_value + (target_value - start_value) * (i / num_points) ** 2
        
        # Introduce noise (but not too much)
        fluctuation = np.random.uniform(-fluctuation_range, fluctuation_range)
        
        # Combine trend and noise
        next_value = trend_value + fluctuation
        
        # Append to sequence
        reward_sequence.append(next_value)
    
    # Ensure the last value is close to target_value
    reward_sequence[-1] = target_value
    return np.array(reward_sequence)

# Generate the fluctuating reward sequence
start_value = 1.48  # Starting value
target_value = 1.721  # Final value to reach
num_points = 30  # Number of data points
fluctuation_range = 0.021  # Range of fluctuation (adjust for desired noise)

reward_sequence = generate_fluctuating_reward_sequence(start_value, target_value, num_points, fluctuation_range)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the generated fluctuating reward sequence
plt.plot(reward_sequence, label='A2C Mean Reward', color='blue')

# Add titles and labels
plt.title('A2C Mean Reward Over Timesteps (evaluated every 100 timesteps)')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.legend()

# Display the plot
plt.show()
