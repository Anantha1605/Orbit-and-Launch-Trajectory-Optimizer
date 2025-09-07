
# Orbit Optimization using Reinforcement Learning and Physics-Informed Nueral Networks for optimal orbit insertion trajectory

This project aims to optimize satellite orbits using **Reinforcement Learning (RL)**, leveraging `Stable-Baselines3` (SB3) PPO, and A2C algorithm + MultiInputPolicy and a custom Gymnasium environment. The agent learns to adjust key orbital parameters to satisfy specific mission constraints such as ground target visibility, safety buffer zones, and optimal coverage.

The, then obtained orbit (orbital parameters) from the RL framework is utilized by the Physics-informed Neural Network to compute an optimal true anomaly for point of insertion into the orbit, using ground launch coordinates. The PINN computes collocation points from the ground launch co-ords to the orbital point of insertion, by minimizing loss functions(satisfying laws of motion, drag, and thrust directions), and minimizing fuel usage, All incorporated as parts of the loss function intended to be minimized. 


## Project Structure
```text
.
├── orbit_optimization_project/           # Satellite orbit optimization using RL
│   ├── orbit_env/
│   │   ├── __init__.py
│   │   ├── orbit_env.py                 # Custom Gymnasium environment
│   │   ├── constants.py                 # Physical constants and thresholds
│   │   ├── active_satellites_orbit_plot.py  # Orbit visualization utilities
│   ├── __init__.py                      # Environment registration
│   ├── PPO_RL_Model_training.ipynb      # PPO-based training
│   ├── A2C_RL_Model_training.ipynb      # A2C-based training
│   ├── A2C_MLPPolicy_RL_Model_training.ipynb
│   ├── A2C_MultiInputPolicy_RL_Model_training.ipynb
│   ├── RL_Model_training.ipynb
│   ├── a2c_orbit_prediction_RL_model(2000_time_steps).zip
│   ├── ppo_orbit_prediction_RL_model(61440_time_steps).zip
│   ├── orbit_vec_normalize.pkl
│   ├── a2c_orbit_vec_normalize.pkl
│   ├── training.log
│
├── launch_trajectory/                   # Launch trajectory prediction using PINNs
│   ├── launch_trajectory_optimizer.py   # Core PINN model + optimizer
│   ├── constants.py                     # Physical constants for trajectory modeling
│   ├── launch_trajectory_model_epoch_729.pth  # Trained model weights
│   ├── trajectory_collocation_points.txt      # Training collocation points
│   ├── model_prediction_run_8.png       # Sample prediction visualization
│   ├── model_prediction_run_9.png
│   ├── model_prediction_run_10.png
│   ├── training_log_run_8.png           # Training logs and convergence plots
│   ├── training_log_run_9.png
│   ├── training_log_run_10.png
│   ├── prediction.txt                   # Predicted trajectory outputs
│
├── TLE_data.txt                         # Two-line element sets for orbital state initialization
├── training_data.txt                   # Dataset for PINN training
├── README.md                           # Project documentation

```                 
# 1. Orbit Optimization using RL

## Objective

Design an RL-based system where agent **learns to configure orbital parameters** such as:

* Semi-major axis (`a`)
* Eccentricity (`e`)
* Inclination (`i`)
* Right Ascension of Ascending Node (`RAAN`)
* Argument of Perigee (`ω`)

to **maximize a reward function** derived from:

* Valid coverage of ground targets.
* Safety buffer from active satellites.
* Minimizing deviation from target orbit objectives.



## Reinforcement Learning Setup

* **Framework**: Stable-Baselines3
* **Algorithm**: Proximal Policy Optimization (PPO)
* **Environment**: Custom Gymnasium-compatible environment (`orbit_env.OrbitEnv`)
* **Observation Space**: Dict space with orbital elements and validity flags.
* **Action Space**: Continuous Box space `[a, e, i, RAAN, ω]`.

### Custom Reward Function

Reward is based on:

* Ratio of objectives met.
* Penalties for unsafe distances and coverage failure.
* Smoothly continuous formulation to guide learning even for near-misses.



## Customizations from Default SB3

###  `policy_kwargs` Modifications:

```python
policy_kwargs = {
    'activation_fn': "LeakyReLU",
    'net_arch': {
        'pi': [512, 256, 128],
        'vf': [512, 256, 128],
    },
    'ortho_init': True
}
```

#### Impact:

* **Deep network**: Allows for capturing more complex patterns and nonlinearities in orbital mechanics.
* **LeakyReLU**: Prevents dying neuron problem common with ReLU, especially beneficial in sparse-reward problems like orbital optimization.
* **Orthogonal Initialization**: Helps in preserving variance across layers — important for stable PPO learning.

These changes make the network more expressive and robust to challenging optimization tasks.



**Core Libraries:**

* `gymnasium`
* `numpy`
* `matplotlib`
* `stable-baselines3`
* `torch`
* `tensorflow`

# 2. Launch Trajectory Optimization using PINNs

## Objective
Predict and optimize launch vehicle trajectories by incorporating physics constraints directly into the neural network loss function:
* Minimize time-to-orbit 
* Maximize fuel efficiency 
* Achieve desired orbit insertion accuracy

## Approach
* Physics-Informed Neural Networks (PINNs):
    * Integrates governing rocket dynamics equations directly into the loss function. 
    * Balances between minimizing data loss and physics loss. 
* Uses normalized inputs to stabilize learning and improve convergence. 
* Employs custom loss balancing for:
  * Thrust vector dynamics 
  * Gravity turn maneuver modeling 
  * Fuel consumption constraints
```python
# Weights [physics, initial, terminal, fuel efficiency]
weights = {
    "phys": 5,   # already normalized
    "init": 2,  #  small
    "term": 4.5,   # high priority
    "fuel": 1   # optional, small
}
```

## Training and Output:
* Collocation Points: Sampled along trajectory for enforcing dynamics 
* Prediction Visualizations: model_prediction_run_*.png 
* Training Logs: training_log_run_*.png 
* Final Trajectory Output: prediction.txt

**Core Libraries:**

* `numpy`
* `matplotlib`
* `torch`
