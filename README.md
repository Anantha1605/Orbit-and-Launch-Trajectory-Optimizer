
# Orbit Optimization using Reinforcement Learning

This project aims to optimize satellite orbits using **Reinforcement Learning (RL)**, leveraging `Stable-Baselines3` (SB3) PPO, and A2C algorithm + MultiInputPolicy and a custom Gymnasium environment. The agent learns to adjust key orbital parameters to satisfy specific mission constraints such as ground target visibility, safety buffer zones, and optimal coverage.



## Project Structure
```text
orbit_optimization_project/
│
├── orbit_env/
│   ├── __init__.py
│   ├── orbit_env.py                     # Custom Gymnasium environment
│   ├── constants.py                     # Physical constants and thresholds
│   ├── active_satellites_orbit_plot.py  # Orbit visualization utilities
│
├── __init__.py                          # Environment registration
```                 


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

