from gymnasium.envs.registration import register

register(
    id='OrbitEnv-v0',
    entry_point='orbit_optimization_project.orbit_env:OrbitEnv',
    max_episode_steps=5000
)
