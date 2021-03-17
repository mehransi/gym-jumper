from gym.envs.registration import register

register(
    id="Jumper-v0",
    entry_point="gym_jumper.envs:JumperEnv",
    max_episode_steps=1000,
    reward_threshold=70.0,
)
