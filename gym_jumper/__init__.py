from gym.envs.registration import register

register(
    id="Jumper-v0",
    entry_point="gym_jumper.envs:JumperEnv"
)

register(
    id="Jumper-v1",
    entry_point="gym_jumper.envs:JumperEnv",
    kwargs=dict(force_mag=5)
)
