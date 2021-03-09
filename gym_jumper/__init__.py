from gym.envs.registration import register

register(
    id="Jumper-v0",
    entry_point="gym_jumper.envs:JumperEnv"
)
