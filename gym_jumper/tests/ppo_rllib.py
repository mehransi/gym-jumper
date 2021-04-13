import time

import ray
from gym_jumper.envs import JumperEnv
from ray.tune import tune
from ray.rllib.agents.ppo import PPOTrainer

ray.init()

stop = {
    # "training_iteration": 50,
    # "timesteps_total": 25000,
    "episode_reward_mean": 60.0,
}

config = {
    "env": JumperEnv,
    "framework": "torch",
    "env_config": {"ball_radius": 3}
}
analysis = tune.run(
    PPOTrainer, config=config, stop=stop, name="gym_jumper", checkpoint_at_end=True, verbose=0
)

env = JumperEnv(config=config["env_config"])

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
agent = PPOTrainer(config=config, env=env.__class__)
analysis.default_mode = "max"
analysis.default_metric = "episode_reward_mean"
agent.restore(analysis.best_checkpoint)
print("checkpoint path is:", analysis.best_checkpoint)
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    env.render()
    time.sleep(0.03)

ray.shutdown()
