import random

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class JumperEnv(gym.Env):
    """
    Description:
        A ball is in the air and the goal is to keep it in the air! (not collide with ground or roof)


    Observation:
        Type: Box(2)
        Num     Observation               Min                     Max
        0       Ball Position             0                       100
        1       Ball Velocity             -Inf                    Inf
        2       Force applied to ball     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Void
        1     Push cart to the up

    Reward:
        Reward is 0.1 for no collision, -10 for collision

    Starting State:
        Ball position: random between 10 and 90
        Ball velocity: random between -5 and +5
        Ball force: random between -5 and +5

    Episode Termination:
        Ball collision with ground or roof.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, **kwargs):
        self.gravity = 9.8
        self.ball_mass = .2  # kg
        self.force_mag = 15
        self.tau = 0.1  # seconds between state updates
        self.ground_y = 0
        self.roof_y = 100
        self.ball_radius = kwargs.get("ball_radius", 2.5)

        low = np.array([self.ground_y, -np.finfo(np.float32).max, -np.finfo(np.float32).max], dtype=np.float32)
        high = np.array([self.roof_y, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        position, velocity, force = self.state
        force_mag = self.force_mag if action == 1 else 0
        #  Fixme: not physically correct
        force += force_mag + (-self.gravity * self.ball_mass)
        velocity += (force/self.ball_mass) * self.tau
        position += velocity * self.tau

        self.state = [round(position), round(velocity) // 5 * 5, round(force) // 5 * 5]

        done = bool(
            position <= self.ground_y + self.ball_radius or
            position >= self.roof_y - self.ball_radius
        )

        if not done:
            difference_to_middle = round(abs(position - (self.roof_y - self.ground_y) / 2))
            if difference_to_middle == 0:
                difference_to_middle = 1
            reward = 1 / difference_to_middle
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = -10
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = [random.randint(10, 90), random.randint(-5, 5), random.randint(-5, 5)]
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_height = self.roof_y - self.ground_y
        scale = screen_height/world_height

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self._ball = rendering.make_circle(self.ball_radius * scale, res=10, filled=True)
            self._ball.set_color(0, 0, 0)
            self.balltrans = rendering.Transform()
            self._ball.add_attr(self.balltrans)
            self.viewer.add_geom(self._ball)

        if self.state is None:
            return None

        x = self.state
        ball_position = x[0] * scale
        self.balltrans.set_translation(screen_width//2, ball_position)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
