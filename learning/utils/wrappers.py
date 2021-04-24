import gym
from gym import spaces
import numpy as np


class DiscreteWrapper(gym.ActionWrapper):
    """
    Discrete actions (left, right, forward) instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Left
        if action == 0:
            vels = [0.35, +1.0]
        # Right
        elif action == 1:
            vels = [0.35, -1.0]
        # Forward
        elif action == 2:
            vels = [0.44, 0.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class DiscreteWrapper_9(gym.ActionWrapper):
    """
    Discrete actions (left, right, forward) instead of continuous control.
    9 Actions (4 angles left, 4 angles right, straight forward)
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(9)

    def action(self, action):
        # Left
        if action == 0:
            vels = [0.35, +1.0]
        elif action == 1:
            vels = [0.3725, +.75]
        elif action == 2:
            vels = [0.395, +.5]
        elif action == 3:
            vels = [0.4175, +.25]
        # Right
        elif action == 4:
            vels = [0.35, -1.0]
        elif action == 5:
            vels = [0.3725, -.75]
        elif action == 6:
            vels = [0.395, -.5]
        elif action == 7:
            vels = [0.4175, -.25]
        # Forward
        elif action == 8:
            vels = [0.44, 0.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class DiscreteWrapper_9_map5(gym.ActionWrapper):
    """
    Discrete actions (left, right, forward) instead of continuous control.
    9 Actions (4 angles left, 4 angles right, straight forward)
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(9)

    def action(self, action):
        # Left
        if action == 0:
            vels = [0.15, +1.0]
        elif action == 1:
            vels = [0.225, +.75]
        elif action == 2:
            vels = [0.30, +.5]
        elif action == 3:
            vels = [0.375, +.25]
        # Right
        elif action == 4:
            vels = [0.15, -1.0]
        elif action == 5:
            vels = [0.225, -.75]
        elif action == 6:
            vels = [0.30, -.5]
        elif action == 7:
            vels = [0.375, -.25]
        # Forward
        elif action == 8:
            vels = [0.44, 0.0]
        # Turn on the spot to search for ideal starting orientation
        elif action == 9:
            vels = [0.0, +1.0]
        elif action == 10:
            vels = [0.0, -1.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class DiscreteWrapper_9_testing(gym.ActionWrapper):
    """
    Discrete actions (left, right, forward) instead of continuous control.
    9 Actions (4 angles left, 4 angles right, straight forward)
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(9)

    def action(self, action):
        # Left
        if action == 0:
            vels = [0.35, +1.0]
        elif action == 1:
            vels = [0.3725, +.75]
        elif action == 2:
            vels = [0.395, +.5]
        elif action == 3:
            vels = [0.4175, +.25]
        # Right
        elif action == 4:
            vels = [0.35, -1.0]
        elif action == 5:
            vels = [0.3725, -.75]
        elif action == 6:
            vels = [0.395, -.5]
        elif action == 7:
            vels = [0.4175, -.25]
        # Forward
        elif action == 8:
            vels = [0.44, 0.0]
        # Turn on the spot to search for ideal starting orientation
        elif action == 9:
            vels = [0.0, +1.0]
        elif action == 10:
            vels = [0.0, -1.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class DiscreteWrapper_9_script(gym.ActionWrapper):
    """
    Discrete actions (left, right, forward) instead of continuous control.
    9 Actions (4 angles left, 4 angles right, straight forward)
    """

    def __init__(self, env, actions):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(9)
        self.actions = actions

    def action(self, action):
        return np.array(self.actions[action])

class DiscreteWrapper_9_custom(gym.ActionWrapper):
    """
    Discrete actions (left, right, forward) instead of continuous control.
    9 Actions (4 angles left, 4 angles right, straight forward)
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(9)

    def action(self, action):
        # Left
        if action == 0:
            vels = [0.576, +1.0]
        elif action == 1:
            vels = [0.612, +.75]
        elif action == 2:
            vels = [0.648, +.5]
        elif action == 3:
            vels = [0.684, +.25]
        # Right
        elif action == 4:
            vels = [0.576, -1.0]
        elif action == 5:
            vels = [0.612, -.75]
        elif action == 6:
            vels = [0.648, -.5]
        elif action == 7:
            vels = [0.684, -.25]
        # Forward
        elif action == 8:
            vels = [0.72, 0.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from scipy.misc import imresize
        return imresize(observation, self.shape)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


class DtRewardWrapper2(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper2, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -100

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_


class ActionClampWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionClampWrapper, self).__init__(env)

    def action(self, action):
        action_ = np.clip(action, -1, 1)
        return action_


class PreventBackwardsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(PreventBackwardsWrapper, self).__init__(env)

    def action(self, action):
        action_ = action
        action_[0] = np.clip(action[0], 0, 1)
        return action_
