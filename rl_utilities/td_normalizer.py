from rl_utilities import WelfordRunningStat
import numpy as np
import torch


class TDNormalizer:
    """ Copied from: https://github.com/gauthamvasan/avg/blob/main/incremental_rl/td_error_scaler.py

    Usage: Push the statistics online _before_ a learning update. Scale TD error by sigma. 
    """
    def __init__(self):
        self.gamma_rms = WelfordRunningStat(shape=(), backend="python")
        self.return_sq_rms = WelfordRunningStat(shape=(), backend="python")
        self.reward_rms = WelfordRunningStat(shape=(), backend="python")
        self.return_rms = WelfordRunningStat(shape=(), backend="python")

    @torch.no_grad()
    def update(self, reward, gamma, episode_return):
        if episode_return is not None:
            self.return_sq_rms.increment(episode_return*episode_return, 1)
            self.return_rms.increment(episode_return, 1)
        self.reward_rms.increment(reward, 1)
        self.gamma_rms.increment(gamma, 1)

    @property
    def sigma(self):
        variance = max(self.reward_rms.var + self.gamma_rms.var * self.return_sq_rms.mean, 1e-4)
        
        # N.B: Do not scale until the first return is seen
        if variance <= 0.01 and self.return_sq_rms.count == 0: 
            return 1                                     
        return np.sqrt(variance)
  