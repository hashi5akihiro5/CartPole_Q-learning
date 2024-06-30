import numpy as np


class Agent:
    def __init__(self, env):
        self.num_dizitized = 6
        self.q_talbel = np.random.uniform(
            low=-1, high=1, size=(self.num_dizitized**4, env.action_spance.n)
        )