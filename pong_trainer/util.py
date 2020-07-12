
import numpy as np
from collections import deque
import json
import os

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_tuple):
        self.buffer.append(state_tuple)

    def sample(self, batch_size):
        indices = np.random.randint(len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [
            np.array([self.buffer[index][field]
                      for index in indices])
            for field in range(5)
        ]
        return state, np.int32(action), np.float32(reward), next_state, np.float32(done)
