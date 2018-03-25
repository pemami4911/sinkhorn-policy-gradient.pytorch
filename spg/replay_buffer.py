""" 
Data structure for implementing experience replay

Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def store(self, s, psi, a, r):
        batch_size = s.shape[0]
        for i in range(batch_size):
            #if r[i] == 0.:
            #    continue
            experience = (s[i], psi[i], a[i], r[i])
            if self.count < self.buffer_size: 
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = [_[0] for _ in batch]
        psi_batch = [_[1] for _ in batch]
        a_batch = [_[2] for _ in batch]
        r_batch = [_[3] for _ in batch]

        return s_batch, psi_batch, a_batch, r_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
