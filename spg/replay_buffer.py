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

    def store(self, s, a, r):
        batch_size = s.shape[0]
        for i in range(batch_size):
            #if r[i] == 0.:
            #    continue
            experience = (s[i], a[i], r[i])
            if self.count < self.buffer_size: 
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)
    
    def sort(self):
        return sorted(self.buffer, key=lambda x: x[2])
   
    @staticmethod
    def get_1_sided_range(sorted_buffer, percentile, end='front'):
        count = len(sorted_buffer)
        if percentile <= 0. or percentile > 1.:
            print("Percentile must be in (0, 1]")
            return None
        n = int(np.floor(percentile * count))
        if n <= 0 or n > count:
            return None
        if end == 'front':
            return sorted_buffer[:n]
        elif end == 'back':
            return sorted_buffer[-n:]

    @staticmethod
    def get_2_sided_range(sorted_buffer, p1, p2):
        if p1 <= 0 or p1 >= p2 or p1 > 1 or p2 <= 0 or p2 > 1:
            print("Invalid range {}-{}".format(p1, p2))
            return None
        count = len(sorted_buffer)
        n1 = int(np.floor(p1 * count))
        n2 = int(min(np.ceil(p2 * count), count))
        return sorted_buffer[n1:n2]

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = [_[0] for _ in batch]
        a_batch = [_[1] for _ in batch]
        r_batch = [_[2] for _ in batch]

        return s_batch, a_batch, r_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

if __name__ == '__main__':
    rb = ReplayBuffer(10)
    s = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
    a = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
    r = np.array([[45], [2], [7], [1], [0], [-2], [66], [75], [-0.3], [0.3]])
    rb.store(s, a, r)
    rb_sorted = rb.sort()
    print(rb_sorted)
    worst = rb.get_1_sided_range(rb_sorted, 0.2)
    best = rb.get_1_sided_range(rb_sorted, 0.2, end='back')
    middle = rb.get_2_sided_range(rb_sorted, 0.2, 0.8)

    print('worst: {}'.format(worst))
    print('best: {}'.format(best))
    print('middle: {}'.format(middle))
    
