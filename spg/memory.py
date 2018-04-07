# Converted to PyTorch from
# https://github.com/openai/baselines/blob/master/baselines/ddpg/memory.py
import numpy as np
import torch
import pdb

class RingBuffer:
    def __init__(self, maxlen, shape, use_cuda, dtype='torch.FloatTensor'):
        self.maxlen = maxlen
        self.start = 0 # the idx of the 0th element in the buffer
        self.length = 0
        self.data = torch.zeros(maxlen, *shape).type(dtype)
        self.use_cuda = use_cuda
        if use_cuda:
            self.data = self.data.cuda()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        torch_idxs = torch.from_numpy((self.start + idxs) % self.maxlen).long()
        if self.use_cuda:
            torch_idxs = torch_idxs.cuda()
        return self.data[torch_idxs]


    def append(self, v):
        batch_size = v.shape[0]
        for i in range(batch_size):
            if self.length < self.maxlen:
                # We have space, simply increase the length.
                self.length += 1
            elif self.length == self.maxlen:
                # No space, "remove" the first item.
                self.start = (self.start + 1) % self.maxlen
            else:
                # This should never happen.
                raise RuntimeError()
            self.data[(self.start + self.length - 1) % self.maxlen] = v[i]


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory:
    def __init__(self, limit, action_shape, observation_shape, use_cuda=True):
        self.limit = limit

        self.observations = RingBuffer(limit, observation_shape, use_cuda)
        self.discrete_actions = RingBuffer(limit, action_shape, use_cuda, dtype='torch.ByteTensor')
        self.dense_actions = RingBuffer(limit, action_shape, use_cuda)
        self.rewards = RingBuffer(limit, [1], use_cuda)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs_batch = self.observations.get_batch(batch_idxs)
        discrete_actions_batch = self.discrete_actions.get_batch(batch_idxs)
        dense_actions_batch = self.dense_actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)

        return obs_batch, discrete_actions_batch, dense_actions_batch, reward_batch

    def append(self, obs, discrete_action, dense_action, reward):
        
        self.observations.append(obs)
        self.discrete_actions.append(discrete_action)
        self.dense_actions.append(dense_action)
        self.rewards.append(reward)

    @property
    def nb_entries(self):
        return len(self.observations)

if __name__ == '__main__':

    rb = Memory(100000, action_shape=[10, 10], observation_shape=[10,2], use_cuda=False)
    states = torch.zeros(128, 10, 2)
    discrete_actions = torch.ones(128, 10, 10)
    dense_actions = torch.ones(128, 10, 10)
    rewards = torch.zeros(128)
    
    rb.append(states, discrete_actions, dense_actions, rewards)
    s_batch, psi_batch, a_batch, r_batch = rb.sample(28)
