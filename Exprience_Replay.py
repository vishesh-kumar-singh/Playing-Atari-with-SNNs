import random
import numpy as np
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim, device="cpu"):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.state_dim = state_dim

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer"""
        state = np.array(state, copy=False)
        next_state = np.array(next_state, copy=False)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        return (
            torch.tensor(state, dtype=torch.float32, device=self.device).view(batch_size, -1),
            torch.tensor(action, dtype=torch.int64, device=self.device).view(batch_size, 1),
            torch.tensor(reward, dtype=torch.float32, device=self.device).view(batch_size, 1),
            torch.tensor(next_state, dtype=torch.float32, device=self.device).view(batch_size, -1),
            torch.tensor(done, dtype=torch.float32, device=self.device).view(batch_size, 1)
        )

    def __len__(self):
        return len(self.buffer)