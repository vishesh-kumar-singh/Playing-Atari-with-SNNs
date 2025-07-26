import torch.nn as nn
import torch
from typing import Tuple
import numpy as np

def set_device():
    """Set the device for PyTorch operations"""
    # Check if CUDA is available and set the device accordingly.
    # If not, default to CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class DQNNet(nn.Module):
    def __init__(self, state_shape, action_size):
        super(DQNNet, self).__init__()
        self.device = set_device()
        input_dim = np.prod(state_shape) 
        hidden_dim = 1000
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_size)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x