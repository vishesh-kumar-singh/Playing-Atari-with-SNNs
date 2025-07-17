import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQNNet(nn.Module):
    def __init__(self, state_shape, action_size):
        super(DQNNet, self).__init__()
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

class Agent:
    def __init__(self, state_shape, action_size, config):
        """Initialize agent with networks and replay buffer"""
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNet(state_shape, action_size).to(self.device)
         
        self.epsilon = config['dqn']['init_epsilon']
        self.epsilon_min = config['dqn']['final_epsilon']
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (config['dqn']['final_exploration_step'])
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=config['dqn']['lr'],
            momentum=config['dqn']['momentum'],
            alpha=config['dqn']['alpha'],
            eps=config['dqn']['eps'],
        )
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        return action
