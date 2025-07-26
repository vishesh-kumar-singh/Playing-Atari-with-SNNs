import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Architectures.ANN import DQNNet
from config import config
from Experience_Replay import ReplayBuffer

def MSELoss(policy_net, target_net, states, actions, 
                             rewards, next_states, dones, gamma):
    
    q_values = policy_net(states).gather(1, actions)
  
    next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]

    target_q_values = rewards + (1 - dones) * gamma * next_q_values
    target_q_values = target_q_values.detach()

   
    td_errors = target_q_values - q_values

    loss = (td_errors ** 2).mean()

    return loss, td_errors


class ANNAgent:
    def __init__(self, state_shape, action_size, config=config):
        """Initialize agent with networks and replay buffer"""
        self.state_shape = state_shape
        self.action_size = action_size

        self.policy_net = DQNNet(state_shape, action_size)
        self.policy_net.to(self.policy_net.device)
        self.target_net = DQNNet(state_shape, action_size)
        self.target_net.to(self.target_net.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.replay_buffer= ReplayBuffer(capacity=config["dqn"]["replay_memory_size"], state_dim=state_shape)
        self.epsilon = config['dqn']['initial_epsilon']
        self.epsilon_min = config['dqn']['final_epsilon']
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (config['dqn']['final_exploration_step'])
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=config['dqn']['learning_rate'],
            momentum=config['dqn']['gradient_momentum'],
        )
        self.step_count = 0
        self.batch_size = config['dqn']['mini_batch_size']
        self.gamma = config['dqn']['discount_factor']

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.policy_net.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        return action
    
    def train_dqn_step(self):
        device = self.policy_net.device

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        actions = actions if torch.is_tensor(actions) else torch.tensor(actions, dtype=torch.long)
        actions = actions.view(-1, 1).to(device)

        rewards = rewards if torch.is_tensor(rewards) else torch.tensor(rewards, dtype=torch.float32)
        rewards = rewards.unsqueeze(1).to(device)

        dones = dones if torch.is_tensor(dones) else torch.tensor(dones, dtype=torch.float32)
        dones = dones.unsqueeze(1).to(device)



        # Compute and apply loss
        loss, td_error = MSELoss(self.policy_net, self.target_net, states, actions, rewards, next_states, dones, self.gamma)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        self.update_epsilon()

        if self.step_count % config['dqn']['target_network_update_frequency'] == 0:
            self.update_target_network()

        return loss.item()
    


    def update_epsilon(self):
        """Update epsilon for epsilon-greedy action selection"""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, filepath: str="model_weights.pth"):
        """Save model weights to file"""
        # Save the policy network state dictionary to the specified filepath.
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath: str="model_weights.pth"):
        """Load model weights from file"""
        # Load the policy network state dictionary from the specified filepath.
        self.policy_net.load_state_dict(torch.load(filepath))
        self.update_target_network()