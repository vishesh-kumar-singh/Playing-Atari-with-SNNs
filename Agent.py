from Architectures import SNN, DQNNet
from config import config
import numpy as np
import torch
import torch.optim as optim
import random
from collections import deque

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


def Loss(policy_net, target_net, states, actions, rewards, next_states, dones, gamma, is_snn=False):

    q_values = policy_net(states)  # [batch, num_actions]



    q_values = q_values.gather(1, actions)

    with torch.no_grad():
        if is_snn:
            next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        else:
            next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + gamma * next_q_values * (1 - dones)

    td_errors = target_q - q_values
    loss = (td_errors ** 2).mean()
    return loss, td_errors


def poisson_spike_encoding(images, time_steps=config['time_steps']):
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    images = images.to(torch.float32)  # Ensure float for comparison
    rand_vals = torch.rand((images.shape[0], time_steps, 80*80), device=images.device)
    return (rand_vals < images.unsqueeze(1)).to(torch.uint8)



class SNNAgent:
    def __init__(self, state_shape,action_size):
        """Initialize agent with networks and replay buffer"""
        self.action_size = action_size

        self.policy_net = SNN(input_dim=80*80,action_size=action_size)
        self.policy_net.to(self.policy_net.device)

        self.target_net = SNN(input_dim=80*80,action_size=action_size)
        self.target_net.to(self.policy_net.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.replay_buffer= ReplayBuffer(capacity=config["replay_memory_size"], state_dim=state_shape)
        self.epsilon = config['initial_epsilon']
        self.epsilon_min = config['final_epsilon']
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (config['final_exploration_step'])
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=config['learning_rate'],
            momentum=config['gradient_momentum'],
            alpha=config['squared_gradient_momentum'],
            eps=config['min_squared_gradient']
        )
        self.step_count = 0
        self.batch_size = config['mini_batch_size']
        self.gamma = config['discount_factor']
        self.memory_init_size=config['replay_memory_init_size']


    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        encoded_state= poisson_spike_encoding(state[None,:], time_steps=config['time_steps'])[0]
        state_tensor = encoded_state.detach().clone().float().to(self.policy_net.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            q_values_mean = q_values.mean(dim=1)  # shape: [1, 4]
        action = int(q_values_mean.argmax().item())  # shape [1] → scalar

        return action
    

    
    def train_step(self):
        device = self.policy_net.device

        self.step_count += 1
        self.update_epsilon()

        if len(self.replay_buffer) < self.memory_init_size:
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = poisson_spike_encoding(states, time_steps=config['time_steps']).float().to(device)

        next_states = poisson_spike_encoding(next_states, time_steps=config['time_steps']).float().to(device)

        actions = actions if torch.is_tensor(actions) else torch.tensor(actions, dtype=torch.long)
        actions = actions.view(-1, 1).to(device)

        rewards = rewards if torch.is_tensor(rewards) else torch.tensor(rewards, dtype=torch.float32)
        rewards = rewards.unsqueeze(1).to(device)

        dones = dones if torch.is_tensor(dones) else torch.tensor(dones, dtype=torch.float32)
        dones = dones.unsqueeze(1).to(device)



        # Compute and apply loss
        loss, td_error = Loss(self.policy_net, self.target_net, states, actions, rewards, next_states, dones, self.gamma, is_snn=True)
        self.policy_net.reset()
        self.target_net.reset()
        self.optimizer.zero_grad()
        loss.backward()
        if self.step_count % config['update_frequency'] != 0:
            self.optimizer.step()


        if self.step_count % config['target_network_update_frequency'] == 0:
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

        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath: str="model_weights.pth"):
        """Load model weights from file"""

        self.policy_net.load_state_dict(torch.load(filepath))
        self.update_target_network()

    def load_ann_weights(self, filepath: str="weights/greyscale_model_weights.pth", scale_1=20,scale_2=100):
        self.policy_net.load_ann_weights(filepath, scale_1, scale_2)
        self.update_target_network()

    
    




class ANNAgent:
    def __init__(self, state_shape,action_size, config=config):
        """Initialize agent with networks and replay buffer"""
        self.action_size = action_size

        self.policy_net = DQNNet(input_dim=80*80,action_size=action_size)
        self.policy_net.to(self.policy_net.device)
        self.target_net = DQNNet(input_dim=80*80, action_size=action_size)
        self.target_net.to(self.target_net.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.replay_buffer= ReplayBuffer(capacity=config["replay_memory_size"], state_dim=state_shape)
        self.epsilon = config['initial_epsilon']
        self.epsilon_min = config['final_epsilon']
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (config['final_exploration_step'])
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=config['learning_rate'],
            momentum=config['gradient_momentum'],
            alpha=config['squared_gradient_momentum'],
            eps=config['min_squared_gradient']
        )
        self.step_count = 0
        self.batch_size = config['mini_batch_size']
        self.gamma = config['discount_factor']
        self.memory_init_size=config['replay_memory_init_size']

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.policy_net.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        return action
    
    def train_step(self):
        device = self.policy_net.device
        self.update_epsilon()
        self.step_count += 1

        if len(self.replay_buffer) < self.memory_init_size:

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
        loss, td_error = Loss(self.policy_net, self.target_net, states, actions, rewards, next_states, dones, self.gamma)
        
        self.optimizer.zero_grad()
        loss.backward()

        if self.step_count % config['update_frequency'] != 0:
            self.optimizer.step()


        if self.step_count % config['target_network_update_frequency'] == 0:
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

        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath: str="model_weights.pth"):
        """Load model weights from file"""
        print(f"Loading model weights from {filepath} into {self.policy_net.device}")
        self.policy_net.load_state_dict(torch.load(filepath,map_location=self.policy_net.device))
        self.update_target_network()