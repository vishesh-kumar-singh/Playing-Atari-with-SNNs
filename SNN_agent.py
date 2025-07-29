from Architectures.Stochastic_SNN_Adap_Thresh import AdaptiveStochasticSNN
from Architectures.SNN import SNN
from Architectures.SNN_Adap_Thresh import AdaptiveSNN
from Architectures.ANN import DQNNet
from config import config
import numpy as np
import torch
from Environment import BreakoutEnv


class SNNAgent:
    def __init__(self, action_size, ann):
        """Initialize agent with networks and replay buffer"""
        self.action_size = action_size

        self.policy_net = AdaptiveSNN(input_dim=80*80,action_size=action_size,config=config)
        self.policy_net.transfer_weights(ann,scale_layer1=10, scale_layer2=100)

        self.epsilon = 0
        self.step_count = 0


    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.policy_net.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        return action



ann = DQNNet(input_dim=80*80, action_size=4)
ann.load_state_dict(torch.load("weights/binary_model_weights.pth"))

env=BreakoutEnv(rendering_mode='human',processing_method="Binary",is_SNN=True)
state=env.reset()
action_size = env.action_space.n
agent=SNNAgent(action_size=action_size,ann=ann)
agent.epsilon = 0.1


for i in range(5):
    state = env.reset()
    episode_reward = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)

        state = next_state
        episode_reward += reward
        if done:
            break
    
    print(f"Episode {i+1} Reward: {episode_reward:.2f}")
    
