from config import config
import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
from snntorch import functional as SF
from Neuron import SNNLayer

def set_device():
    """Set the device for PyTorch operations"""
    # Check if CUDA is available and set the device accordingly.
    # If not, default to CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class SNN(nn.Module):
    def __init__(self, input_dim=80*80, hidden_size=1000, action_size=4, time_steps=config["time_steps"]):
        super().__init__()
        self.time_steps = time_steps
        self.layer1 = SNNLayer(input_dim, hidden_size)
        self.layer2 = SNNLayer(hidden_size, action_size)

        self.device = set_device()
        self.action_size = action_size

    def reset(self):
        self.layer1.neuron.reset()
        self.layer2.neuron.reset()

    def forward(self, x):
        batch_size = x.size(0)
        output_spike_count = torch.zeros((batch_size, self.action_size), device=x.device)

        for t in range(self.time_steps):
            input_t = x[:, t, :]
            spikes_hidden = self.layer1(input_t)
            spikes_out = self.layer2(spikes_hidden)
            output_spike_count += spikes_out

        return output_spike_count
    
    def load_ann_weights(self, filepath: str="weights/greyscale_model_weights.pth", scale_1=10,scale_2=100):
        ann_state_dict = torch.load(filepath, map_location=self.device)

        # Check if ANN layer names match or map manually
        if 'fc1.weight' in ann_state_dict and 'fc2.weight' in ann_state_dict:
            with torch.no_grad():
                self.layer1.linear.weight.copy_(ann_state_dict['fc1.weight'] * scale_1)
                # self.layer1.linear.bias.copy_(ann_state_dict['fc1.bias'] * scale_1)
                self.layer2.linear.weight.copy_(ann_state_dict['fc2.weight'] * scale_2)
                # self.layer2.linear.bias.copy_(ann_state_dict['fc2.bias'] * scale_2)
        else:
            raise ValueError("ANN state dict does not contain expected layer names (fc1, fc2).")

    


class DQNNet(nn.Module):
    def __init__(self, input_dim: int= 80 * 80, action_size: int = 4):
        super(DQNNet, self).__init__()
        self.device = set_device()
        hidden_dim = 1000
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_size)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


