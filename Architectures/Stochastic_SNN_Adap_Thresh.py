import torch
import torch.nn as nn
from config import config


class StochasticAdaptiveLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 1.0 - config["snn"]["voltage_decay"]  # decay of membrane potential
        self.threshold_base = config["snn"]["threshold_voltage"]  # base threshold
        self.resting_voltage = config["snn"]["resting_voltage"]  # resting voltage
        self.theta_plus = config["adaptive_threshold"]["theta_plus"]  # adaptation increment
        self.theta_decay = config["adaptive_threshold"]["theta_decay"]  # adaptation decay

    def forward(self, input_, mem, theta):
        # Update membrane potential with leak
        mem = self.beta * (mem - self.resting_voltage) + self.resting_voltage + input_

        # Compute threshold with adaptive component
        threshold = self.threshold_base + theta

        # Probabilistic spike generation
        prob_spk = torch.sigmoid(mem - threshold)
        spk = torch.bernoulli(prob_spk)

        # Reset membrane potential where spikes occurred
        mem = mem * (1 - spk)

        # Update adaptive threshold
        theta = self.theta_decay * theta + self.theta_plus * spk

        return spk, mem, theta


class AdaptiveStochasticSNN(nn.Module):
    def __init__(self, input_dim=6400, action_size=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 1000)
        self.lif1 = StochasticAdaptiveLIF()

        self.fc2 = nn.Linear(1000, action_size)
        self.lif2 = StochasticAdaptiveLIF()

        self.time_steps = config["snn"]["time_steps"]  # Number of time steps

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        batch_size, time_steps, input_dim = x.shape

        # Initialize neuron state variables
        mem1 = torch.zeros(batch_size, 1000, device=self.device)
        mem2 = torch.zeros(batch_size, 4, device=self.device)
        theta1 = torch.zeros_like(mem1)
        theta2 = torch.zeros_like(mem2)

        out_spk = torch.zeros(batch_size, 4, device=self.device)

        for t in range(time_steps):
            xt = x[:, t, :]  # shape: [batch, input_dim]

            cur1 = self.fc1(xt)
            spk1, mem1, theta1 = self.lif1(cur1, mem1, theta1)

            cur2 = self.fc2(spk1)
            spk2, mem2, theta2 = self.lif2(cur2, mem2, theta2)

            out_spk += spk2

        return out_spk / time_steps  # normalized spike count as Q-values

    def transfer_weights(self, ann, scale_layer1=1.0, scale_layer2=1.0):
        self.fc1.weight.data = ann.fc1.weight.data.clone().to(self.device) * scale_layer1
        self.fc1.bias.data = ann.fc1.bias.data.clone().to(self.device) * scale_layer1

        self.fc2.weight.data = ann.fc2.weight.data.clone().to(self.device) * scale_layer2
        self.fc2.bias.data = ann.fc2.bias.data.clone().to(self.device) * scale_layer2


