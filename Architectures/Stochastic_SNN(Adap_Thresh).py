import torch
import torch.nn as nn
from config import config


class StochasticAdaptiveLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 1.0 - config["snn"]["voltage_decay"]
        self.threshold_base = config["snn"]["threshold_voltage"]
        self.resting_voltage = config["snn"]["resting_voltage"]
        self.theta_plus = config["adaptive_threshold"]["theta_plus"]
        self.theta_decay = config["adaptive_threshold"]["theta_decay"]

    def forward(self, input_, mem, theta):
        mem = self.beta * (mem - self.resting_voltage) + self.resting_voltage + input_

        threshold = self.threshold_base + theta
        prob_spk = torch.sigmoid(mem - threshold)
        spk = torch.bernoulli(prob_spk)
        mem = mem * (1 - spk)

        theta = self.theta_decay * theta + self.theta_plus * spk
        return spk, mem, theta


class AdaptiveStochasticSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(6400, 1000)
        self.lif1 = StochasticAdaptiveLIF()

        self.fc2 = nn.Linear(1000, 4)
        self.lif2 = StochasticAdaptiveLIF()

        self.to(self.device)

    def forward(self, x, time_steps=100):
        x = x.to(self.device)
        batch_size = x.shape[0]

        mem1 = torch.zeros(batch_size, 1000, device=self.device)
        mem2 = torch.zeros(batch_size, 4, device=self.device)
        theta1 = torch.zeros_like(mem1)
        theta2 = torch.zeros_like(mem2)

        out_spk = torch.zeros(batch_size, 4, device=self.device)

        for _ in range(time_steps):
            cur1 = self.fc1(x)
            spk1, mem1, theta1 = self.lif1(cur1, mem1, theta1)

            cur2 = self.fc2(spk1)
            spk2, mem2, theta2 = self.lif2(cur2, mem2, theta2)

            out_spk += spk2

        return out_spk / config["snn"]["time_steps"]

    def transfer_weights(self, ann, scale_layer1=1, scale_layer2=1):
        self.fc1.weight.data = ann.fc1.weight.data.clone().to(self.device) * scale_layer1
        self.fc1.bias.data = ann.fc1.bias.data.clone().to(self.device) * scale_layer1

        self.fc2.weight.data = ann.fc2.weight.data.clone().to(self.device) * scale_layer2
        self.fc2.bias.data = ann.fc2.bias.data.clone().to(self.device) * scale_layer2

