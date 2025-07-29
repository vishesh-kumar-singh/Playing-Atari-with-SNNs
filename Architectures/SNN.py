from config import config
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNN(nn.Module):
    def __init__(self, input_dim=6400, hidden_dim=1000, action_size=4):
        super(SNN, self).__init__()

        beta = 1 - config['snn']['voltage_decay']

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, threshold=config["snn"]["threshold_voltage"], reset_mechanism="zero", surrogate_fn=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_dim, action_size)
        self.lif2 = snn.Leaky(beta=beta, threshold=config["snn"]["threshold_voltage"], reset_mechanism="zero", surrogate_fn=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []

        for t in range(x.shape[1]):  # loop over time
            cur = x[:, t, :]  # shape: (batch, input_dim)

            h1 = self.fc1(cur)
            spk1, mem1 = self.lif1(h1, mem1)

            h2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(h2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=1)  # shape: (batch, time, output_dim)

    def transfer_weights(self, ann, scale_layer1=1.0, scale_layer2=1.0):
        self.fc1.weight.data = ann.fc1.weight.data.clone().to(self.device) * scale_layer1
        self.fc1.bias.data = ann.fc1.bias.data.clone().to(self.device) * scale_layer1

        self.fc2.weight.data = ann.fc2.weight.data.clone().to(self.device) * scale_layer2
        self.fc2.bias.data = ann.fc2.bias.data.clone().to(self.device) * scale_layer2
