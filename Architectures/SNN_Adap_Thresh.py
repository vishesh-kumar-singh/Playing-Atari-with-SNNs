
import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
from snntorch import functional as SF

class AdaptiveSNN(nn.Module):
    def __init__(self, input_dim=6400, hidden_dim=1000, action_size=4,config=None):
        super(AdaptiveSNN, self).__init__()

        beta = 1 - config['snn']['voltage_decay']
        theta_plus = config["adaptive_threshold"]["theta_plus"]
        theta_decay = config["adaptive_threshold"]["theta_decay"]

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, threshold=config["snn"]["threshold_voltage"], reset_mechanism="zero")
        self.theta1 = snn.threshold.MembraneAdaptive(thresh=config["snn"]["threshold_voltage"], theta_plus=theta_plus, theta_decay=theta_decay)

        self.fc2 = nn.Linear(hidden_dim, action_size)
        self.lif2 = snn.Leaky(beta=beta, threshold=config["snn"]["threshold_voltage"], reset_mechanism="zero")
        self.theta2 = snn.threshold.MembraneAdaptive(thresh=config["snn"]["threshold_voltage"], theta_plus=theta_plus, theta_decay=theta_decay)

    def forward(self, x):
        mem1, theta1 = self.lif1.init_leaky(), self.theta1.init_adaptive()
        mem2, theta2 = self.lif2.init_leaky(), self.theta2.init_adaptive()
        spk2_rec = []

        for t in range(x.shape[1]):
            cur = x[:, t, :]

            h1 = self.fc1(cur)
            thr1 = self.theta1(mem1, theta1)
            spk1, mem1 = self.lif1(h1, mem1, threshold=thr1)

            h2 = self.fc2(spk1)
            thr2 = self.theta2(mem2, theta2)
            spk2, mem2 = self.lif2(h2, mem2, threshold=thr2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=1)  # (batch, time, output)
    
    def transfer_weights(self, ann, scale_layer1=1.0, scale_layer2=1.0):
        self.fc1.weight.data = ann.fc1.weight.data.clone().to(self.device) * scale_layer1
        self.fc1.bias.data = ann.fc1.bias.data.clone().to(self.device) * scale_layer1

        self.fc2.weight.data = ann.fc2.weight.data.clone().to(self.device) * scale_layer2
        self.fc2.bias.data = ann.fc2.bias.data.clone().to(self.device) * scale_layer2

