import torch
import torch.nn as nn
from config import config


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input - threshold)
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        beta = 10.0
        grad = beta * torch.exp(-beta * x.abs()) / ((1 + torch.exp(-beta * x.abs())) ** 2)
        return grad_output * grad, None
    
def spike_fn(input, threshold=1):
    return SurrogateSpike.apply(input, threshold)

class LIFNeuron(nn.Module):
    def __init__(self, size, base_threshold=-52, decay=0.01, theta_plus=0.05, theta_decay=1e-7, resting_voltage=-65):
        super().__init__()
        self.V = None  # Membrane potential
        self.theta = None  # Adaptive threshold
        self.resting_voltage = resting_voltage

        self.base_threshold = base_threshold  # Initial baseline threshold
        self.theta_plus = theta_plus
        self.theta_decay = theta_decay

        self.decay = decay
        self.size = size

    def reset(self):
        self.V = None
        self.theta = None

    def forward(self, input):
        if self.V is None or self.V.shape != input.shape:
            self.V = torch.full_like(input, self.resting_voltage)
        if self.theta is None or self.theta.shape != input.shape:
            self.theta = torch.zeros_like(input)

        # Leaky integration
        self.V = self.V + (-self.V + self.resting_voltage) * self.decay + input

        # Dynamic threshold = base + adaptation
        dynamic_threshold = self.base_threshold + self.theta

        # Generate spike (using surrogate spike_fn)
        S = spike_fn(self.V, dynamic_threshold)

        # Reset membrane after spike
        self.V = self.V * (1 - S)

        # Update adaptive threshold: increase on spike, decay otherwise
        self.theta = (1 - self.theta_decay) * self.theta + self.theta_plus * S

        return S

    

class SNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, threshold=config['threshold_voltage'], decay=config["voltage_decay"],theta_plus=config["theta_plus"], theta_decay=config["theta_decay"], resting_voltage=config["resting_voltage"]):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.neuron = LIFNeuron(size=out_dim, base_threshold=threshold, decay=decay, theta_plus=theta_plus, theta_decay=theta_decay, resting_voltage=resting_voltage)

    def forward(self, input_spikes):
        I = self.linear(input_spikes)
        return self.neuron(I)