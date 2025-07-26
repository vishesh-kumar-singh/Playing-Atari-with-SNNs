import torch
import torch.nn as nn
import numpy as np
from config import config

class SNNAdaptiveThreshold(nn.Module):

    def __init__(self, ann_weights_path):
        super(SNNAdaptiveThreshold, self).__init__()
        # Load ANN weights
        ann_state = torch.load(ann_weights_path)
        
        # Build layers matching ANN
        self.layers = nn.ModuleList()
        self.weights = []
        prev_dim = 80 * 80  # Assuming input size is 80x80 flattened
        layer_idx = 0
        while f'layers.{layer_idx}.weight' in ann_state:
            W = ann_state[f'layers.{layer_idx}.weight'].cpu().numpy()
            b = ann_state.get(f'layers.{layer_idx}.bias', torch.zeros(W.shape[0])).cpu().numpy()
            self.weights.append((W, b))
            # register a dummy PyTorch layer for compatibility
            self.layers.append(nn.Linear(prev_dim, W.shape[0], bias=True))
            prev_dim = W.shape[0]
            layer_idx += 1

        # SNN hyperparameters
        self.time_steps = config['snn']['time_steps']
        self.decay = config['snn']['voltage_decay']
        self.rest_voltage = config['snn']['resting_voltage']
        self.init_threshold = config['snn']['threshold_voltage']
        self.theta_decay = config['adaptive_threshold']['theta_decay']
        self.theta_increment = config['adaptive_threshold']['theta_plus']

    def forward(self, input_spikes):
        """
        input_spikes: numpy array [time_steps, input_size], values 0/1
        returns: spike counts per output neuron
        """
        # Initialize state
        num_layers = len(self.weights)
        # membrane potentials and thresholds per layer
        for l in range(num_layers):
            Vs = np.ones((self.weights[l][0].shape[0],), dtype=np.float32) * self.rest_voltage 

        for l in range(num_layers):
            thetas = np.ones_like(Vs[l]) * self.init_threshold 

        spike_counts = np.zeros_like(Vs[-1])

        # Simulation loop
        for t in range(self.time_steps):
            x_t = input_spikes[t]  
            for l, (W, b) in enumerate(self.weights):
                I = W.dot(x_t) + b
                Vs[l] = Vs[l] * (1 - self.decay) + I

                S = (Vs[l] >= thetas[l]).astype(np.float32)
                Vs[l][S == 1] = self.rest_voltage

                # Update threshold
                thetas[l] = thetas[l] * (1 - self.theta_decay) + self.theta_increment * S

                x_t = S
                if l == num_layers - 1:
                    spike_counts += S

        return spike_counts