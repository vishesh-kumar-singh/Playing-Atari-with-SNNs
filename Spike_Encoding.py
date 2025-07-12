import numpy as np
from config import config

def poisson_spike_encoding(images, time_steps=100):
    return (np.random.rand(images.shape[0], config["snn"]["time_steps"], 80*80) < images[:, None, :]).astype(np.uint8)