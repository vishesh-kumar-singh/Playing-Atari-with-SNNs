import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

def poisson_spike_encoding(images, time_steps=100):
    return (np.random.rand(images.shape[0], config["snn"]["time_steps"], 80*80) < images[:, None, :]).astype(np.uint8)

def Binary(frame1, frame2, frame3, frame4):
    frames = [frame1, frame2, frame3, frame4]
    processed = []
    for i in range(1, 4):
        curr = resize(rgb2gray(frames[i]), (80, 80), anti_aliasing=True)
        prev = resize(rgb2gray(frames[i-1]), (80, 80), anti_aliasing=True)
        diff = curr - prev
        diff[diff < 0] = 0
        binary = (diff > 0.1).astype(np.float32)
        processed.append(binary)
    stacked = np.sum(processed, axis=0)
    return stacked

def Greyscale(frame1, frame2, frame3, frame4):
    frames = [frame1, frame2, frame3, frame4]
    weights = [1.0, 0.75, 0.5, 0.25]
    stacked = np.zeros((80, 80), dtype=np.float32)
    for i, w in enumerate(weights):
        gray = resize(rgb2gray(frames[3-i]), (80, 80), anti_aliasing=True)
        stacked += w * gray
    return stacked
