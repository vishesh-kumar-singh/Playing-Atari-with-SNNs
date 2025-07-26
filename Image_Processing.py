from config import config
import numpy as np
import cv2

def poisson_spike_encoding(images, time_steps=100):
    return (np.random.rand(images.shape[0], time_steps, 80*80) < images[:, None, :]).astype(np.uint8)

def Binary(frame1, frame2, frame3, frame4):
    frames = [frame1, frame2, frame3, frame4]
    processed = []
    for i in range(1,4):
        curr = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), (80, 80))
        prev = cv2.resize(cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY), (80, 80))
        diff = curr.astype(np.float32) - prev.astype(np.float32)
        diff[diff < 0] = 0
        binary = (diff > 25).astype(np.float32)  # 0.1 (on [0,1]) ~ 25 (on [0,255])
        processed.append(binary)
    stacked = np.sum(processed, axis=0)
    return stacked

def Greyscale(frame1, frame2, frame3, frame4):
    frames = [frame1, frame2, frame3, frame4]
    weights = [1.0, 0.75, 0.5, 0.25]
    stacked = np.zeros((80, 80), dtype=np.float32)
    for i, w in enumerate(weights):
        gray = cv2.resize(cv2.cvtColor(frames[3-i], cv2.COLOR_BGR2GRAY), (80, 80)).astype(np.float32) / 255.0
        stacked += w * gray
    return stacked