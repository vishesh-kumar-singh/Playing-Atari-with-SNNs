from config import config
import numpy as np
import cv2
from collections import deque

def poisson_spike_encoding(images, time_steps=100):
    return (np.random.rand(images.shape[0], time_steps, 80*80) < images[:, None, :]).astype(np.uint8)

class FrameStack():
    def __init__(self):
        """Initialize deque with maxlen capacity"""
        self.maxlen=config['dqn']['agent_history_length']
        self.deque_of_frames =deque(maxlen=self.maxlen)
        # Store the maximum length for reference in other methods.
        self.frame_shape=None
        self.processed = deque(maxlen=self.maxlen-1)
    def push(self, frame: np.ndarray) -> None:
        """Add preprocessed frame to deque"""
        if self.frame_shape is None:
            self.frame_shape = frame.shape
        self.deque_of_frames.append(frame)
        # Add the new frame to the collection.
        # The data structure should automatically handle overflow.
    
    def crop(self, img):
        h, w, _ = img.shape
        center_crop = img[10*h//100: h, 0: w]  # crop center 50% area
        return center_crop

    def get_binary(self):
        frames = list(self.deque_of_frames)
        stacked = np.zeros((80, 80), dtype=np.float32)
        frame0 = cv2.resize(cv2.cvtColor(self.crop(frames[0]), cv2.COLOR_BGR2GRAY), (80, 80))
        stacked += (frame0 > 0).astype(np.float32)

        for i in range(1,len(self.deque_of_frames)):
            frame1= self.crop(frames[i])
            frame2= self.crop(frames[i-1])
            curr = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (80, 80))
            prev = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (80, 80))
            diff = curr.astype(np.float32) - prev.astype(np.float32)
            diff[diff < 0] = 0
            binary = (diff > 0).astype(np.float32)
            self.processed.append(binary)
        processed_list=list(self.processed)
        for i, frame in enumerate(processed_list):
            stacked += frame

        stacked[stacked > 0] = 1.0
        return stacked

    def get_greyscale(self):
        frames = list(self.deque_of_frames)
        weights = [1.0, 0.75, 0.5, 0.25]
        stacked = np.zeros((80, 80), dtype=np.float32)
        frame0 = cv2.resize(cv2.cvtColor(self.crop(frames[0]), cv2.COLOR_BGR2GRAY), (80, 80))
        if len(self.deque_of_frames)>1:
            for i in range(1,len(self.deque_of_frames)):
                frame1= self.crop(frames[i])
                frame2= self.crop(frames[i-1])
                curr = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (80, 80))
                prev = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (80, 80))
                diff = curr.astype(np.float32) - prev.astype(np.float32)
                diff[diff < 0] = 0
                binary = (diff > 0).astype(np.float32)
                self.processed.append(binary)
            processed_list=list(self.processed)
            processed_list.reverse()
            for i, frame in enumerate(processed_list):
                stacked += frame * weights[i]
            stacked+= (frame0 > 0).astype(np.float32) * weights[-1]
        else:
            stacked = (frame0 > 0).astype(np.float32)
        return stacked
    
    def reset(self) -> None:
        """Clear the deque"""
        self.deque_of_frames.clear()
        self.frame_shape = None