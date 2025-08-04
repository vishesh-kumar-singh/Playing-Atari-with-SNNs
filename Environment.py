import numpy as np
from typing import Tuple
import gym as gym
from gym import spaces
import ale_py
import os
from config import config
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import deque


class FrameStack():
    def __init__(self):
        """Initialize deque with maxlen capacity"""
        self.maxlen=config['agent_history_length']
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

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

class BreakoutEnv:
    def __init__(self, env_name: str = "BreakoutNoFrameskip-v4", frame_skip: int = config['frame_skip'],rendering_mode: str = "rgb_array", processing_method: str ="Binary", save_sample_frames: bool = False):
        """Initialize environment and frame stack"""
        # Create the Gym environment and frame stack manager.
        self.env = gym.make(env_name,render_mode=rendering_mode,frameskip=frame_skip)

        # Define the action space mapping for Breakout game.
        self.action_space = spaces.Discrete(4)     # 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT
        self.observation_space = self.env.observation_space
        self.state=None
        # Store frame skipping parameter for temporal efficiency.
        self.frame_skip = frame_skip
        self.processing_method = processing_method
        self.frame_stack = FrameStack()
        self.save_sample_frames = save_sample_frames
        if self.save_sample_frames:
            self.reset()
            self.step(1)
            self.step(2)
            self.save_sample_binary_frame()
            self.save_sample_greyscale_frame()
            
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial stacked state"""
        # Reset both the environment and frame stack.
        initial_obs,_=self.env.reset()
        self.frame_stack.reset()
        self.frame_stack.push(initial_obs)     

        # Process the initial observation and create the first state stack.
        if self.processing_method == "Binary":
            self.state = self.frame_stack.get_binary()
        elif self.processing_method == "Greyscale":
            self.state = self.frame_stack.get_greyscale()
        else:
            raise ValueError(f"Unknown processing method: {self.processing_method}")

        # Return the properly formatted state for the agent.
        return self.state.flatten()[None, :]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action with frame skipping"""
        # Convert agent action to environment action.
        agent_action = action
        done = False
        total_reward = 0.0

        for _ in range(config['action_repeat']):
            # Step the environment with the agent's action.
            obs, reward, done, truncated, info = self.env.step(agent_action)
            total_reward+=reward
            if done or truncated:
                break

            # If the episode is done, break out of the loop.
            self.frame_stack.push(obs)

        if self.processing_method == "Binary":
            self.state = self.frame_stack.get_binary()
        elif self.processing_method == "Greyscale":
            self.state = self.frame_stack.get_greyscale()


        # Return the new state, total reward, done flag, truncated, and info.
        return self.state.flatten()[None, :], total_reward, done, truncated, info
    
    def save_sample_binary_frame(self) -> np.ndarray:
        """Sample a binary frame from the environment"""
        # Reset the environment to get the initial observation.
        obs, _ = self.env.reset()
        # Process the observation to binary format.
        binary_frame = self.frame_stack.get_binary()
        plt.imshow(binary_frame, cmap='gray')
        plt.axis('off')
        plt.savefig("sample_frames/sample_binary_frame.png")
        plt.close()

    def save_sample_greyscale_frame(self) -> np.ndarray:
        """Sample a binary frame from the environment"""
        # Reset the environment to get the initial observation.
        obs, _ = self.env.reset()
        # Process the observation to binary format.
        binary_frame = self.frame_stack.get_greyscale()
        plt.imshow(binary_frame, cmap='gray')
        plt.axis('off')
        plt.savefig("sample_frames/sample_greyscale_frame.png")
        plt.close()


    def close(self):
        """Close the environment"""

        self.env.close()
