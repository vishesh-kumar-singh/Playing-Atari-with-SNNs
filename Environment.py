import numpy as np
from typing import Tuple
import gym as gym
from gym import spaces
from Image_Processing import FrameStack, poisson_spike_encoding
import ale_py
import os
from config import config
import matplotlib.pyplot as plt

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

class BreakoutEnv:
    def __init__(self, env_name: str = "BreakoutNoFrameskip-v4", frame_skip: int = config['dqn']['frame_skip'],rendering_mode: str = "rgb_array", processing_method: str ="Binary", is_SNN=False):
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
        self.is_SNN = is_SNN
    
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

        if self.is_SNN:
            self.state= poisson_spike_encoding(self.state[None,:], time_steps=config['snn']['time_steps'])[0]

        # Return the properly formatted state for the agent.
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action with frame skipping"""
        # Convert agent action to environment action.
        agent_action = action
        done = False
        total_reward = 0.0

        for _ in range(config['dqn']['action_repeat']):
            # Step the environment with the agent's action.
            obs, reward, done, truncated, info = self.env.step(agent_action)
            if reward==0:
                new_reward = 0
            else:
                new_reward = 1
            total_reward += new_reward
            if done or truncated:
                break

            # If the episode is done, break out of the loop.
            self.frame_stack.push(obs)

        if self.processing_method == "Binary":
            self.state = self.frame_stack.get_binary()
        elif self.processing_method == "Greyscale":
            self.state = self.frame_stack.get_greyscale()

        if self.is_SNN:
            flat_state = self.state.flatten()[None, :]  # shape: (1, 6400)

            self.state= poisson_spike_encoding(self.state[None,:], time_steps=config['snn']['time_steps'])[0]


        # Return the new state, total reward, done flag, truncated, and info.
        return self.state, total_reward, done, truncated, info


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
