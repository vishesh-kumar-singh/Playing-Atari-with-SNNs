import numpy as np
from typing import Tuple
import gym as gym
from gym import spaces
from Image_Processing import Binary, Greyscale
import ale_py
import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

class BreakoutEnv:
    def __init__(self, env_name: str = "BreakoutNoFrameskip-v4", frame_skip: int = 4,rendering_mode: str = "rgb_array"):
        """Initialize environment and frame stack"""
        # Create the Gym environment and frame stack manager.
        self.env = gym.make(env_name,render_mode=rendering_mode)

        # Define the action space mapping for Breakout game.
        self.action_space = spaces.Discrete(4)     # 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT
        self.observation_space = self.env.observation_space
        self.state=None
        # Store frame skipping parameter for temporal efficiency.
        self.frame_skip = frame_skip
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial stacked state"""
        # Reset both the environment and frame stack.
        initial_obs1,_=self.env.reset()
        initial_obs2,_,_,_,_=self.env.step(0)  # Perform a NOOP action to get the first frame.
        initial_obs3,_,_,_,_=self.env.step(0)  # Perform a NOOP action to get the second frame.
        initial_obs4,_,_,_,_=self.env.step(0)  # Perform a NOOP action to get the third frame.

        # Process the initial observation and create the first state stack.
        self.state = Binary(initial_obs1,initial_obs2,initial_obs3,initial_obs4)


        # Return the properly formatted state for the agent.
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action with frame skipping"""
        # Convert agent action to environment action.
        agent_action = action
        # Execute the action multiple times to skip frames.
        total_reward = 0.0
        done = False
        frames=[]
        for i in range(4):
            for _ in range(self.frame_skip):
                obs, reward, done, truncated, info = self.env.step(agent_action)
                if reward==0:
                    total_reward += reward
                else:
                    total_reward += 1
                
                # If the episode is done, break out of the loop.
                if done or truncated:
                    break
            frames.append(obs)
        if not done and not truncated:
            # If the episode is not done, give the new state
            self.state = Binary(frames[0], frames[1], frames[2], frames[3])  


        # Return the new state, total reward, done flag, truncated, and info.
        return self.state, total_reward, done, truncated, info


    
    def close(self):
        """Close the environment"""

        self.env.close()
