import torch
import torch.nn as nn
from config import config
import numpy as np
from collections import deque
from tqdm import tqdm


def train_agent(agent, env, num_episodes=config["dqn"]["training_length"], max_steps_per_episode=500):

    rewards_per_episode = []
    mean_scores = []
    losses = []
    scores_window = deque(maxlen=50)
    for i in range(config["dqn"]["training_length"]//config['logging_interval']):
        print(f"Training for episode {i*config['logging_interval']+1} to {i*config['logging_interval']+config['logging_interval']}")
        for episode in tqdm(range(config['logging_interval'])):
            state = env.reset()
            episode_reward = 0
            episode_loss = []

            while True:
                action = agent.select_action(state)
                next_state, reward, done, _, info = env.step(action)

                agent.replay_buffer.push(state, action, reward, next_state, done)
                loss = agent.train_dqn_step()

                state = next_state
                episode_reward += reward
                if loss:
                    episode_loss.append(loss)

                if done:
                    break

            rewards_per_episode.append(episode_reward)
            # Monitor training progress with comprehensive logging and statistics.
            scores_window.append(episode_reward)
            mean_score = np.mean(scores_window)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            rewards_per_episode.append(episode_reward)
            mean_scores.append(mean_score)
            losses.append(avg_loss)


        print(f"Episode {(i+1)*config['logging_interval']} | Reward: {episode_reward:.2f} | "
          f"Mean Score: {mean_score:.2f} | Loss: {avg_loss:.4f}")

    return agent, rewards_per_episode, mean_scores, losses
