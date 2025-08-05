from Environment import BreakoutEnv
from Agent import SNNAgent, ANNAgent
import numpy as np
from tqdm import tqdm

env = BreakoutEnv(rendering_mode='rgb_array',processing_method="Binary", save_sample_frames=False)
state= env.reset()
state_shape = state.shape
action_size = env.action_space.n
agent=SNNAgent(state_shape=state_shape,action_size=action_size)
# agent.load_model(filepath="weights/binary_model_weights.pth")
agent.load_ann_weights(filepath="weights/binary_model_weights.pth",scale_1=20,scale_2=100)
agent.epsilon = 0.1

def plot_rewards(rewards, save=True, show=True, save_path='rewards_plot.png'):
    import matplotlib.pyplot as plt
    plt.hist(rewards, bins=range(16), edgecolor='black')  # bins from 0 to 15
    plt.xlabel('Rewards')
    plt.ylabel('Frequency')
    plt.title('SNN Reward Frequency')
    plt.xticks(range(0, 16))
    plt.yticks(range(0, 101, 10))
    plt.xlim(0, 15)
    plt.ylim(0, 100)

    if save:
        plt.savefig(save_path)
        print(f"Plot saved as '{save_path}'")
    if show:
        plt.show()

print(f"Testing SNN with {agent.epsilon}% randomness")
rewards=[]
for i in tqdm(range(100)):
    state = env.reset()
    episode_reward = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)

        state = next_state
        episode_reward += reward
        if done:
            break
    rewards.append(episode_reward)
env.close()
    

print(f"Mean Reward: {np.mean(rewards)} | Max Reward: {max(rewards)}")
plot_rewards(rewards,save=True, show=False, save_path='plots/SNN_binary_test_rewards.png')

    
