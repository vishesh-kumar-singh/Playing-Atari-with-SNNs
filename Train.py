from config import config
import numpy as np
from collections import deque
from tqdm import tqdm
from Environment import BreakoutEnv
from Agent import SNNAgent, ANNAgent
import matplotlib.pyplot as plt

def plot_training_results(episode_rewards, mean_scores, losses, results_path : str= 'training_results.png'):
    """Create comprehensive training visualization"""

    # Set up a multi-panel figure to display various training metrics.
    fig, axs = plt.subplots(4, 1, figsize=(12, 24))
    fig.suptitle('Training Results Overview', fontsize=16)

    # Plot episode rewards with rolling mean to show learning progress.
    axs[0].plot(episode_rewards, label='Episode Rewards', color='blue', alpha=0.6)
    axs[2].set_xlabel('Episodes')
    axs[0].set_ylabel('Rewards')
    axs[0].legend()
    axs[0].grid(True)

    # Display mean score progression with target achievement line.
    axs[1].plot(mean_scores, label='Mean Scores', color='green', alpha=0.6)
    axs[2].set_xlabel('Episodes')
    axs[1].set_ylabel('Mean Score')
    axs[1].legend()
    axs[1].grid(True)

    # Show training loss evolution to monitor learning stability.
    axs[2].plot(losses, label='Training Loss', color='orange', alpha=0.6)
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid(True)


    axs[3].hist(episode_rewards, bins=30, color='purple', alpha=0.7)
    axs[3].set_xlabel('Rewards')
    axs[3].set_ylabel('Frequency')
    axs[2].legend()
    axs[3].grid(True)

    # Add proper labeling, legends, and save the results.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(results_path)
    plt.show()


def train_agent(agent, env):


    mean_scores = []
    losses = []
    scores_window = deque(maxlen=50)
    scores=[]
    for i in range(config["training_length"]//config['logging_interval']):
        print(f"Training for episode {i*config['logging_interval']+1} to {i*config['logging_interval']+config['logging_interval']}")
        for episode in tqdm(range(config['logging_interval'])):
            state = env.reset()
            episode_reward = 0
            episode_loss = []

            while True:
                action = agent.select_action(state)
                reward=0


                next_state, r, done, _, info = env.step(action)
                reward+=r

                agent.replay_buffer.push(state, action, reward, next_state, done)
                loss = agent.train_step()

                if loss:
                    episode_loss.append(loss)
                if done:
                    break
                state = next_state
                episode_reward += reward

                if done:
                    break

            
            # Monitor training progress with comprehensive logging and statistics.
            scores_window.append(episode_reward)
            mean_score = np.mean(scores_window)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            scores.append(episode_reward)
            mean_scores.append(mean_score)
            losses.append(avg_loss)


        print(f"Episode {(i+1)*config['logging_interval']} | Episode Reward: {episode_reward:.2f} | "
          f"Mean Score: {mean_score:.2f} | Average Loss: {avg_loss:.4f} | Maximum Reward: {max(scores_window)} | Step Count : {agent.step_count} \n")

    return agent, scores, mean_scores, losses


env = BreakoutEnv(rendering_mode='rgb_array',processing_method="Binary", save_sample_frames=False)
state= env.reset()
state_shape = state.shape
action_size = env.action_space.n
agent=SNNAgent(state_shape=state_shape,action_size=action_size)
agent.load_ann_weights(filepath="weights/binary_model_weights.pth", scale_1=10,scale_2=100)
agent.epsilon= 0.1


if __name__ == "__main__":
    """Main execution block for the assignment"""
    # Display assignment title and information.
    print("Final Assignment: SpikeVerse")


    # Execute the training process and collect results.
    agent, episode_rewards, mean_scores, losses = train_agent(agent=agent, env=env)

    # Saving the trained model
    agent.save_model(filepath="weights/snn_model_weights.pth")
    print("Model Weights are saved")

    # Generate and display comprehensive visualizations
    plot_training_results(episode_rewards, mean_scores, losses, results_path="plots/SNN_Fine_Tune.png")

    # Print completion message and file references.
    print(f'''Training is Completed.
      - The model weights are being saved in weights folder
      - And the plots of the results had been saved in plots folder''')