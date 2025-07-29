from Environment import BreakoutEnv
from ANNAgent import ANNAgent
from Experience_Replay import ReplayBuffer
from plots import plot_training_results
from Train import train_agent
from config import config

env = BreakoutEnv(rendering_mode='rgb_array',processing_method="Greyscale")
state=env.reset()
state_shape=state.shape
action_size = env.action_space.n
agent=ANNAgent(state_shape=state_shape,action_size=action_size)


if __name__ == "__main__":
    """Main execution block for the assignment"""
    # Display assignment title and information.
    print("Final Assignment")
    print("Team: Runtime Terrors")
    

    # Execute the training process and collect results.
    agent, episode_rewards, mean_scores, losses = train_agent(agent=agent, env=env)

    # Saving the trained model
    agent.save_model(filepath="weights/greyscale_model_weights.pth")
    print("Model Weights are saved")

    # Generate and display comprehensive visualizations
    plot_training_results(episode_rewards, mean_scores, losses, results_path="plots/Greyscale_Train.png")

    # Print completion message and file references.
    print(f'''Training is Completed.
      - The model weights are being saved in weights folder
      - And the plots of the results had been saved in plots folder''')
    
    