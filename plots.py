import matplotlib.pyplot as plt


def plot_training_results(episode_rewards, mean_scores, losses):
    """Create comprehensive training visualization"""

    # Set up a multi-panel figure to display various training metrics.
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle('Training Results Overview', fontsize=16)

    # Plot episode rewards with rolling mean to show learning progress.
    axs[0].plot(episode_rewards, label='Episode Rewards', color='blue', alpha=0.6)
    axs[0].set_ylabel('Rewards')
    axs[0].legend()
    axs[0].grid(True)

    # Display mean score progression with target achievement line.
    axs[1].plot(mean_scores, label='Mean Scores', color='green', alpha=0.6)
    axs[1].axhline(y=30, color='red', linestyle='--', label='Mean Human Score (30)')
    axs[1].axhline(y=30, color='red', linestyle='--', label='Target Score (12)')
    axs[1].set_ylabel('Mean Score')
    axs[1].legend()
    axs[1].grid(True)

    # Show training loss evolution to monitor learning stability.
    axs[2].plot(losses, label='Training Loss', color='orange', alpha=0.6)
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results.png')
    plt.show()


    # Create score distribution histogram to analyze performance spread.
    plt.figure(figsize=(12, 6))
    plt.hist(episode_rewards, bins=30, color='purple', alpha=0.7)
    plt.title('Distribution of Episode Rewards')
    plt.xlabel('Rewards')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Add proper labeling, legends, and save the results.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results_overview.png')
    plt.show()