from Environment import BreakoutEnv
from ANNAgent import ANNAgent

env = BreakoutEnv(rendering_mode='human',processing_method="Greyscale")
state=env.reset()
state_shape = state.shape
action_size = env.action_space.n
agent=ANNAgent(state_shape=state_shape,action_size=action_size)
agent.load_model(filepath="weights/greyscale_model_weights.pth")
agent.epsilon = 0.1
agent.epsilon_min=0.1

for i in range(5):
    state = env.reset()
    episode_reward = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)

        state = next_state
        episode_reward += reward
        if done:
            break
    
    print(f"Episode {i+1} Reward: {episode_reward:.2f}")
    
    
