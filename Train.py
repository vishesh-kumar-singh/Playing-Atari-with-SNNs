def train_dqn_step(agent, replay_buffer, device=None):

    device = device or agent.device
    if len(replay_buffer) < agent.batch_size:
        return None
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(agent.batch_size)
    states      = torch.tensor(states, dtype=torch.float32, device=device)
    actions     = torch.tensor(actions, dtype=torch.long, device=device)
    rewards     = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones       = torch.tensor(dones, dtype=torch.float32, device=device)
    
    q_values = agent.policy_net(states)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_q_values = agent.target_net(next_states)
        max_next_q = next_q_values.max(1)[0]
        target_q = rewards + agent.gamma * max_next_q * (1 - dones)
    
    loss = nn.MSELoss()(q_selected, target_q)
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    
    agent.step_count += 1
    agent.update_epsilon()
    if agent.step_count % agent.target_update_freq == 0:
        agent.update_target_network()
    return loss.item()
