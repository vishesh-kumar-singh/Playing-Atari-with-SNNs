config = {
    
    "training_length": 4000,  # episodes
    "mini_batch_size": 64,
    "replay_memory_size": 200000,
    "replay_memory_init_size": 50000,
    "agent_history_length": 4,
    "target_network_update_frequency": 10000,
    "discount_factor": 0.99,  # gamma
    "frame_skip": 3,
    "action_repeat": 4,
    "update_frequency": 4,
    "update_rule": "RMSProp",
    "learning_rate": 0.00001,
    "gradient_momentum": 0.95,
    "squared_gradient_momentum": 0.95,
    "min_squared_gradient": 0.01,
    "initial_epsilon": 1.0,
    "final_epsilon": 0.1,
    "final_exploration_step": 200000,
    "refractory_period": 0,  # in ms
    "threshold_voltage": -52,
    "resting_voltage": -65,
    "voltage_decay": 0.01,
    "time_steps": 500,
    "theta_plus": 0.05,
    "theta_decay": 1e-7,
    

    "logging_interval":10
}