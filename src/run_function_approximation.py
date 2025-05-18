#!/usr/bin/env python
import numpy as np
import argparse
import matplotlib.pyplot as plt
from environments.gridworld import GridWorld
from algorithms.function_approximation import LinearQFunctionApproximation, DeepQNetwork, run_episode_with_fa

def simple_feature_extractor(state):
    """
    Extract features from state representation
    For GridWorld: Convert (x, y) state to one-hot encoding
    
    Args:
        state: State representation from environment
        
    Returns:
        numpy.array: Feature vector
    """
    # GridWorld states are (x, y) coordinates
    x, y = state
    
    # Create one-hot encoding of the state
    # For a 5x5 grid, we have 25 possible states
    features = np.zeros(25)
    features[x * 5 + y] = 1.0
    
    return features

def run_experiment(env, agent, num_episodes, feature_extractor=None, render_every=None):
    """
    Run experiment for given number of episodes
    
    Args:
        env: Environment
        agent: Agent
        num_episodes (int): Number of episodes to run
        feature_extractor: Function to extract features from state
        render_every (int): Render every n episodes (None for no rendering)
        
    Returns:
        tuple: (episode_rewards, episode_steps)
    """
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        # Render occasionally
        render = render_every is not None and episode % render_every == 0
        
        # Run episode
        reward, steps = run_episode_with_fa(env, agent, feature_extractor, render=render)
        
        # Store results
        episode_rewards.append(reward)
        episode_steps.append(steps)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward (last 10): {avg_reward:.2f}")
    
    return episode_rewards, episode_steps

def create_linear_agent(env, alpha=0.01, gamma=0.99, epsilon=0.1):
    """Create a linear function approximation agent"""
    num_features = 25  # For 5x5 GridWorld
    num_actions = len(env.get_all_actions())
    return LinearQFunctionApproximation(num_features, num_actions, alpha, gamma, epsilon)

def create_deep_agent(env, alpha=0.001, gamma=0.99, epsilon=0.1):
    """Create a deep Q-network agent"""
    input_dim = 25  # For 5x5 GridWorld
    num_actions = len(env.get_all_actions())
    return DeepQNetwork(input_dim, num_actions, alpha=alpha, gamma=gamma, epsilon=epsilon)

def plot_results(rewards, steps, title):
    """Plot rewards and steps"""
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'{title} - Rewards')
    
    # Plot moving average of rewards
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r--')
    
    # Plot steps
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'{title} - Steps')
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run function approximation experiments')
    parser.add_argument('--algorithm', type=str, choices=['linear', 'deep'], default='linear',
                      help='Algorithm to use')
    parser.add_argument('--episodes', type=int, default=500,
                      help='Number of episodes to run')
    parser.add_argument('--alpha', type=float, default=0.01,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                      help='Exploration rate')
    parser.add_argument('--render_every', type=int, default=None,
                      help='Render every n episodes')
    parser.add_argument('--obstacles', action='store_true',
                      help='Add obstacles to the environment')
    args = parser.parse_args()
    
    # Create environment
    env = GridWorld(width=5, height=5, obstacles=[] if not args.obstacles else [(1, 1), (2, 3), (3, 1)])
    
    # Create agent
    if args.algorithm == 'linear':
        agent = create_linear_agent(env, args.alpha, args.gamma, args.epsilon)
        algorithm_name = "Linear Q-Function Approximation"
    else:  # deep
        agent = create_deep_agent(env, args.alpha, args.gamma, args.epsilon)
        algorithm_name = "Deep Q-Network"
    
    # Run experiment
    rewards, steps = run_experiment(
        env, 
        agent, 
        args.episodes, 
        feature_extractor=simple_feature_extractor,
        render_every=args.render_every
    )
    
    # Plot results
    plot_results(rewards, steps, algorithm_name)
    
    # Test policy (greedy)
    print("\nTesting final policy (greedy):")
    state = env.reset()
    done = False
    total_reward = 0
    env.render()
    
    while not done:
        features = simple_feature_extractor(state)
        action = agent.get_action(features, greedy=True)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main() 