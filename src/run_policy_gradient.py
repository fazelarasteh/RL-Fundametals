#!/usr/bin/env python
import numpy as np
import argparse
import matplotlib.pyplot as plt
from src.environments.gridworld import GridWorldEnv
from src.algorithms.policy_gradient import (
    REINFORCEAgent, 
    REINFORCEWithBaseline, 
    ActorCritic, 
    run_policy_gradient_episode
)

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

def run_experiment(env, agent, num_episodes, render_every=None):
    """
    Run experiment for given number of episodes
    
    Args:
        env: Environment
        agent: Agent
        num_episodes (int): Number of episodes to run
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
        reward, steps = run_policy_gradient_episode(env, agent, render=render)
        
        # Store results
        episode_rewards.append(reward)
        episode_steps.append(steps)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward (last 10): {avg_reward:.2f}")
    
    return episode_rewards, episode_steps

def create_agent(env, algorithm, learning_rate=0.01, gamma=0.99):
    """Create a policy gradient agent"""
    input_dim = 25  # For 5x5 GridWorld with one-hot encoding
    num_actions = len(env.actions)
    
    if algorithm == 'reinforce':
        return REINFORCEAgent(input_dim, num_actions, alpha=learning_rate, gamma=gamma)
    elif algorithm == 'reinforce_baseline':
        return REINFORCEWithBaseline(input_dim, num_actions, alpha_policy=learning_rate, 
                                   alpha_value=learning_rate, gamma=gamma)
    elif algorithm == 'actor_critic':
        return ActorCritic(input_dim, num_actions, alpha_actor=learning_rate, 
                         alpha_critic=learning_rate, gamma=gamma)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

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
    plt.savefig(f"{title.lower().replace(' ', '_').replace('-', '_')}.png")
    plt.show()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run policy gradient experiments')
    parser.add_argument('--algorithm', type=str, 
                      choices=['reinforce', 'reinforce_baseline', 'actor_critic'], 
                      default='reinforce',
                      help='Algorithm to use')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of episodes to run')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--render_every', type=int, default=None,
                      help='Render every n episodes')
    parser.add_argument('--obstacles', action='store_true',
                      help='Add obstacles to the environment')
    args = parser.parse_args()
    
    # Create environment
    env = GridWorldEnv(size=5, obstacles=args.obstacles)
    
    # Wrap environment to transform states to features
    class FeatureEnv:
        def __init__(self, env, feature_extractor):
            self.env = env
            self.feature_extractor = feature_extractor
            self.actions = env.actions
        
        def reset(self):
            state = self.env.reset()
            return self.feature_extractor(state)
        
        def step(self, action):
            state, reward, done, info = self.env.step(action)
            return self.feature_extractor(state), reward, done, info
        
        def render(self):
            return self.env.render()
    
    # Create wrapped environment
    feature_env = FeatureEnv(env, simple_feature_extractor)
    
    # Create agent
    agent = create_agent(env, args.algorithm, args.learning_rate, args.gamma)
    
    # Algorithm name for display
    algorithm_names = {
        'reinforce': 'REINFORCE',
        'reinforce_baseline': 'REINFORCE with Baseline',
        'actor_critic': 'Actor-Critic'
    }
    algorithm_name = algorithm_names[args.algorithm]
    
    # Run experiment
    rewards, steps = run_experiment(
        feature_env, 
        agent, 
        args.episodes, 
        render_every=args.render_every
    )
    
    # Plot results
    plot_results(rewards, steps, algorithm_name)
    
    # Test policy
    print("\nTesting final policy:")
    features = feature_env.reset()
    done = False
    total_reward = 0
    env.render()  # Render the unwrapped environment for better visualization
    
    while not done:
        action = agent.get_action(features)
        features, reward, done, _ = feature_env.step(action)
        total_reward += reward
        env.render()  # Render the unwrapped environment
    
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main() 