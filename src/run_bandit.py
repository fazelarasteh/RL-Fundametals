import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.multi_armed_bandit import MultiArmedBandit
from src.algorithms.bandit_strategies import EpsilonGreedy, UCB
from src.utils.plotting import plot_rewards
from src.utils.logger import setup_logger

def run_experiment(strategy_class, strategy_params, env_params, n_steps=1000, n_runs=10):
    """
    Run a bandit experiment with the given strategy and environment parameters
    
    Args:
        strategy_class: Class of the strategy to use
        strategy_params (dict): Parameters for the strategy
        env_params (dict): Parameters for the environment
        n_steps (int): Number of steps per run
        n_runs (int): Number of independent runs
        
    Returns:
        tuple: (average_rewards, average_optimal_actions, average_regret)
    """
    # Arrays to store results
    rewards = np.zeros((n_runs, n_steps))
    optimal_actions = np.zeros((n_runs, n_steps))
    regrets = np.zeros((n_runs, n_steps))
    
    for run in range(n_runs):
        # Create environment and strategy
        env = MultiArmedBandit(**env_params)
        strategy = strategy_class(**strategy_params, n_arms=env.n_arms)
        
        # Get the optimal arm for this run
        optimal_arm = env.get_optimal_arm()
        
        # Run the experiment
        for step in range(n_steps):
            # Select arm
            arm = strategy.select_arm()
            
            # Check if selected the optimal arm
            optimal_actions[run, step] = 1 if arm == optimal_arm else 0
            
            # Get reward
            reward = env.step(arm)
            rewards[run, step] = reward
            
            # Update strategy
            strategy.update(arm, reward)
            
            # Record cumulative regret
            regrets[run, step] = env.total_regret
    
    # Average results over runs
    avg_rewards = np.mean(rewards, axis=0)
    avg_optimal_actions = np.mean(optimal_actions, axis=0)
    avg_regret = np.mean(regrets, axis=0)
    
    return avg_rewards, avg_optimal_actions, avg_regret

def main():
    parser = argparse.ArgumentParser(description='Run bandit experiments')
    parser.add_argument('--strategy', type=str, default='epsilon_greedy',
                        choices=['epsilon_greedy', 'ucb'],
                        help='Strategy to use')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon value for epsilon-greedy')
    parser.add_argument('--c', type=float, default=2.0,
                        help='Exploration parameter for UCB')
    parser.add_argument('--n_arms', type=int, default=10,
                        help='Number of arms')
    parser.add_argument('--distribution', type=str, default='normal',
                        choices=['normal', 'bernoulli'],
                        help='Reward distribution')
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='Number of steps per run')
    parser.add_argument('--n_runs', type=int, default=10,
                        help='Number of independent runs')
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(name=f'bandit_{args.strategy}')
    logger.info(f'Starting experiment with args: {args}')
    
    # Set up parameters
    env_params = {
        'n_arms': args.n_arms,
        'reward_distributions': args.distribution
    }
    
    if args.strategy == 'epsilon_greedy':
        strategy_class = EpsilonGreedy
        strategy_params = {'epsilon': args.epsilon}
    elif args.strategy == 'ucb':
        strategy_class = UCB
        strategy_params = {'c': args.c}
    
    # Run experiment
    avg_rewards, avg_optimal_actions, avg_regret = run_experiment(
        strategy_class, strategy_params, env_params, 
        n_steps=args.n_steps, n_runs=args.n_runs
    )
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot average rewards
    plt.subplot(1, 3, 1)
    plot_rewards(avg_rewards.tolist(), window=10, title="Average Rewards")
    
    # Plot percentage of optimal actions
    plt.subplot(1, 3, 2)
    plt.plot(avg_optimal_actions)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title('Percentage of Optimal Actions')
    plt.grid(True, alpha=0.3)
    
    # Plot average regret
    plt.subplot(1, 3, 3)
    plt.plot(avg_regret)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Average Cumulative Regret')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save the figure
    plt.savefig(f'results/bandit_{args.strategy}.png')
    logger.info(f'Results saved to results/bandit_{args.strategy}.png')
    
    # Print final results
    logger.info(f'Final average reward: {avg_rewards[-1]:.4f}')
    logger.info(f'Final percentage of optimal actions: {avg_optimal_actions[-1] * 100:.2f}%')
    logger.info(f'Final average regret: {avg_regret[-1]:.4f}')

if __name__ == '__main__':
    main() 