import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.gridworld import GridWorld
from src.algorithms.td_learning import SARSA, QLearning, run_episode
from src.utils.plotting import plot_rewards
from src.utils.logger import setup_logger

def visualize_q_values(q_table, env, save_path=None):
    """
    Visualize Q-values for each state
    
    Args:
        q_table (defaultdict): Q-value table
        env (GridWorld): Environment
        save_path (str): Path to save the visualization
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Plot state values (max Q-value for each state)
    state_values = np.zeros((env.height, env.width))
    
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            if state not in env.obstacles:
                state_values[env.height - 1 - y, x] = np.max(q_table[state])
    
    # Plot heatmap
    im = axs[0].imshow(state_values, cmap='viridis')
    plt.colorbar(im, ax=axs[0])
    axs[0].set_title('State Values (Max Q-value)')
    
    # Add grid
    for ax in axs:
        ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 2. Plot policy (action with highest Q-value)
    # Create a grid
    ax = env.render(axs[1])
    
    # Draw policy arrows
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            if state in env.obstacles or state == env.goal_pos:
                continue
            
            # Get action with highest Q-value
            action = np.argmax(q_table[state])
            
            # Draw arrow based on action
            if action == env.UP:
                dx, dy = 0, 0.3
            elif action == env.RIGHT:
                dx, dy = 0.3, 0
            elif action == env.DOWN:
                dx, dy = 0, -0.3
            elif action == env.LEFT:
                dx, dy = -0.3, 0
            
            axs[1].arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    axs[1].set_title('Policy (Best Action)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def run_training(env, agent, n_episodes=500, eval_interval=10, render_final=True):
    """
    Train an agent on an environment
    
    Args:
        env (GridWorld): Environment
        agent (TDAgent): Agent
        n_episodes (int): Number of episodes to train
        eval_interval (int): Interval between evaluations
        render_final (bool): Whether to render the final policy
        
    Returns:
        tuple: (rewards, steps)
    """
    rewards = []
    steps_list = []
    eval_rewards = []
    
    for ep in range(1, n_episodes + 1):
        # Run a training episode
        reward, steps = run_episode(env, agent)
        rewards.append(reward)
        steps_list.append(steps)
        
        # Evaluate the agent
        if ep % eval_interval == 0:
            # Run an evaluation episode (greedy policy)
            env.reset()
            total_reward = 0
            state = env.reset()
            done = False
            step = 0
            
            while not done and step < 100:
                action = agent.get_action(state, greedy=True)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                step += 1
            
            eval_rewards.append(total_reward)
            print(f"Episode {ep}/{n_episodes}: Reward = {reward:.2f}, Steps = {steps}, " +
                  f"Eval Reward = {total_reward:.2f}")
    
    # Render final policy if requested
    if render_final:
        visualize_q_values(agent.q_table, env)
    
    return rewards, steps_list, eval_rewards

def main():
    parser = argparse.ArgumentParser(description='Run TD learning experiments on GridWorld')
    parser.add_argument('--algorithm', type=str, default='q_learning',
                        choices=['sarsa', 'q_learning'],
                        help='Algorithm to use')
    parser.add_argument('--width', type=int, default=5,
                        help='Width of the grid')
    parser.add_argument('--height', type=int, default=5,
                        help='Height of the grid')
    parser.add_argument('--obstacles', action='store_true',
                        help='Add obstacles to the grid')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Exploration rate')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes')
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(name=f'grid_td_{args.algorithm}')
    logger.info(f'Starting experiment with args: {args}')
    
    # Create environment
    obstacles = [(2, 2), (2, 3), (2, 4)] if args.obstacles else []
    env = GridWorld(width=args.width, height=args.height, 
                   start_pos=(0, 0), 
                   goal_pos=(args.width-1, args.height-1),
                   obstacles=obstacles)
    
    # Create agent
    actions = env.get_all_actions()
    
    if args.algorithm == 'sarsa':
        agent = SARSA(actions, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    elif args.algorithm == 'q_learning':
        agent = QLearning(actions, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    
    # Train the agent
    logger.info(f'Training {args.algorithm} agent for {args.episodes} episodes')
    start_time = time.time()
    rewards, steps, eval_rewards = run_training(env, agent, n_episodes=args.episodes, 
                                               eval_interval=args.episodes // 10, render_final=False)
    training_time = time.time() - start_time
    logger.info(f'Training completed in {training_time:.2f} seconds')
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Plot rewards
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_rewards(rewards, window=10, title="Training Rewards")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, args.episodes, args.episodes // 10), eval_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Reward')
    plt.title('Evaluation Rewards (Greedy Policy)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/grid_td_{args.algorithm}_rewards.png')
    
    # Visualize Q-values
    visualize_q_values(agent.q_table, env, save_path=f'results/grid_td_{args.algorithm}_q_values.png')
    
    logger.info(f'Results saved to results/grid_td_{args.algorithm}_*.png')
    
    # Print final results
    mean_last_rewards = np.mean(rewards[-10:])
    logger.info(f'Mean reward over last 10 episodes: {mean_last_rewards:.4f}')
    logger.info(f'Final evaluation reward: {eval_rewards[-1]:.4f}')

if __name__ == '__main__':
    main() 