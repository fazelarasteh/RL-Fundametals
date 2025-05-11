import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.gridworld import GridWorld
from src.algorithms.dynamic_programming import value_iteration, policy_iteration
from src.utils.plotting import plot_values
from src.utils.logger import setup_logger

def visualize_policy(policy, env, ax=None):
    """
    Visualize a policy on a grid
    
    Args:
        policy (dict): Policy mapping states to actions
        env (GridWorld): The environment
        ax (matplotlib.axes.Axes): Axes to plot on
    
    Returns:
        matplotlib.axes.Axes: Axes with the policy visualization
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the grid
    ax = env.render(ax)
    
    # Draw policy arrows
    for state, action in policy.items():
        x, y = state
        
        if state == env.goal_pos:
            continue  # Skip goal state
        
        if action == env.UP:
            dx, dy = 0, 0.3
        elif action == env.RIGHT:
            dx, dy = 0.3, 0
        elif action == env.DOWN:
            dx, dy = 0, -0.3
        elif action == env.LEFT:
            dx, dy = -0.3, 0
        else:
            continue  # Skip if no action
        
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    return ax

def visualize_value_function(V, env):
    """
    Visualize a value function on a grid
    
    Args:
        V (dict): Value function mapping states to values
        env (GridWorld): The environment
    
    Returns:
        matplotlib.figure.Figure: Figure with the value function visualization
    """
    # Convert V to a 2D array
    values = np.zeros((env.height, env.width))
    min_val = float('inf')
    max_val = float('-inf')
    
    for state, value in V.items():
        x, y = state
        values[env.height - 1 - y, x] = value  # Flip y for visualization
        min_val = min(min_val, value)
        max_val = max(max_val, value)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(values, cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Value')
    
    # Add labels
    for i in range(env.width):
        for j in range(env.height):
            state = (i, env.height - 1 - j)  # Convert back to state coordinates
            if state in V:
                text = ax.text(i, j, f"{V[state]:.2f}", ha="center", va="center", 
                              color="white" if values[j, i] > (min_val + max_val) / 2 else "black", 
                              fontsize=8)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Remove major ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title("Value Function")
    
    return fig

def run_experiment(args):
    """
    Run an experiment with the specified algorithm
    
    Args:
        args: Command line arguments
    """
    # Set up logger
    logger = setup_logger(name=f'grid_dp_{args.algorithm}')
    logger.info(f'Starting experiment with args: {args}')
    
    # Create environment
    obstacles = [(2, 2), (2, 3), (2, 4)] if args.obstacles else []
    env = GridWorld(width=args.width, height=args.height, 
                   start_pos=(0, 0), 
                   goal_pos=(args.width-1, args.height-1),
                   obstacles=obstacles)
    
    # Run algorithm
    logger.info(f'Running {args.algorithm} with gamma={args.gamma}')
    if args.algorithm == 'value_iteration':
        V, policy = value_iteration(env, gamma=args.gamma, theta=args.theta, max_iterations=args.max_iterations)
    elif args.algorithm == 'policy_iteration':
        V, policy = policy_iteration(env, gamma=args.gamma, theta=args.theta, max_iterations=args.max_iterations)
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Visualize policy
    fig, ax = plt.subplots(figsize=(10, 10))
    visualize_policy(policy, env, ax)
    plt.title(f'Optimal Policy ({args.algorithm})')
    plt.savefig(f'results/grid_dp_{args.algorithm}_policy.png')
    
    # Visualize value function
    fig = visualize_value_function(V, env)
    plt.title(f'Value Function ({args.algorithm})')
    plt.savefig(f'results/grid_dp_{args.algorithm}_values.png')
    
    logger.info(f'Results saved to results/grid_dp_{args.algorithm}_*.png')
    
    return V, policy

def main():
    parser = argparse.ArgumentParser(description='Run dynamic programming experiments on GridWorld')
    parser.add_argument('--algorithm', type=str, default='value_iteration',
                        choices=['value_iteration', 'policy_iteration'],
                        help='Algorithm to use')
    parser.add_argument('--width', type=int, default=5,
                        help='Width of the grid')
    parser.add_argument('--height', type=int, default=5,
                        help='Height of the grid')
    parser.add_argument('--obstacles', action='store_true',
                        help='Add obstacles to the grid')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--theta', type=float, default=1e-8,
                        help='Convergence threshold')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Maximum number of iterations')
    
    args = parser.parse_args()
    
    V, policy = run_experiment(args)
    
    # Print some results
    print(f"\nFinal value of start state (0,0): {V[(0,0)]:.6f}")
    print(f"Final value of state next to goal ({args.width-2},{args.height-1}): {V[(args.width-2, args.height-1)]:.6f}")
    print(f"Final value of goal state ({args.width-1},{args.height-1}): {V[(args.width-1, args.height-1)]:.6f}")

if __name__ == '__main__':
    main() 