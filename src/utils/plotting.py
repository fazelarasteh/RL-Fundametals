import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, window=10, title="Rewards over Time", save_path=None):
    """
    Plot rewards over time with optional smoothing
    
    Args:
        rewards (list): List of rewards
        window (int): Size of the smoothing window
        title (str): Title of the plot
        save_path (str): Path to save the figure, if None, the figure is shown
    """
    plt.figure(figsize=(10, 6))
    
    # Original rewards
    plt.plot(rewards, alpha=0.3, label='Original Rewards')
    
    # Smoothed rewards
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed, label=f'Smoothed (window={window})')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_values(values, title="State Values", save_path=None):
    """
    Plot state values as a heatmap
    
    Args:
        values (numpy.ndarray): 2D array of state values
        title (str): Title of the plot
        save_path (str): Path to save the figure, if None, the figure is shown
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(values, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show() 