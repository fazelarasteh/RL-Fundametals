import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    """
    A simple deterministic grid world environment.
    The agent can move in four directions: up, down, left, right.
    If the agent tries to move outside the grid, it stays in place.
    """
    # Action indices
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, width=5, height=5, start_pos=(0, 0), goal_pos=None, obstacles=None):
        """
        Initialize the GridWorld environment
        
        Args:
            width (int): Width of the grid
            height (int): Height of the grid
            start_pos (tuple): Starting position (x, y)
            goal_pos (tuple): Goal position (x, y), if None, default to bottom-right
            obstacles (list): List of obstacle positions (x, y)
        """
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos if goal_pos is not None else (width-1, height-1)
        self.obstacles = obstacles if obstacles is not None else []
        
        # Check valid positions
        if not self._is_valid_pos(start_pos):
            raise ValueError(f"Invalid start position: {start_pos}")
        if not self._is_valid_pos(self.goal_pos):
            raise ValueError(f"Invalid goal position: {self.goal_pos}")
        for obs in self.obstacles:
            if not self._is_valid_pos(obs):
                raise ValueError(f"Invalid obstacle position: {obs}")
            if obs == start_pos or obs == self.goal_pos:
                raise ValueError(f"Obstacle position {obs} conflicts with start or goal")
        
        # Initialize state
        self.current_pos = self.start_pos
        self.steps = 0
        
    def reset(self):
        """
        Reset the environment to initial state
        
        Returns:
            tuple: Initial state (x, y)
        """
        self.current_pos = self.start_pos
        self.steps = 0
        return self.current_pos
    
    def step(self, action):
        """
        Take an action and return the new state, reward, and done flag
        
        Args:
            action (int): Action to take (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            tuple: (new_state, reward, done, info)
        """
        if action not in [self.UP, self.RIGHT, self.DOWN, self.LEFT]:
            raise ValueError(f"Invalid action: {action}")
        
        # Current position
        x, y = self.current_pos
        
        # Calculate new position
        if action == self.UP:
            new_pos = (x, min(y+1, self.height-1))
        elif action == self.RIGHT:
            new_pos = (min(x+1, self.width-1), y)
        elif action == self.DOWN:
            new_pos = (x, max(y-1, 0))
        elif action == self.LEFT:
            new_pos = (max(x-1, 0), y)
        
        # Check if new position is valid
        if new_pos in self.obstacles:
            new_pos = self.current_pos  # Stay in place if hitting an obstacle
        
        # Update state
        self.current_pos = new_pos
        self.steps += 1
        
        # Check if done
        done = (self.current_pos == self.goal_pos)
        
        # Calculate reward
        if done:
            reward = 1.0  # Positive reward for reaching the goal
        else:
            reward = -0.01  # Small negative reward for each step
        
        # Info dictionary
        info = {'steps': self.steps}
        
        return self.current_pos, reward, done, info
    
    def _is_valid_pos(self, pos):
        """
        Check if a position is valid (within the grid)
        
        Args:
            pos (tuple): Position (x, y)
            
        Returns:
            bool: True if valid, False otherwise
        """
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height
    
    def render(self, ax=None):
        """
        Render the grid world
        
        Args:
            ax (matplotlib.axes.Axes): Axes to plot on
            
        Returns:
            matplotlib.axes.Axes: The axes with the rendered grid
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create grid
        ax.set_xlim(-0.5, self.width-0.5)
        ax.set_ylim(-0.5, self.height-0.5)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        ax.grid(True)
        
        # Remove axis labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Plot obstacles
        for obs in self.obstacles:
            rect = plt.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, color='black')
            ax.add_patch(rect)
        
        # Plot goal
        rect = plt.Rectangle((self.goal_pos[0]-0.5, self.goal_pos[1]-0.5), 1, 1, color='green', alpha=0.5)
        ax.add_patch(rect)
        
        # Plot agent
        circle = plt.Circle((self.current_pos[0], self.current_pos[1]), 0.3, color='red')
        ax.add_patch(circle)
        
        # Add coordinates
        for i in range(self.width):
            for j in range(self.height):
                ax.text(i, j, f"({i},{j})", ha='center', va='center', fontsize=8)
        
        return ax

    def get_all_states(self):
        """
        Get all possible states in the environment
        
        Returns:
            list: List of all states (x, y)
        """
        states = []
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                if pos not in self.obstacles:
                    states.append(pos)
        return states
    
    def get_all_actions(self):
        """
        Get all possible actions
        
        Returns:
            list: List of all actions
        """
        return [self.UP, self.RIGHT, self.DOWN, self.LEFT] 
        
    def get_transition_prob(self, state, action, next_state):
        """
        Get the transition probability from state to next_state by taking action
        Since this is a deterministic environment, the probability is either 1.0 or 0.0
        
        Args:
            state (tuple): Current state (x, y)
            action (int): Action to take
            next_state (tuple): Next state (x, y)
            
        Returns:
            float: Transition probability (1.0 or 0.0)
        """
        # If state is invalid or an obstacle, return 0
        if state not in self.get_all_states():
            return 0.0
            
        # Calculate the expected next state
        x, y = state
        expected_next_state = state  # Default to current state (for obstacles/boundaries)
        
        if action == self.UP:
            if y + 1 < self.height:
                expected_next_state = (x, y + 1)
        elif action == self.RIGHT:
            if x + 1 < self.width:
                expected_next_state = (x + 1, y)
        elif action == self.DOWN:
            if y - 1 >= 0:
                expected_next_state = (x, y - 1)
        elif action == self.LEFT:
            if x - 1 >= 0:
                expected_next_state = (x - 1, y)
        
        # If expected next state is an obstacle, stay in place
        if expected_next_state in self.obstacles:
            expected_next_state = state
            
        # Return 1.0 if next_state matches expected_next_state, 0.0 otherwise
        return 1.0 if expected_next_state == next_state else 0.0
        
    def get_reward(self, state, action, next_state):
        """
        Get the reward for transitioning from state to next_state by taking action
        
        Args:
            state (tuple): Current state (x, y)
            action (int): Action to take
            next_state (tuple): Next state (x, y)
            
        Returns:
            float: Reward value
        """
        # If next state is the goal, return positive reward
        if next_state == self.goal_pos:
            return 1.0
        # Otherwise return small negative reward
        return -0.01 