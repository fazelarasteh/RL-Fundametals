import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1, initial_values=0.0):
        """
        Epsilon-Greedy strategy for multi-armed bandits
        
        Args:
            n_arms (int): Number of arms
            epsilon (float): Exploration rate
            initial_values (float): Initial value estimates for each arm
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.ones(n_arms) * initial_values
        self.arm_counts = np.zeros(n_arms)
        
    def select_arm(self):
        """
        Select an arm using the epsilon-greedy strategy
        
        Returns:
            int: Selected arm index
        """
        # TODO: Implement epsilon-greedy action selection ????
        # Hint: With probability epsilon, choose a random arm
        # Otherwise, choose the arm with the highest estimated value
        pass
        
    def update(self, arm, reward):
        """
        Update value estimates based on observed reward
        
        Args:
            arm (int): The arm that was pulled
            reward (float): The reward that was received
        """
        # TODO: Implement the update rule for the value estimate
        # Hint: Use incremental update formula to update the value estimate for the selected arm
        # Remember to also update the count for the selected arm
        pass


class UCB:
    def __init__(self, n_arms, c=2.0, initial_values=0.0):
        """
        Upper Confidence Bound strategy for multi-armed bandits
        
        Args:
            n_arms (int): Number of arms
            c (float): Exploration parameter
            initial_values (float): Initial value estimates for each arm
        """
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.ones(n_arms) * initial_values
        self.arm_counts = np.zeros(n_arms)
        self.t = 0  # Total number of time steps
        
    def select_arm(self):
        """
        Select an arm using the UCB strategy
        
        Returns:
            int: Selected arm index
        """
        # TODO: Implement UCB action selection
        # Hint: For each arm, calculate its UCB value as: Q(a) + c * sqrt(ln(t) / N(a))
        # where Q(a) is the current value estimate, t is the total time steps,
        # and N(a) is the number of times arm a has been selected
        # Return the arm with the highest UCB value
        pass
        
    def update(self, arm, reward):
        """
        Update value estimates based on observed reward
        
        Args:
            arm (int): The arm that was pulled
            reward (float): The reward that was received
        """
        # TODO: Implement the update rule for the value estimate
        # Remember to also update the count for the selected arm and the total time step
        pass 