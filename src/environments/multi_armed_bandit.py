import numpy as np

class MultiArmedBandit:
    def __init__(self, n_arms=10, reward_distributions='normal'):
        """
        A simple multi-armed bandit environment
        
        Args:
            n_arms (int): Number of arms/actions
            reward_distributions (str): Type of reward distribution ('normal', 'bernoulli')
        """
        self.n_arms = n_arms
        self.reward_distributions = reward_distributions
        self.reset()
        
    def reset(self):
        """
        Reset the environment with new reward parameters
        
        Returns:
            None
        """
        if self.reward_distributions == 'normal':
            # For normal distribution, we'll have mean and standard deviation
            self.true_means = np.random.normal(0, 1, self.n_arms)
            self.true_stds = np.ones(self.n_arms) * 0.5
        elif self.reward_distributions == 'bernoulli':
            # For Bernoulli, we just need success probabilities
            self.true_means = np.random.uniform(0, 1, self.n_arms)
        else:
            raise ValueError(f"Unknown distribution: {self.reward_distributions}")
        
        self.optimal_arm = np.argmax(self.true_means)
        self.total_regret = 0
        
    def step(self, arm):
        """
        Take an action (pull an arm) and receive a reward
        
        Args:
            arm (int): The arm to pull (action to take)
            
        Returns:
            float: The reward received
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm index: {arm}. Must be between 0 and {self.n_arms-1}")
        
        # Generate reward based on the distribution
        if self.reward_distributions == 'normal':
            reward = np.random.normal(self.true_means[arm], self.true_stds[arm])
        elif self.reward_distributions == 'bernoulli':
            reward = np.random.binomial(1, self.true_means[arm])
        
        # Calculate regret
        optimal_reward = self.true_means[self.optimal_arm]
        regret = optimal_reward - self.true_means[arm]
        self.total_regret += regret
        
        return reward
    
    def get_optimal_arm(self):
        """
        Return the index of the optimal arm (for evaluation)
        
        Returns:
            int: Index of the arm with highest expected reward
        """
        return self.optimal_arm
    
    def get_expected_rewards(self):
        """
        Return the true expected rewards for each arm (for evaluation)
        
        Returns:
            numpy.ndarray: Array of expected rewards
        """
        return self.true_means.copy() 