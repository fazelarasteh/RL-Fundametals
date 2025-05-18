import numpy as np
import random
from collections import defaultdict

class TDAgent:
    """Base class for temporal difference learning agents"""
    
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize the agent
        
        Args:
            actions (list): List of possible actions
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Exploration rate
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        
    def get_action(self, state, greedy=False):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state
            greedy (bool): If True, select the best action (no exploration)
            
        Returns:
            int: Selected action
        """
        if not greedy and random.random() < self.epsilon:
            # Explore: select a random action
            return random.choice(self.actions)
        else:
            # Exploit: select the best action
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        raise NotImplementedError("Subclasses must implement update method")


class SARSA(TDAgent):
    """SARSA: On-policy TD control"""
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value using SARSA update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Select next action using the policy (epsilon-greedy)
        next_action = self.get_action(next_state)
        
        # Update Q-value using the SARSA update rule:
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state][next_action] * (1-done) - self.q_table[state][action]
        )


class QLearning(TDAgent):
    """Q-Learning: Off-policy TD control"""
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value using Q-Learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Update Q-value using the Q-Learning update rule:
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        max_next_q_value = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next_q_value * (1-done) - self.q_table[state][action]
        )


def run_episode(env, agent, max_steps=1000, render=False):
    """
    Run a single episode
    
    Args:
        env: Environment with reset() and step() methods
        agent: Agent with get_action() and update() methods
        max_steps (int): Maximum number of steps per episode
        render (bool): Whether to render the environment
        
    Returns:
        tuple: (total_reward, steps)
    """
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # Select action
        action = agent.get_action(state)
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        
        # Update agent
        agent.update(state, action, reward, next_state, done)
        
        # Update state and reward
        state = next_state
        total_reward += reward
        
        # Render if needed
        if render:
            env.render()
        
        # Break if done
        if done:
            break
    
    return total_reward, step + 1 