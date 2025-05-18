import numpy as np
import random
from collections import defaultdict

class REINFORCEAgent:
    """REINFORCE: Monte Carlo Policy Gradient"""
    
    def __init__(self, input_dim, num_actions, alpha=0.01, gamma=0.99):
        """
        Initialize the agent
        
        Args:
            input_dim (int): Dimension of input state
            num_actions (int): Number of possible actions
            alpha (float): Learning rate
            gamma (float): Discount factor
        """
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize policy parameters (weights for a linear model)
        self.policy_weights = np.zeros((num_actions, input_dim))
        
        # Store episode history
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def get_policy_probabilities(self, state):
        """
        Calculate action probabilities using softmax policy
        
        Args:
            state (numpy.array): Current state
            
        Returns:
            numpy.array: Action probabilities
        """
        # Calculate action scores
        action_scores = np.dot(self.policy_weights, state)
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(action_scores - np.max(action_scores))  # Subtract max for numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def get_action(self, state):
        """
        Sample an action from the policy
        
        Args:
            state (numpy.array): Current state
            
        Returns:
            int: Selected action
        """
        probabilities = self.get_policy_probabilities(state)
        action = np.random.choice(self.num_actions, p=probabilities)
        return action
    
    def store_transition(self, state, action, reward):
        """
        Store state, action, reward for the current step
        
        Args:
            state (numpy.array): Current state
            action (int): Action taken
            reward (float): Reward received
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def end_episode(self):
        """
        End the episode and update policy parameters
        """
        # TODO: Implement REINFORCE update rule
        # 1. Calculate discounted returns for each step
        # 2. Update policy parameters using policy gradient
        # 3. Clear episode history
        pass
        episode_discounted_returns = [None]*len(self.episode_rewards)
        for i in range(len(self.episode_rewards)-1,-1,-1):
            episode_discounted_returns[i] = self.episode_rewards[i] + self.gamma * episode_discounted_returns[i+1] if i<len(self.episode_rewards)-1 else self.episode_rewards[i]
        
        for s,a,g in zip(self.episode_states,self.episode_actions,episode_discounted_returns):
            policy_probabilities = self.get_policy_probabilities(s)
            gradient = - policy_probabilities.reshape(-1,1)*s.reshape(1,-1)
            gradient[a] += s
            self.policy_weights += self.alpha * g * gradient
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []


class REINFORCEWithBaseline:
    """REINFORCE with baseline for variance reduction"""
    
    def __init__(self, input_dim, num_actions, alpha_policy=0.01, alpha_value=0.01, gamma=0.99):
        """
        Initialize the agent
        
        Args:
            input_dim (int): Dimension of input state
            num_actions (int): Number of possible actions
            alpha_policy (float): Learning rate for policy
            alpha_value (float): Learning rate for value function
            gamma (float): Discount factor
        """
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.alpha_policy = alpha_policy
        self.alpha_value = alpha_value
        self.gamma = gamma
        
        # Initialize policy parameters (weights for a linear model)
        self.policy_weights = np.zeros((num_actions, input_dim))
        
        # Initialize value function parameters
        self.value_weights = np.zeros(input_dim)
        
        # Store episode history
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def get_policy_probabilities(self, state):
        """
        Calculate action probabilities using softmax policy
        
        Args:
            state (numpy.array): Current state
            
        Returns:
            numpy.array: Action probabilities
        """
        # Calculate action scores
        action_scores = np.dot(self.policy_weights, state)
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(action_scores - np.max(action_scores))  # Subtract max for numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def get_state_value(self, state):
        """
        Calculate state value
        
        Args:
            state (numpy.array): Current state
            
        Returns:
            float: State value
        """
        return np.dot(self.value_weights, state)
    
    def get_action(self, state):
        """
        Sample an action from the policy
        
        Args:
            state (numpy.array): Current state
            
        Returns:
            int: Selected action
        """
        probabilities = self.get_policy_probabilities(state)
        action = np.random.choice(self.num_actions, p=probabilities)
        return action
    
    def store_transition(self, state, action, reward):
        """
        Store state, action, reward for the current step
        
        Args:
            state (numpy.array): Current state
            action (int): Action taken
            reward (float): Reward received
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def end_episode(self):
        """
        End the episode and update policy and value parameters
        """
        # TODO: Implement REINFORCE with baseline update rule
        # 1. Calculate discounted returns for each step
        # 2. Update value function parameters
        # 3. Calculate advantages (returns - values)
        # 4. Update policy parameters using policy gradient with baseline
        # 5. Clear episode history
        T = len(self.episode_states)

        episode_state_values = [self.get_state_value(state) for state in self.episode_states]
        episode_discounted_returns = [None]*len(self.episode_rewards)

        for i in range(T-1,-1,-1):
            episode_discounted_returns[i] = 0 if i == (T-1) else self.gamma * episode_discounted_returns[i+1] + self.episode_rewards[i]

        for s,g,v in zip(self.episode_states,episode_discounted_returns,episode_state_values):
            value_grad = (g-v)*s
            self.value_weights += self.alpha_value * value_grad

        episode_advantages = [r-v for r,v in zip(episode_discounted_returns,episode_state_values)]

        for s,a,adv in zip(self.episode_states,self.episode_actions,episode_advantages):
            p = self.get_policy_probabilities(s)
            policy_grad = - p.reshape(-1,1)*s.reshape(1,-1)
            policy_grad[a] += s
            self.policy_weights += self.alpha_policy * adv * policy_grad

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []


class ActorCritic:
    """Actor-Critic method with TD learning"""
    
    def __init__(self, input_dim, num_actions, alpha_actor=0.01, alpha_critic=0.01, gamma=0.99):
        """
        Initialize the agent
        
        Args:
            input_dim (int): Dimension of input state
            num_actions (int): Number of possible actions
            alpha_actor (float): Learning rate for actor (policy)
            alpha_critic (float): Learning rate for critic (value function)
            gamma (float): Discount factor
        """
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma = gamma
        
        # Initialize actor parameters (weights for a linear model)
        self.actor_weights = np.zeros((num_actions, input_dim))
        
        # Initialize critic parameters
        self.critic_weights = np.zeros(input_dim)
    
    def get_policy_probabilities(self, state):
        """
        Calculate action probabilities using softmax policy
        
        Args:
            state (numpy.array): Current state
            
        Returns:
            numpy.array: Action probabilities
        """
        # Calculate action scores
        action_scores = np.dot(self.actor_weights, state)
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(action_scores - np.max(action_scores))  # Subtract max for numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def get_state_value(self, state):
        """
        Calculate state value
        
        Args:
            state (numpy.array): Current state
            
        Returns:
            float: State value
        """
        return np.dot(self.critic_weights, state)
    
    def get_action(self, state):
        """
        Sample an action from the policy
        
        Args:
            state (numpy.array): Current state
            
        Returns:
            int: Selected action
        """
        probabilities = self.get_policy_probabilities(state)
        action = np.random.choice(self.num_actions, p=probabilities)
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Update actor and critic parameters
        
        Args:
            state (numpy.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.array): Next state
            done (bool): Whether the episode is done
        """
        # TODO: Implement Actor-Critic update rule
        # 1. Calculate TD error using critic (value function)
        # 2. Update critic parameters using TD error
        # 3. Update actor parameters using TD error as the advantage

        # td_error = r+ gamma max q(ns,na) (1-done) - q(s,a)
        td_error = reward + self.gamma * self.get_state_value(next_state) * (1-done) - self.get_state_value(state)
        grad_td_error = td_error * state

        action_probs = self.get_policy_probabilities(state)
        grad_actor = - action_probs.reshape(-1,1)*state.reshape(1,-1)
        grad_actor[action] += state

        self.critic_weights += self.alpha_critic * grad_td_error
        self.actor_weights += self.alpha_actor * td_error * grad_actor


def run_policy_gradient_episode(env, agent, max_steps=1000, render=False):
    """
    Run a single episode for a policy gradient agent
    
    Args:
        env: Environment with reset() and step() methods
        agent: Policy gradient agent
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
        
        # For Actor-Critic
        if hasattr(agent, 'update'):
            agent.update(state, action, reward, next_state, done)
        # For REINFORCE and REINFORCE with baseline
        elif hasattr(agent, 'store_transition'):
            agent.store_transition(state, action, reward)
        
        # Update state and reward
        state = next_state
        total_reward += reward
        
        # Render if needed
        if render:
            env.render()
        
        # Break if done
        if done:
            break
    
    # For REINFORCE and REINFORCE with baseline
    if hasattr(agent, 'end_episode'):
        agent.end_episode()
    
    return total_reward, step + 1 