import numpy as np
from collections import defaultdict, deque

import torch.utils
import torch.utils.data

class LinearQFunctionApproximation:
    """Q-learning with linear function approximation"""
    
    def __init__(self, num_features, num_actions, alpha=0.01, gamma=0.99, epsilon=0.1):
        """
        Initialize the agent
        
        Args:
            num_features (int): Number of features in the state representation
            num_actions (int): Number of possible actions
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Exploration rate
        """
        self.num_features = num_features
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize weights for each action
        self.weights = np.zeros((num_actions, num_features))
        
    def get_q_value(self, features, action):
        """
        Calculate Q-value for given features and action
        
        Args:
            features (numpy.array): Feature vector
            action (int): Action
            
        Returns:
            float: Q-value
        """
        return np.dot(self.weights[action], features)
    
    def get_action(self, features, greedy=False):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            features (numpy.array): Feature vector
            greedy (bool): If True, select the best action (no exploration)
            
        Returns:
            int: Selected action
        """
        if not greedy and np.random.random() < self.epsilon:
            # Explore: select a random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploit: select the best action
            q_values = np.array([self.get_q_value(features, a) for a in range(self.num_actions)])
            return np.argmax(q_values)
    
    def update(self, features, action, reward, next_features, done):
        """
        Update the weights using Q-learning with linear function approximation
        
        Args:
            features (numpy.array): Current state features
            action (int): Action taken
            reward (float): Reward received
            next_features (numpy.array): Next state features
            done (bool): Whether the episode is done
        """
        # TODO: Implement Q-learning update rule with linear function approximation
        # 1. Calculate the current Q-value using the current weights
        # 2. Calculate the target Q-value:
        #    - If done, target = reward
        #    - Otherwise, target = reward + gamma * max_a Q(next_features, a)
        # 3. Calculate the TD error (target - current)
        # 4. Update the weights for the chosen action using gradient descent

        q = self.get_q_value(features, action)
        q_target = reward + self.gamma * max(self.get_q_value(next_features, a) for a in range(self.num_actions)) * (1-done)
        td_error = q_target - q
        
        # Update weights using the TD error and features
        self.weights[action] += self.alpha * td_error * features
        


import torch
import copy
class ExperienceDataset(torch.utils.data.Dataset):
    def __init__(self,max_size:int):
        super().__init__()
        self.experiences = deque(maxlen=max_size)
    
    def __len__(self):
        return len(self.experiences)
    
    def add(self,s,a,r,ns,d):
        self.experiences.append((s, a, ns, r, d))

    def __getitem__(self, index):
        s,a,ns,r,d = self.experiences[index]
        return {
            's': torch.FloatTensor(s),
            'a': torch.LongTensor([a]),  # Wrap action in a list to make it a 1D tensor
            'ns': torch.FloatTensor(ns),
            'r': torch.LongTensor([r]),
            'd': torch.FloatTensor([d])
        }
    
class DeepQNetwork:
    """Q-learning with neural network function approximation"""
    
    def __init__(self, input_dim, num_actions, hidden_dims=[32, 32], 
                 alpha=0.001, gamma=0.99, epsilon=0.1, buffer_size=10000, 
                 batch_size=32, target_update_freq=100):
        """
        Initialize the agent
        
        Args:
            input_dim (int): Dimension of input state
            num_actions (int): Number of possible actions
            hidden_dims (list): List of hidden layer dimensions
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Exploration rate
            buffer_size (int): Size of experience replay buffer
            batch_size (int): Mini-batch size for training
            target_update_freq (int): Frequency to update target network
        """
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # TODO: Initialize neural network models (policy network and target network)
        # Hint: You can use any neural network library like TensorFlow, PyTorch, or Keras
        self.q_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], num_actions)
        )
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.q_network.parameters())

        # Initialize experience dataset
        self.experience_dataset = ExperienceDataset(max_size=buffer_size)
        self.batch_size = batch_size
        self.dataloader = None  # We'll create this when we have enough samples
        
        # Step counter for target network update
        self.steps = 0
    
    def _get_dataloader(self):
        """Create or return existing dataloader"""
        if self.dataloader is None and len(self.experience_dataset) >= self.batch_size:
            self.dataloader = torch.utils.data.DataLoader(
                self.experience_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            )
        return self.dataloader
    
    def get_action(self, state, greedy=False):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state (numpy.array): Current state
            greedy (bool): If True, select the best action (no exploration)
            
        Returns:
            int: Selected action
        """
        # TODO: Implement epsilon-greedy action selection
        # 1. With probability epsilon, select a random action
        # 2. Otherwise, select the action with highest Q-value from the policy network
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_network(torch.tensor(state, dtype=torch.float32)).detach().numpy())
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state (numpy.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.array): Next state
            done (bool): Whether the episode is done
        """
        # TODO: Add experience to replay buffer with fixed size
        # If buffer is full, remove the oldest experience
        self.experience_dataset.add(state, action, reward, next_state, done)
    
    def update(self, state, action, reward, next_state, done):
        """
        Store experience and perform batch update if enough samples are available
        
        Args:
            state (numpy.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.array): Next state
            done (bool): Whether the episode is done
        """
        # Store experience
        self.store_experience(state, action, reward, next_state, done)
        
        # Increment step counter
        self.steps += 1
        
        # Perform batch update if enough samples are available
        self._batch_update()
        
        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self._update_target_network()
    
    def _batch_update(self):
        """Perform batch update using samples from replay buffer"""
        if len(self.experience_dataset) < self.batch_size:
            return
            
        # Get dataloader (creates it if needed)
        dataloader = self._get_dataloader()
        if dataloader is None:
            return
            
        # Get a batch of experiences
        batch = next(iter(dataloader))
        
        # Calculate target Q-values using the target network
        with torch.no_grad():
            q_targets = batch['r']+ self.gamma * self.target_network(batch['ns']).max(dim=-1)[0].unsqueeze(-1) * (1-batch['d'])
        
        # Calculate current Q-values using the policy network
        q_vals = self.q_network(batch['s']).gather(1, batch['a'])
        
        # Calculate the loss (e.g., mean squared error between target and current Q-values)
        td_error = torch.nn.functional.mse_loss(q_targets,q_vals)

        # Update the policy network using gradient descent
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

    def _update_target_network(self):
        """Update target network weights with policy network weights"""
        # TODO: Update target network weights
        # We can't directly assign parameters because:
        # 1. parameters() returns an iterator, not a modifiable collection
        # 2. PyTorch parameters are tensors that maintain computational graphs
        # 3. Direct assignment would break the gradient tracking
        # Instead, we use state_dict() and load_state_dict() which properly handle the copying
        self.target_network.load_state_dict(self.q_network.state_dict())




def run_episode_with_fa(env, agent, feature_extractor=None, max_steps=1000, render=False):
    """
    Run a single episode with function approximation
    
    Args:
        env: Environment with reset() and step() methods
        agent: Agent with get_action() and update() methods
        feature_extractor: Function to extract features from state (if None, use state as features)
        max_steps (int): Maximum number of steps per episode
        render (bool): Whether to render the environment
        
    Returns:
        tuple: (total_reward, steps)
    """
    state = env.reset()
    if feature_extractor:
        features = feature_extractor(state)
    else:
        features = state
        
    total_reward = 0
    
    for step in range(max_steps):
        # Select action
        action = agent.get_action(features)
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        
        if feature_extractor:
            next_features = feature_extractor(next_state)
        else:
            next_features = next_state
        
        # Update agent
        agent.update(features, action, reward, next_features, done)
        
        # Update state and reward
        features = next_features
        total_reward += reward
        
        # Render if needed
        if render:
            env.render()
        
        # Break if done
        if done:
            break
    
    return total_reward, step + 1 