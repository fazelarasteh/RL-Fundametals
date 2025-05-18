# Fourth Challenge: Function Approximation

## Overview

In this challenge, you will move beyond tabular methods and implement function approximation techniques for reinforcement learning. Function approximation allows RL algorithms to scale to problems with large or continuous state spaces by generalizing across similar states.

## Tasks

### 1. Linear Q-Function Approximation

Complete the `update()` method in the `LinearQFunctionApproximation` class in `src/algorithms/function_approximation.py`:

```python
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
    pass
```

You need to:
1. Calculate the current Q-value for the state-action pair using the current weights
2. Calculate the target Q-value using the Q-learning update rule
3. Compute the TD error (difference between target and current Q-value)
4. Update the weights for the chosen action using gradient descent
   - The gradient of a linear function approximator is simply the feature vector
   - Update rule: weights += alpha * TD_error * features

### 2. Deep Q-Network (DQN)

Complete the missing methods in the `DeepQNetwork` class:

1. Implement the `get_action()` method:
   - With probability epsilon, select a random action
   - Otherwise, select the action with the highest Q-value from the policy network

2. Implement the `store_experience()` method:
   - Add the experience tuple (state, action, reward, next_state, done) to the replay buffer
   - If the buffer exceeds the maximum size, remove the oldest experience

3. Implement the `_batch_update()` method:
   - Sample a mini-batch of experiences from the replay buffer
   - Calculate target Q-values using the target network
   - Calculate current Q-values using the policy network
   - Calculate the loss (e.g., mean squared error between target and current Q-values)
   - Update the policy network using gradient descent

4. Implement the `_update_target_network()` method:
   - Copy the weights from the policy network to the target network

## Testing Your Implementation

Run your implementation with:

```bash
# Test linear function approximation
python src/run_function_approximation.py --algorithm linear --episodes 500

# Test DQN with obstacles in the environment
python src/run_function_approximation.py --algorithm deep --episodes 500 --obstacles

# Try different learning rates
python src/run_function_approximation.py --algorithm linear --alpha 0.05
```

## Expected Results

1. **Linear Function Approximation**:
   - The agent should learn a reasonable policy for the GridWorld environment
   - The average reward should increase over time
   - The agent should find the optimal path to the goal

2. **Deep Q-Network**:
   - With proper implementation, DQN should perform at least as well as linear function approximation
   - The agent may require more episodes to converge due to the complexity of neural network training
   - The experience replay mechanism should help stabilize learning

## Hints

1. For linear function approximation:
   - The gradient of Q(s,a) with respect to weights is simply the feature vector
   - Only update the weights for the action that was taken

2. For DQN:
   - Use a simple neural network architecture for this task (1-2 hidden layers should be sufficient)
   - The target network helps stabilize training by providing a fixed target for the Q-learning update
   - Experience replay breaks correlations between consecutive samples and improves data efficiency

3. Feature representation:
   - The provided `simple_feature_extractor` function creates a one-hot encoding of the state
   - This representation works well for the GridWorld environment but may not be optimal for more complex environments

## Bonus Challenges

1. Implement a more sophisticated feature extractor that captures more information about the environment
2. Add prioritized experience replay to the DQN implementation
3. Implement Double DQN to reduce overestimation bias
4. Compare the performance of different function approximation methods on various environments 