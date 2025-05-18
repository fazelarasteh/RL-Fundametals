# Fifth Challenge: Policy Gradient Methods

## Overview

In this challenge, you will implement policy gradient methods, which directly optimize the policy without using a value function as an intermediary. Policy gradient methods are particularly useful for continuous action spaces and can learn stochastic policies.

## Tasks

### 1. REINFORCE (Monte Carlo Policy Gradient)

Complete the `end_episode()` method in the `REINFORCEAgent` class in `src/algorithms/policy_gradient.py`:

```python
def end_episode(self):
    """
    End the episode and update policy parameters
    """
    # TODO: Implement REINFORCE update rule
    # 1. Calculate discounted returns for each step
    # 2. Update policy parameters using policy gradient
    # 3. Clear episode history
    pass
```

You need to:
1. Calculate discounted returns (G_t) for each step in the episode
   - G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k+1}
   - Or use the recursive formula: G_t = r_{t+1} + gamma * G_{t+1}
2. Update policy parameters using the policy gradient theorem
   - For each (state, action) pair, increase the probability of actions that led to higher returns
   - Update rule: policy_weights[action] += alpha * G_t * gradient of log policy
3. Clear the episode history for the next episode

### 2. REINFORCE with Baseline

Complete the `end_episode()` method in the `REINFORCEWithBaseline` class:

```python
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
    pass
```

You need to:
1. Calculate discounted returns for each step (same as in REINFORCE)
2. Update value function parameters to predict the expected returns
   - Update rule: value_weights += alpha_value * (G_t - V(s_t)) * features
3. Calculate advantages as the difference between returns and estimated values
   - A_t = G_t - V(s_t)
4. Update policy parameters using the advantage as a baseline
   - Update rule: policy_weights[action] += alpha_policy * A_t * gradient of log policy
5. Clear the episode history for the next episode

### 3. Actor-Critic

Complete the `update()` method in the `ActorCritic` class:

```python
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
    pass
```

You need to:
1. Calculate the TD error using the critic (value function)
   - If done: TD_error = reward - V(state)
   - Otherwise: TD_error = reward + gamma * V(next_state) - V(state)
2. Update critic parameters using the TD error
   - Update rule: critic_weights += alpha_critic * TD_error * state
3. Update actor parameters using the TD error as the advantage
   - Update rule: actor_weights[action] += alpha_actor * TD_error * gradient of log policy

## Testing Your Implementation

Run your implementation with:

```bash
# Test REINFORCE
python src/run_policy_gradient.py --algorithm reinforce --episodes 1000

# Test REINFORCE with baseline
python src/run_policy_gradient.py --algorithm reinforce_baseline --episodes 1000

# Test Actor-Critic with obstacles
python src/run_policy_gradient.py --algorithm actor_critic --episodes 1000 --obstacles

# Try different learning rates
python src/run_policy_gradient.py --algorithm reinforce --learning_rate 0.005
```

## Expected Results

1. **REINFORCE**:
   - The agent should learn a reasonable policy but with high variance in performance
   - Learning may be slow due to the Monte Carlo nature of the algorithm
   - The policy should eventually converge to a near-optimal solution

2. **REINFORCE with Baseline**:
   - Should learn faster than vanilla REINFORCE due to reduced variance
   - The baseline helps stabilize learning by reducing the variance of policy gradient estimates
   - The value function should provide a good estimate of expected returns

3. **Actor-Critic**:
   - Should learn faster than REINFORCE methods due to TD learning
   - The actor-critic architecture allows for online updates without waiting for episode completion
   - The critic helps reduce variance while the actor improves the policy

## Hints

1. For REINFORCE:
   - Calculate returns by working backwards from the end of the episode
   - The gradient of the log policy for a softmax policy is: features - weighted sum of features

2. For REINFORCE with Baseline:
   - The baseline should not change the expected gradient, only reduce its variance
   - A good baseline is the state value function, which predicts the expected return

3. For Actor-Critic:
   - The TD error serves as an unbiased estimate of the advantage function
   - The critic's job is to accurately predict the value of each state
   - The actor's job is to improve the policy based on the critic's feedback

4. General tips:
   - Policy gradient methods can be sensitive to the learning rate
   - Start with small learning rates and gradually increase if needed
   - Monitor the entropy of the policy to ensure it doesn't become too deterministic too quickly

## Bonus Challenges

1. Implement a natural policy gradient method
2. Implement Proximal Policy Optimization (PPO) or Trust Region Policy Optimization (TRPO)
3. Extend the implementation to handle continuous action spaces
4. Add entropy regularization to encourage exploration
5. Compare the performance of different policy gradient methods on various environments 