# Challenge 1: Implement Epsilon-Greedy for Multi-Armed Bandits

In this first challenge, you'll implement the epsilon-greedy strategy for the multi-armed bandit problem. This is a fundamental algorithm in reinforcement learning that balances exploration and exploitation.

## Background

The multi-armed bandit problem is a classic reinforcement learning problem where:

- You have `n` slot machines (arms)
- Each arm gives rewards from a fixed but unknown probability distribution
- Your goal is to maximize the total reward over a series of pulls

The epsilon-greedy strategy works as follows:
- With probability ε (epsilon), explore: choose a random arm
- With probability 1-ε, exploit: choose the arm with the highest estimated value

## Your Task

Open `src/algorithms/bandit_strategies.py` and implement:

1. The `select_arm()` method in the `EpsilonGreedy` class:
   ```python
   def select_arm(self):
       """
       Select an arm using the epsilon-greedy strategy
       
       Returns:
           int: Selected arm index
       """
       # TODO: Implement epsilon-greedy action selection
       # Hint: With probability epsilon, choose a random arm
       # Otherwise, choose the arm with the highest estimated value
   ```

2. The `update()` method in the `EpsilonGreedy` class:
   ```python
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
   ```

## Incremental Update Formula

The incremental update formula for the value estimate of arm `a` is:

```
Q(a) = Q(a) + (1/N(a)) * (R - Q(a))
```

Where:
- `Q(a)` is the estimated value of arm `a`
- `N(a)` is the number of times arm `a` has been pulled
- `R` is the reward received after pulling arm `a`

This formula computes the new estimate as the old estimate plus a step toward the observed reward, where the step size is `1/N(a)`.

## Testing Your Implementation

After implementing the methods, run:

```bash
python src/run_bandit.py --strategy epsilon_greedy --epsilon 0.1
```

Experiment with different values of epsilon (e.g., 0.01, 0.1, 0.5) and observe how it affects the performance.

## Expected Outcomes

If your implementation is correct:
1. The agent should converge to the optimal arm over time (% Optimal Action should increase)
2. The average reward should increase and stabilize
3. The cumulative regret should grow sublinearly

## Hints

- For `select_arm()`, use `random.random()` to generate a random number between 0 and 1, and compare with epsilon
- For choosing the best arm, use `np.argmax(self.q_values)` to get the arm with the highest estimated value
- For `update()`, make sure to increment `self.arm_counts[arm]` before updating the value estimate

Good luck! 