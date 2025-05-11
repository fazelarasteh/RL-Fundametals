# Reinforcement Learning Fundamentals - Implementation Guide

This repository provides a structured approach to learn and implement fundamental reinforcement learning (RL) algorithms from scratch. Below is a guide to help you implement these algorithms step by step.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Explore the project structure:
   - `src/algorithms/`: Implementation of RL algorithms
   - `src/environments/`: Custom environments for testing
   - `src/utils/`: Utility functions and helpers

## Implementation Path

Follow this recommended path to implement the algorithms:

### 1. Multi-Armed Bandits

The simplest RL problem. Start by implementing:

- **Epsilon-Greedy** strategy in `src/algorithms/bandit_strategies.py`
  - Implement the `select_arm()` method to choose arms with epsilon-greedy policy
  - Implement the `update()` method to update value estimates

- **UCB (Upper Confidence Bound)** strategy
  - Implement the `select_arm()` method using the UCB formula
  - Implement the `update()` method to track arm counts and values

**Test your implementation:**
```bash
python src/run_bandit.py --strategy epsilon_greedy --epsilon 0.1
python src/run_bandit.py --strategy ucb --c 2.0
```

### 2. Dynamic Programming

Learn how to solve known MDPs with:

- **Value Iteration** in `src/algorithms/dynamic_programming.py`
  - Complete the implementation based on the provided skeleton
  - Focus on the Bellman optimality equation

- **Policy Iteration**
  - Implement policy evaluation
  - Implement policy improvement
  - Combine them in the policy iteration algorithm

**Test your implementation:**
```bash
python src/run_grid_dp.py --algorithm value_iteration --gamma 0.99
python src/run_grid_dp.py --algorithm policy_iteration --gamma 0.99 --obstacles
```

### 3. Temporal Difference Learning

Implement model-free methods:

- **SARSA (on-policy TD)** in `src/algorithms/td_learning.py`
  - Complete the `update()` method using the SARSA update rule
  - Remember to select the next action using the epsilon-greedy policy

- **Q-Learning (off-policy TD)**
  - Complete the `update()` method using the Q-learning update rule
  - Use the max future Q-value regardless of the policy

**Test your implementation:**
```bash
python src/run_td_learning.py --algorithm sarsa --episodes 500
python src/run_td_learning.py --algorithm q_learning --episodes 500 --obstacles
```

### 4. Function Approximation (Next Steps)

After mastering tabular methods, move to function approximation:

- Implement linear function approximation for Q-learning
- Explore simple neural networks for Q-value estimation
- Experiment with experience replay

### 5. Policy Gradient Methods (Advanced)

For more advanced implementations:

- Implement REINFORCE algorithm
- Add baseline for variance reduction
- Implement Actor-Critic methods

## Guidance for Implementation

When implementing the algorithms, follow these best practices:

1. **Understand First**: Make sure you understand the algorithm before implementing it
2. **Start Simple**: Begin with the simplest version of the algorithm
3. **Test Incrementally**: After implementing each component, test it on simple problems
4. **Experiment**: Try different parameters and environments to see how they affect performance
5. **Visualize**: Use the provided plotting utilities to visualize results

## Debugging Tips

- For bandits: Check if your agent converges to the optimal arm over time
- For DP: Verify that your value function converges and the resulting policy is sensible
- For TD: Compare the performance of different algorithms and parameter settings

## Resources

- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd Edition)
- David Silver's RL Course (UCL/DeepMind)
- OpenAI Spinning Up

Happy learning and implementing! 