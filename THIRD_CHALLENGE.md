# Challenge 3: Implement Q-Learning for Grid World

In this challenge, you'll implement Q-Learning, a foundational off-policy temporal difference learning algorithm. Q-Learning learns the optimal action-value function directly, without needing a model of the environment.

## Background

Q-Learning maintains a table of Q-values for each state-action pair. The Q-value Q(s,a) represents the expected cumulative reward for taking action a in state s and then following the optimal policy.

The Q-Learning update rule is:

Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

Where:
- Q(s,a) is the value of taking action a in state s
- α (alpha) is the learning rate
- r is the immediate reward
- γ (gamma) is the discount factor
- s' is the next state
- max_a' Q(s',a') is the maximum Q-value over all actions a' in the next state s'

Key characteristics of Q-Learning:
- It's an off-policy algorithm, meaning it learns about the optimal policy while following an exploration policy
- It directly approximates the optimal action-value function, regardless of the policy being followed
- It guarantees convergence to the optimal policy (given sufficient exploration and a decreasing learning rate)

## Your Task

Open `src/algorithms/td_learning.py` and implement the `update` method in the `QLearning` class:

```python
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
    # TODO: Implement Q-Learning update rule
    # 1. Update Q-value using the Q-Learning update rule:
    #    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    # 2. If done (terminal state), the Q-value for the next state is 0
    pass
```

## Implementation Approach

Here's a step-by-step approach to implement the Q-learning update:

1. Get the current Q-value for the state-action pair: `current_q = self.q_table[state][action]`
2. If the episode is done (terminal state), set the next max Q-value to 0
3. Otherwise, find the maximum Q-value for the next state: `next_max_q = np.max(self.q_table[next_state])`
4. Calculate the TD target: `target = reward + self.gamma * next_max_q` (if not done)
5. Update the Q-value using the learning rate: `self.q_table[state][action] = current_q + self.alpha * (target - current_q)`

## How to Test Your Implementation

After implementing Q-Learning, run:

```bash
python src/run_td_learning.py --algorithm q_learning --episodes 500
```

Then try with obstacles:

```bash
python src/run_td_learning.py --algorithm q_learning --episodes 500 --obstacles
```

You can also experiment with different learning parameters:
```bash
python src/run_td_learning.py --algorithm q_learning --alpha 0.2 --gamma 0.9 --epsilon 0.2 --episodes 1000
```

## Expected Outcomes

If your implementation is correct:
1. The agent should learn to navigate to the goal in an increasingly efficient manner
2. The average reward per episode should increase over time
3. The learned policy should find the shortest path to the goal (accounting for obstacles)
4. The Q-values should reflect the expected future rewards, with higher values near the goal

## Hints

- Remember that Q-Learning is off-policy - it updates its value estimates using the maximum next state-action value, regardless of what action might actually be taken next
- When dealing with terminal states (done=True), there are no future rewards, so the next state Q-value should be treated as 0
- The state in Grid World is a tuple (x, y) representing the agent's position
- The action is an integer (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
- The q_table is a defaultdict mapping states to numpy arrays of Q-values for each action

## Extension

Once you've implemented basic Q-Learning, try these extensions:
1. Implement a decaying epsilon value that decreases over time (start with high exploration, gradually shift to exploitation)
2. Compare the performance of Q-Learning with SARSA by implementing the SARSA update method
3. Implement a more complex reward function in the GridWorld environment (e.g., penalties for certain regions)

Good luck! 