# Challenge 2: Implement Value Iteration for Grid World

In this challenge, you'll implement the Value Iteration algorithm to solve the Grid World environment. This is a fundamental dynamic programming technique in reinforcement learning that iteratively improves the value function to find the optimal policy.

## Background

Value Iteration is based on the Bellman optimality equation, which states that the optimal value of a state is the expected return for taking the best action from that state:

V*(s) = max_a [ sum_s' P(s'|s,a) * (R(s,a,s') + γ * V*(s')) ]

Where:
- V*(s) is the optimal value of state s
- P(s'|s,a) is the probability of transitioning to state s' when taking action a from state s
- R(s,a,s') is the reward received for taking action a in state s and ending up in state s'
- γ (gamma) is the discount factor

For the deterministic Grid World, the transition probability P(s'|s,a) is 1 for the resulting state and 0 for all others, simplifying the equation.

## Your Task

Open `src/algorithms/dynamic_programming.py` and implement the `value_iteration` function:

```python
def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    """
    Value Iteration algorithm for finding the optimal value function
    
    Args:
        env: The environment with methods get_all_states() and get_all_actions()
        gamma (float): Discount factor
        theta (float): Convergence threshold
        max_iterations (int): Maximum number of iterations
    
    Returns:
        tuple: (V, policy) where V is the value function and policy is the optimal policy
    """
    # Initialize value function
    states = env.get_all_states()
    actions = env.get_all_actions()
    V = {state: 0.0 for state in states}
    
    # TODO: Implement value iteration algorithm
    # For each iteration:
    #   1. Set delta = 0
    #   2. For each state s:
    #      a. Store the old value v = V[s]
    #      b. Find new V[s] = max_a sum_s' p(s'|s,a)[r(s,a,s') + gamma*V[s']]
    #      c. Update delta = max(delta, |v - V[s]|)
    #   3. If delta < theta, break
    
    # TODO: Extract policy from value function
    # For each state, choose the action that maximizes the value
    policy = {state: None for state in states}  # Replace None with the best action
    
    return V, policy
```

## Implementation Approach

For the Grid World environment, you can implement Value Iteration as follows:

1. **Iteration Loop**:
   - Create a loop that runs for `max_iterations` or until convergence
   - Initialize `delta = 0` at the start of each iteration

2. **Value Update**:
   - For each state `s` in the environment:
     - Store the current value `v = V[s]`
     - Initialize a list to store values for each action
     - For each action `a`:
       - Simulate taking action `a` from state `s` 
       - Get the next state `s'`, reward `r`, and whether the episode is done
       - Compute the value for this action: `r + gamma * V[s']` (if not done)
     - Set `V[s]` to the maximum value across all actions
     - Update `delta = max(delta, |v - V[s]|)`
   - If `delta < theta`, break the loop (convergence achieved)

3. **Policy Extraction**:
   - For each state `s`:
     - Compute the value for each action as above
     - Set `policy[s]` to the action with the highest value

## How to Test Your Implementation

After implementing Value Iteration, run:

```bash
python src/run_grid_dp.py --algorithm value_iteration --gamma 0.99
```

Then try with obstacles:

```bash
python src/run_grid_dp.py --algorithm value_iteration --gamma 0.99 --obstacles
```

## Expected Outcomes

If your implementation is correct:
1. The value function should show higher values near the goal and lower values farther away
2. The policy should point toward the goal, taking the shortest path while avoiding obstacles
3. The algorithm should converge within a reasonable number of iterations

## Hints

- For step 2.b, you need to find the maximum value across all actions. For each action, simulate the environment to get the next state and reward.
- For the Grid World environment, you can simulate taking an action using:
  ```python
  x, y = state
  # Calculate new position based on action
  # Check if hitting wall or obstacle
  # Calculate reward based on conditions
  ```
- Remember that in the Grid World we provided, states are represented as tuples `(x, y)` and actions are integers (0 for UP, 1 for RIGHT, 2 for DOWN, 3 for LEFT)

Good luck! 