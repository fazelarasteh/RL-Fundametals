import numpy as np
from collections import defaultdict

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

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=1000):
    """
    Policy Evaluation algorithm for finding the value function of a given policy
    
    Args:
        env: The environment with methods get_all_states() and get_all_actions()
        policy (dict): A policy mapping states to actions
        gamma (float): Discount factor
        theta (float): Convergence threshold
        max_iterations (int): Maximum number of iterations
        
    Returns:
        dict: Value function for the given policy
    """
    # Initialize value function
    states = env.get_all_states()
    V = {state: 0.0 for state in states}
    
    # TODO: Implement policy evaluation algorithm
    # For each iteration:
    #   1. Set delta = 0
    #   2. For each state s:
    #      a. Store the old value v = V[s]
    #      b. Calculate new V[s] = sum_s' p(s'|s,policy[s])[r(s,policy[s],s') + gamma*V[s']]
    #      c. Update delta = max(delta, |v - V[s]|)
    #   3. If delta < theta, break
    
    return V

def policy_improvement(env, V, gamma=0.99):
    """
    Policy Improvement algorithm for finding a better policy from a value function
    
    Args:
        env: The environment with methods get_all_states() and get_all_actions()
        V (dict): Value function
        gamma (float): Discount factor
    
    Returns:
        dict: Improved policy
    """
    states = env.get_all_states()
    actions = env.get_all_actions()
    policy = {state: None for state in states}
    
    # TODO: Implement policy improvement algorithm
    # For each state, choose the action that maximizes the value
    
    return policy

def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    """
    Policy Iteration algorithm for finding the optimal policy
    
    Args:
        env: The environment with methods get_all_states() and get_all_actions()
        gamma (float): Discount factor
        theta (float): Convergence threshold for policy evaluation
        max_iterations (int): Maximum number of iterations
        
    Returns:
        tuple: (V, policy) where V is the value function and policy is the optimal policy
    """
    # Initialize policy
    states = env.get_all_states()
    actions = env.get_all_actions()
    policy = {state: actions[0] for state in states}  # Start with first action for all states
    
    # TODO: Implement policy iteration algorithm
    # 1. Policy Evaluation: Compute V^(pi) using policy evaluation
    # 2. Policy Improvement: Compute pi' = greedy(V^(pi))
    # 3. If pi' = pi, stop and return V and pi; otherwise, go to step 1
    
    return V, policy 