# Reinforcement Learning Fundamentals

A repository for implementing and experimenting with fundamental reinforcement learning algorithms from scratch.

## Overview

This project provides a structured approach to learning and implementing key reinforcement learning algorithms. It's designed to help you understand the fundamentals by building each algorithm step by step, focusing on clear implementations rather than performance optimizations.

## Algorithms Covered

The repository is organized to cover the following reinforcement learning algorithms, in order of increasing complexity:

1. **Multi-Armed Bandits**
   - Epsilon-Greedy
   - Upper Confidence Bound (UCB)

2. **Dynamic Programming**
   - Value Iteration
   - Policy Iteration

3. **Temporal Difference Learning**
   - SARSA (on-policy)
   - Q-Learning (off-policy)

4. **Advanced Topics** (planned for future expansion)
   - Function Approximation
   - Policy Gradient Methods
   - Deep Reinforcement Learning

## Project Structure

- `src/algorithms/`: Implementation of RL algorithms
- `src/environments/`: Custom environments and wrappers
- `src/utils/`: Utility functions and helpers
- `results/`: Directory for storing experiment results
- `FIRST_CHALLENGE.md`, `SECOND_CHALLENGE.md`, etc.: Step-by-step challenges to implement each algorithm
- `INSTRUCTIONS.md`: General instructions and learning path

## Getting Started

### Prerequisites

- Python 3.8+
- Basic understanding of reinforcement learning concepts

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/RL-Fundamentals.git
   cd RL-Fundamentals
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Challenges

The repository contains a series of challenges to guide your learning:

1. Start with `FIRST_CHALLENGE.md` to implement the Epsilon-Greedy algorithm for multi-armed bandits
2. Move on to `SECOND_CHALLENGE.md` to implement Value Iteration for grid world
3. Continue with `THIRD_CHALLENGE.md` to implement Q-Learning for grid world

Each challenge provides:
- Theoretical background
- Step-by-step implementation instructions
- Testing procedures
- Expected outcomes

### Running Experiments

For Multi-Armed Bandits:
```bash
python src/run_bandit.py --strategy epsilon_greedy --epsilon 0.1
python src/run_bandit.py --strategy ucb --c 2.0
```

For Dynamic Programming:
```bash
python src/run_grid_dp.py --algorithm value_iteration --gamma 0.99
python src/run_grid_dp.py --algorithm policy_iteration --gamma 0.99 --obstacles
```

For Temporal Difference Learning:
```bash
python src/run_td_learning.py --algorithm sarsa --episodes 500
python src/run_td_learning.py --algorithm q_learning --episodes 500 --obstacles
```

## Learning Path

Follow these steps to get the most out of this repository:

1. Read the theoretical background in each challenge file
2. Implement the required components
3. Run the experiments and analyze the results
4. Experiment with different parameters to understand their effects
5. Compare different algorithms on the same environment

See `INSTRUCTIONS.md` for a more detailed learning path.

## Resources

- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd Edition)
- David Silver's RL Course (UCL/DeepMind)
- OpenAI Spinning Up

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
