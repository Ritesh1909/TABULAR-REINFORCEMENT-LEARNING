# Tabular Reinforcement Learning - Stochastic Windy Gridworld

This repository contains the implementation and analysis of tabular reinforcement learning techniques, including Dynamic Programming and Model-Free methods, applied to the Stochastic Windy Gridworld environment. This project was completed as part of the Reinforcement Learning coursework for the Master's in Computer Science program at Leiden University.

## Project Overview

- **Objective**: To understand and implement fundamental reinforcement learning algorithms in a controlled environment and analyze their behavior and efficiency.
- **Environment**: Stochastic Windy Gridworld, a 10x7 grid with stochastic wind effects that complicate the navigation challenges for the agent.
- **Algorithms Implemented**:
  - Dynamic Programming (Q-value Iteration)
  - Model-Free Methods (Q-learning and SARSA)
  - Exploration strategies (ε-greedy and Boltzmann)
  - Back-up methods comparison (on-policy vs off-policy)
- **Tools Used**: Custom Python scripts to simulate the environment and learning algorithms.

## Environment Setup

The Windy Gridworld is a grid-based environment where the agent moves through a grid with the objective of reaching a goal location. The agent can move in four directions: up, down, left, and right. Wind adds a stochastic element by occasionally altering the agent's movement.

### Challenges

- **State Space**: Each state is represented by the agent's position in the grid.
- **Action Space**: Four possible actions (up, down, left, right) at each step.
- **Rewards**: The agent receives a small penalty for each movement until it reaches the goal, where it receives a significant positive reward.

## Implementation Details

### Dynamic Programming

- **Method**: Q-value iteration using the Bellman optimality equation.
- **Goal**: To find the optimal policy by iteratively updating the Q-values for each state-action pair.

### Model-Free Learning

- **Q-learning (Off-policy)**: Learns the optimal policy independently of the agent's actions.
- **SARSA (On-policy)**: Learns the policy based on the actions taken by the agent, incorporating the exploration strategy directly into the policy update.

### Exploration Strategies

- **ε-Greedy**: Offers a balance between exploration (choosing random actions) and exploitation (choosing actions based on known rewards).
- **Boltzmann**: Uses a temperature parameter to vary the exploration rate, providing a probability distribution over actions based on their estimated Q-values.

## Results and Observations

- **Dynamic Programming**: Converges to an optimal policy that navigates efficiently through the environment.
- **Model-Free Learning**: Shows good convergence properties with proper tuning of hyperparameters. The exploration methods impact the speed and stability of learning.
- **Comparison**: Dynamic programming is found to be more stable due to complete knowledge of the environment, whereas model-free methods are more versatile and can adapt to unknown environments.

