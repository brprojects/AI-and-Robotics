# AI and Robotics Projects

This repository contains a series of projects centered around Artificial Intelligence (AI) and Robotics. From Reinforcement Learning strategies to path planning algorithms and robot localization and mapping examples, these projects showcase various applications and implementations in the field.

## Table of Contents
- [Reinforcement Learning for Optimal Blackjack Strategy](#reinforcement-learning-for-optimal-blackjack-strategy)
- [Path Planning (Dijkstra's, A* and Potential Field)](#path-planning)
- [Localisation and Mapping (Bayes Filter and Occupancy Mapping)](#localisation-and-mapping)
- [Distribution Sampling (Rejection and Metropolis-Hastings)](#distribution-sampling)

## Reinforcement Learning for Optimal Blackjack Strategy

### Overview

This project aims to implement a reinforcement learning (RL) algorithm to learn the optimal strategy for playing blackjack. Blackjack is a popular card game where players aim to beat the dealer by having a hand value closer to 21 without exceeding it.

The project utilizes RL techniques, specifically Q-learning, to train an agent to make decisions in the game of blackjack. The agent learns from experience through interactions with the environment, gradually improving its strategy over time.

### Requirements

- Python 3.11
- NumPy
- Matplotlib (for visualization)
- OpenAI Gym (for the blackjack environment)


### Project Structure

- `blackjack.py`: Main script for training the RL agent using Q-learning.
- `blackjack_optimal_strategy.py`: Script to get performance of optimal blackjack strategy.
- `performance.py`: Script to compare the RL agents performance over training to optimal strategy.
- `learnt_grid.py`: Creates visualisation of learnt strategy.
- `optimal_strategy_grid.py`: Creates visualisation of optimal strategy.


### Results

The Q-table values are updated using the Q-learning formula:
`Q(s, a) = (1 - α) * Q(s, a) + α * (r + γ * maxₐ' Q(s', a'))`
- Q(s, a) - the Q-value of state s and action a.
- α - the learning rate (0 < α ≤ 1), controlling the weight given to new information compared to past Q-values.
- r - the immediate reward obtained after taking action a in state s.
- γ - the discount factor (0 ≤ γ < 1), determining the importance of future rewards.
- s' - the next state after taking action a in state s.
- maxₐ' Q(s', a') - the maximum Q-value of all possible actions in state s', estimating the future cumulative reward.


The trained agent's performance is evaluated against the optimal strategy using its win percentage. The visualisation also shows agent's learning progress over episodes. As can be seen below, the agent's strategy becomes comparable to optimal after around 20,000 episodes. A learning rate is initially 0.4 and the exploration rate is initially 0.9, reduced by a factor of 0.95 and 0.85 respectively every 500 episodes.


![Learning Progress](./images/blackjack_model_improvement.png)

The agent's strategy can be visualised as a grid with dealer card and player total as the axis, for the situation when the player has and doesn't have a usable ace. The optimal strategy is shown on the left and the agent's learnt strategy is on the right.  By the end of the training,
the agent’s policy looks similar to this optimal strategy, differing mainly on fringe cases which would not have a major impact on the agent’s overall win rate. For example, the agent will choose to hit if the player total is 12 and the dealer is showing a 4, whilst the optimal strategy would suggest standing. The expected value of hitting in this scenario is −0.211161 whilst the expected value of standing is −0.211100. Since the expected values are so close the agent would need many more episodes and a much smaller learning rate before it would correctly decide to stand.

| Optimal Strategy | Learnt Strategy |
|--------------|-------------|
| ![Optimal Grid](./images/optimal_grid.png) | ![Learnt Grid](./images/learnt_grid.png) |


## Path Planning

### Overview
Path planning algorithms are used to find the shortest path between a start point and a goal point in a given environment. In this repository, we explore three common path planning algorithms: Dijkstra's, A* and Potential Field. These algorithms are implemented in Python and applied to solve a maze.

### Project Structure
- `create_maze.py`: script creates a maze represented as a numpy array, where 1s are walls and 0s are free space from an imported image (maze.png) of a maze.
- `Dijkstra.py`: implements Dijkstra's algorithm to solve the maze.
- `A_star.py`: implements A* algorithm to solve the maze.
- `potential_field.py`: implements the potential field algorithm to travel between start and end points with randomly generated obstacles in the way.

### Results

#### Dijkstra's Algorithm
Dijkstra's algorithm is a classic pathfinding algorithm that finds the shortest path from a start node to all other nodes in the graph. It explores nodes in increasing order of their distance from the start node, always selecting the node with the smallest known distance from the start node to expand next. Dijkstra's algorithm is complete and optimal for finding the shortest path.

The image below shows Dijkstra's solution to the maze with:
- Black pixels - walls
- White pixels - unexplored empty space
- Light gray pixels - explored empty space
- Dark gray pixels - optimal route found

![Dijkstra](./images/Dijkstra.png)

#### A* Algorithm
A* (pronounced "A star") is an informed search algorithm that uses both the actual cost of reaching a node from the start node (g-value) and an estimate of the cost from the node to the goal (h-value). It selects the next node to explore based on the sum of these two values (f-value), prioritizing nodes that are likely to lead to the goal. A* is complete and optimal when a consistent heuristic is used.

The images below show that the optimal route was found quicker when a large heursitic was used, so increasing the h-value compared to the g-value.

| Heuristic = 3 | Heuristic = 100 |
|--------------|-------------|
| ![A star](./images/A_star.png) | ![A star2](./images/A_star2.png) |

#### Potential Field Algorithm
The Potential Field algorithm is a reactive approach to path planning, commonly used in robotics. It models the environment as a potential field where attractive forces guide the robot towards the goal and repulsive forces avoid obstacles. The robot moves by following the gradient of the potential field towards the goal while avoiding obstacles.

A weakness of the potential field algorithm is that the robot can get stuck in a local minima. To overcome this, as seen in the right image below, we temporarily move the end goal to a new location (e.g. bottom right corner) to get the robot out of the local minima and back on track to finding the global minima (i.e. the finish).



| Not Getting Stuck | Getting stuck |
|--------------|-------------|
| ![Potential field](./images/potential_field.png) | ![Potential field stuck](./images/potential_field2.png) |


## Localisation and Mapping

### Overview
This project contains code for implementing localization and mapping algorithms for robotics applications. In this repository, we explore creating a Bayes Filter with a motion and sensor model to estimate robot location and create an occupancy grid from laser measurements and robot poses. 

### Project Structure
- `bayes_filter.py`: script for 1D Bayesian Filter to estimate location of a robot.
- `grid_mapping.py`: script to create occupancy mapping from poses and ranges data.
- `bresenham.py`: implementation of Bresenham's line drawing algorithm.

### Results

#### Bayes Filter
The Bayes filter is a probabilistic algorithm used for state estimation in robotics and other dynamic systems. It combines information from sensor measurements and control inputs to estimate the probability distribution over the state of the system. It involves a motion and sensor model: 
- The motion model describes the uncertainty in the motion of the system, such as velocity or acceleration, and predicts the future state of the system based on the current state and control inputs.
- The sensor model describes the uncertainty in sensor measurements and predicts the likelihood of observing a particular measurement given the true state of the system.

The results below show the probability of the robot being in each state in a 1D world of black and white tiles. The robot can sense the colour of the tile it is currently on, but has imperfect motion and sensors. The figure on the left is for a known starting point, whilst the figure on the right is for when the starting point is unknown.

| Known Starting Point | Unknown Starting Point |
|--------------|-------------|
| ![Bayes](./images/bayes_filter.png) | ![Bayes2](./images/bayes_filter2.png) |

#### Occupancy Mapping

Occupancy grid mapping is a process of building a map of the environment based on sensor measurements and robot poses. In this script, I use laser scan ranges to update an occupancy grid map, which represents the likelihood of each grid cell being occupied by an obstacle.

Using log-odds instead of probabilities is crucial in occupancy grid mapping due to its benefits in numerical stability, efficient representation, and it facilitates additive operations.

The results shown below show the pose and range measurements represent the floor plan of a building. The lines extending past the walls of the building are due to the laser scanner passing through a window.

![Occupancy mapping](./images/occupancy_map.png)


## Distribution Sampling

### Overview

This repository contains Python implementations of the Metropolis-Hastings and rejection sampling algorithms. Metropolis-Hastings is a Markov chain Monte Carlo (MCMC) method commonly used for sampling from complex probability distributions. Rejection sampling is a simple and widely used method for generating samples from a target probability distribution

### Project Structure
- `metropolis_hastings_sampling.py`: Implementation of the Metropolis-Hastings algorithm.
- `rejection_sampling.py`: Implementation of the rejection sampling algorithm.

### Results

Target distribution: `0.5×Norm(x;μ1=20,σ1=3)+0.5×Norm(x;μ2=40,σ2=10)`  
As shown below, both rejection and Metropolis-Hastings are able to effectively sample from the underlying distribution. However, in 20,000 samples, rejection sampling only obtains 3629 valid samples, an acceptance rate of only 18%. Metropolis-Hastings algorithm guarantees a valid sample for every attempted sample.

| Rejection Sampling | Metropolis-Hastings Sampling |
|--------------|-------------|
| ![rejection](./images/rejection_sampling.png) | ![MH](./images/metropolis_hastings_sampling.png) |