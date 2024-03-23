## Reinforcement Learning for Optimal Blackjack Strategy

### Overview

This project aims to implement a reinforcement learning (RL) algorithm to learn the optimal strategy for playing blackjack. Blackjack, also known as 21, is a popular card game where players aim to beat the dealer by having a hand value closer to 21 without exceeding it.

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

The trained agent's performance is evaluated against the optimal strategy using its win percentage. The visualisation also shows agent's learning progress over episodes. As can be seen below, the agent's strategy becomes comparable to optimal after around 20,000 episodes. A learning rate is initially 0.4 and the exploration rate is initially 0.9, reduced by a factor of 0.95 and 0.85 respectively every 500 episodes.


![Learning Progress](./images/blackjack_model_improvement.png)

The agent's strategy can be visualised as a grid with dealer card and player total as the axis, for the situation when the player has and doesn't have a usable ace. The optimal strategy is shown on the left and the agent's learnt strategy is on the right.  By the end of the training,
the agent’s policy looks similar to this optimal strategy, differing mainly on fringe cases which would not have a major impact on the agent’s overall win rate. For example, the agent will choose to hit if the player total is 12 and the dealer is showing a 4, whilst the optimal strategy would suggest standing. The expected value of hitting in this scenario is −0.211161 whilst the expected value of standing is −0.211100. Since the expected values are so close the agent would need many more episodes and a much smaller learning rate before it would correctly decide to stand.

<div style="display:flex;">
  <div style="flex:1; padding-right:5px;">
    <img src="./images/optimal_grid.png" alt="Optimal Grid" width="400"/>
  </div>
  <div style="flex:1; padding-left:5px;">
    <img src="./images/learnt_grid.png" alt="Learnt Grid" width="400"/>
  </div>
</div>