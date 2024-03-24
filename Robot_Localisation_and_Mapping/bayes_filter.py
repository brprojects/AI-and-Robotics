import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):

    plt.figure()

    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")

    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")
    plt.show()
    # plt.savefig(('../images/bayes_filter.png'))

# When the robot moves it has: 
    # 70% chance of going in the direction they meant to
    # 20% chance of not moving
    # 10% chance of moving in the opposite direction
def motion_model(action, belief):
    # Convolution
    action_dict = {'F': [0.1,0.2,0.7], 'B': [0.7,0.2,0.1]}
    new_belief = np.zeros(15)
    for i in range(len(belief)):
        if i == 0:
            new_belief[i] += belief[i] * round(sum(action_dict[action][0:2]),5)
            new_belief[i+1] += belief[i] * action_dict[action][2]
        elif i == 14:
            new_belief[i] += belief[i] * round(sum(action_dict[action][1:]),5)
            new_belief[i-1] += belief[i] * action_dict[action][0]
        else:
            new_belief[i-1] += belief[i] * action_dict[action][0]
            new_belief[i] += belief[i] * action_dict[action][1]
            new_belief[i+1] += belief[i] * action_dict[action][2]
    belief = new_belief
    return belief

# The sensor is able to recognize that:
    # a tile is white with 70% probability 
    # a tile is **black** with 90% probability
def sensor_model(observation, belief, world):
    black_sum = round(sum([belief[i] for i in range(len(belief)) if world[i] == 0]),5)
    white_sum = round(sum([belief[i] for i in range(len(belief)) if world[i] == 1]),5)
    for i in range(len(belief)):
        if observation == 0 and world[i]==0:
            belief[i] = 0.9 * belief[i]/black_sum
        elif observation == 0 and world[i]==1:
            belief[i] = 0.1 * belief[i]/white_sum
        elif observation == 1 and world[i]==1:
            belief[i] = 0.7 * belief[i]/black_sum
        elif observation == 0 and world[i]==0:
            belief[i] = 0.3 * belief[i]/white_sum
    return belief

# Initial known position
belief = np.zeros(15)
x_start = 7
belief[x_start] = 1.0

# Unknown initial position
belief = [0.0666666667 for i in range(15)]

# World is a 1D line of black (0) or white (1) tiles
world = [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]

# Observed tile colours when executing commands
observations = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
commands = 'FFFFBBFFB'
for i in range(len(commands)):
    belief = motion_model(commands[i], belief)
    belief = sensor_model(observations[i], belief, world)

print('position {0} with probability {1}'.format(np.argmax(belief), max(belief)))
plot_belief(belief)
