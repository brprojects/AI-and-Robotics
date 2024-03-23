
import gym
import numpy as np
import matplotlib.pyplot as plt


EPISODES = 25_000
SHOW_EVERY = 500
epsilon = 0  # exploration settings
LEARNING_RATE = 0.4
DISCOUNT = 1

def sample(env):
    state = env.reset()[0]
    action = env.action_space.sample()
    return action, state[0], state[1], state[2]

def init_q_table():
    q_table = np.ones((22,11,2))
    q_table[17:,:,0] = 0
    q_table[13:17,2:7,0] = 0
    q_table[12,4:7,0] = 0
    q_table[19:,:,1] = 0
    q_table[18,2:9,1] = 0
    return q_table

def get_action(q_table, ts):
    action = q_table[ts[0], ts[1], int(ts[2])]
    return action

def nextmove(q_table, ts, env, counter):

    ts_old = ts

    tmp = False

    if np.random.random() < epsilon:
        tmp = True
        tmp1 = np.random.uniform(low = -1, high = 1) # stand
        tmp2 = np.random.uniform(low = -1, high = 1) # hit

    if tmp:
        if tmp1 < tmp2:
            action = 1
            n_state, reward, done, thing, info = env.step(action)
            ts = [n_state[0], n_state[1], n_state[2], action, reward]
            if ts[0] > 21:
                win_lose(reward, counter)
            else:
                nextmove(q_table, ts, env, counter)
        else:
            action = 0
            n_state, reward, done, thing, info = env.step(action)
            ts = [n_state[0], n_state[1], n_state[2], action, reward]
            win_lose(reward, counter)
    else:
        action = get_action(q_table, ts)
        if action == 1:
            action = 1
            n_state, reward, done, thing, info = env.step(action)
            ts = [n_state[0], n_state[1], n_state[2], action, reward]
            if ts[0] > 21:
                win_lose(reward, counter)
            else:
                nextmove(q_table, ts, env, counter)
        else:
            action = 0
            n_state, reward, done, thing, info = env.step(action)
            ts = [n_state[0], n_state[1], n_state[2], action, reward]
            win_lose(reward, counter)


def win_lose(reward, counter):
    if reward == 1:
        # print('Result : Win')
        counter['Wins'] += 1
    elif reward == 0:
        # print('Result : Draw')
        counter['Draws'] += 1
    else:
        # print('Result : Lose')
        counter['Losses'] += 1

def many_runs(env, runs, q_table):

    counter = {'Wins':0, 'Draws':0, 'Losses':0}

    for i in range(runs):
        sample1 = sample(env)
        ts = []
        ts.append(sample1[1])
        ts.append(sample1[2])
        ts.append(sample1[3])
        ts.append(2)
        ts.append(2)
        # print('Player total : {0}   Dealer card : {1}   Ace : {2}'.format(ts[0], ts[1], ts[2]))
        nextmove(q_table, ts, env, counter)

    return counter


def main():

    environment_name = 'Blackjack-v1'
    env = gym.make(environment_name, natural=False, sab=False)

    percentage_success = []
    episode = []
    q_table = init_q_table()

    for i in range(int(EPISODES/SHOW_EVERY)):
        counter = many_runs(env, SHOW_EVERY, q_table)
        global epsilon, LEARNING_RATE
        print('Episodes = {0} with {1} and epsilon = {2} learning rate = {3}'.format((i+1)*SHOW_EVERY,counter, epsilon, LEARNING_RATE))
        epsilon = epsilon * 0.85
        LEARNING_RATE = LEARNING_RATE * 0.95
        percentage_success.append(counter['Wins']/(counter['Wins']+counter['Losses']))
        episode.append((i+1)*SHOW_EVERY)

    print(percentage_success)
    print(episode)


if __name__ == '__main__':
    main()
