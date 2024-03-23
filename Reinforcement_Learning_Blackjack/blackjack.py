import gym
import numpy as np
import matplotlib.pyplot as plt


EPISODES = 25_000
SHOW_EVERY = 500
epsilon = 0.9   # exploration settings
LEARNING_RATE = 0.4
DISCOUNT = 1

def sample(env):
    state = env.reset()[0]
    action = env.action_space.sample()
    return action, state[0], state[1], state[2]

def init_q_table():
    actions = ('stand', 'hit')
    q_table = np.random.rand(len(actions), 22, 12, 2) - 0.5
    q_table = np.round(q_table, decimals=3)
    return q_table

def get_q_value(q_table, ts):
    stand = q_table[0, ts[0], ts[1], int(ts[2])]
    hit = q_table[1, ts[0], ts[1], int(ts[2])]
    return stand, hit

def update_q_table(q_table, ts_old, ts):
    reward = ts[4]
    action = ts[3]
    stand, hit = get_q_value(q_table, ts_old)
    q_value = [stand, hit][action]
    if ts[0] < 22 and action == 1:
        next_stand, next_hit = get_q_value(q_table, ts)
        max_future_q = max(next_stand, next_hit)
    else:
        max_future_q = 0

    new_q = (1 - LEARNING_RATE) * q_value + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    new_q = np.round(min(1, max(new_q, -1)), decimals=3)
    q_table[action, ts_old[0], ts_old[1], int(ts_old[2])] = new_q
    # print('Player total : {0}   Dealer card : {1}   Ace : {2}  Action : {3}'.format(ts[0], ts[1], ts[2], ts[3]))
    return q_table

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
            q_table = update_q_table(q_table, ts_old, ts)
            if ts[0] > 21:
                win_lose(reward, counter)
            else:
                nextmove(q_table, ts, env, counter)
        else:
            action = 0
            n_state, reward, done, thing, info = env.step(action)
            ts = [n_state[0], n_state[1], n_state[2], action, reward]
            win_lose(reward, counter)
            q_table = update_q_table(q_table, ts_old, ts)
    else:
        stand, hit = get_q_value(q_table, ts)
        if stand < hit:
            action = 1
            n_state, reward, done, thing, info = env.step(action)
            ts = [n_state[0], n_state[1], n_state[2], action, reward]
            q_table = update_q_table(q_table, ts_old, ts)
            if ts[0] > 21:
                win_lose(reward, counter)
            else:
                nextmove(q_table, ts, env, counter)
        else:
            action = 0
            n_state, reward, done, thing, info = env.step(action)
            ts = [n_state[0], n_state[1], n_state[2], action, reward]
            win_lose(reward, counter)
            q_table = update_q_table(q_table, ts_old, ts)


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

def read_table_rules(q_table):
    table = np.zeros((36,10))

    for i in range(18):
        for j in range(10):
            stand_no_ace = q_table[0,i,j,0]
            hit_no_ace = q_table[1,i,j,0]
            if stand_no_ace + 0.2 < hit_no_ace:
                table[i,j] = 1
            elif stand_no_ace < hit_no_ace:
                table[i,j] = 0.5
            elif stand_no_ace > hit_no_ace + 0.2:
                table[i,j] = -1
            else:
                table[i,j] = -0.5
    for i in range(18):
        for j in range(10):
            stand_ace = q_table[0,i,j,1]
            hit_ace = q_table[1,i,j,1]
            if stand_ace + 0.2 < hit_ace:
                table[i+18,j] = 1
            elif stand_ace < hit_ace:
                table[i+18,j] = 0.5
            elif stand_ace > hit_ace + 0.2:
                table[i+18,j] = -1
            else:
                table[i+18,j] = -0.5
    # print(table)

    # np.save('learnt_strategy.npy', table)

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

    # print(percentage_success)

    # read_table_rules(q_table)


if __name__ == '__main__':
    main()
