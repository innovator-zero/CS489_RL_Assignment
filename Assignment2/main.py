import numpy as np
import matplotlib.pyplot as plt
import matplotlib.table as tbl
import random

size = 6
discount = 1
actions = [[0, -1], [-1, 0], [1, 0], [0, 1]]  # W, N, S, E
terminal_state = [[0, 1], [5, 5]]


def draw_table(T, title):
    """visualize the gridworld"""
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = tbl.table(ax, cellText=T, cellLoc='center', loc='upper left', bbox=[0, 0, 1, 1])
    ax.add_table(tb)
    plt.title(title)
    plt.show()


def get_reward(state):
    """randomly select an action, return next state and reward"""
    rand = random.randint(0, 3)
    action = actions[rand]

    x = state[0] + action[0]
    y = state[1] + action[1]

    if x < 0 or x >= size or y < 0 or y >= size:  # out of gridworld
        next_state = state  # state unchanged
    else:
        next_state = [x, y]
    reward = -1

    return next_state, reward


def get_start():
    """randomly choose a start state for an episode"""
    x = random.randint(0, 5)
    y = random.randint(0, 5)
    return [x, y]


def fv_MC():
    V = np.zeros((size, size))  # initialized to 0
    count = np.zeros((size, size))

    for _ in range(100000):
        visited = []  # record the state is visited at the first time in an episode
        g = []  # record the g when the state is visited
        s = get_start()
        g_sum = 0  # return is the total discounted reward

        # generate an episode
        while s not in terminal_state:
            next_s, r = get_reward(s)
            if s not in visited:
                visited.append(s)
                g.append(g_sum)

            g_sum += r

            s = next_s

        for index in range(len(visited)):
            i, j = visited[index][0], visited[index][1]
            count[i, j] += 1
            V[i, j] = V[i, j] + ((g_sum - g[index]) - V[i, j]) / count[i, j]  # increment update

    draw_table(np.round(V, decimals=2), '$v_{\pi}$ from First Visit Monte-Carlo')


def ev_MC():
    V = np.zeros((size, size))  # initialized to 0
    count = np.zeros((size, size))

    for _ in range(100000):
        visited = []  # record the state is visited in an episode
        g = []  # record the g when the state is visited
        s = get_start()
        g_sum = 0  # return is the total discounted reward

        # generate an episode
        while s not in terminal_state:
            next_s, r = get_reward(s)
            visited.append(s)
            g.append(g_sum)
            g_sum += r

            s = next_s

        for index in range(len(visited)):
            i, j = visited[index][0], visited[index][1]
            count[i, j] += 1
            V[i, j] = V[i, j] + ((g_sum - g[index]) - V[i, j]) / count[i, j]  # increment update

    draw_table(np.round(V, decimals=2), '$v_{\pi}$ from Every Visit Monte-Carlo')


def TD0():
    V = np.zeros((size, size))  # initialized to 0
    alpha = 0.05

    for _ in range(10000):
        s = get_start()

        while s not in terminal_state:
            next_s, r = get_reward(s)
            V[s[0], s[1]] = V[s[0], s[1]] + alpha * (r + discount * V[next_s[0], next_s[1]] - V[s[0], s[1]])

            s = next_s

    draw_table(np.round(V, decimals=2), '$v_{\pi}$ from TD0')


fv_MC()
ev_MC()
TD0()
