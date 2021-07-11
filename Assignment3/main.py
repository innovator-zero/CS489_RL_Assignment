import numpy as np
import matplotlib.pyplot as plt
import random

size = [4, 12]
actions = [[0, -1], [-1, 0], [1, 0], [0, 1]]  # W, N, S, E
start_state = [3, 0]
goal_state = [3, 11]
runs = 2000
episode = 1000


def get_reward(state, a):
    x = state[0] + actions[a][0]
    y = state[1] + actions[a][1]

    if x < 0 or x >= size[0] or y < 0 or y >= size[1]:  # out of gridworld
        next_state = state  # state unchanged
    else:
        next_state = [x, y]

    if 0 < y < size[1] - 1 and x == 3:  # cliff
        return start_state, -100

    return next_state, -1


def e_greedy(s, q, epsilon):
    if random.random() < epsilon:  # random explore
        return random.randint(0, 3)
    else:
        return greedy(s, q)


def greedy(s, q):
    q_max = q[s[0]][s[1]][0]
    a = 0
    for i in range(1, 4):
        if q[s[0]][s[1]][i] > q_max:
            q_max = q[s[0]][s[1]][i]
            a = i

    return a


def sarsa(epsilon, alpha):
    reward_list = np.zeros(episode)

    q = np.zeros((4, 12, 4))
    for _ in range(runs):
        q = np.zeros((4, 12, 4))

        for i in range(episode):
            s = start_state
            a = e_greedy(s, q, epsilon)
            reward_sum = 0

            while s != goal_state:
                next_s, r = get_reward(s, a)
                reward_sum += r

                next_a = e_greedy(next_s, q, epsilon)  # on-policy
                q[s[0]][s[1]][a] += alpha * (r + q[next_s[0]][next_s[1]][next_a] - q[s[0]][s[1]][a])
                s = next_s
                a = next_a

            reward_list[i] += reward_sum

    reward_list /= runs
    # opt_path(q, "Sarsa")

    return reward_list


def q_learning(epsilon, alpha):
    reward_list = np.zeros(episode)

    q = np.zeros((4, 12, 4))
    for _ in range(runs):
        q = np.zeros((4, 12, 4))

        for i in range(episode):
            s = start_state
            reward_sum = 0

            while s != goal_state:
                a = e_greedy(s, q, epsilon)
                next_s, r = get_reward(s, a)
                reward_sum += r

                max_a = greedy(next_s, q)  # off-policy
                q[s[0]][s[1]][a] += alpha * (r + q[next_s[0]][next_s[1]][max_a] - q[s[0]][s[1]][a])
                s = next_s

            reward_list[i] += reward_sum

    reward_list /= runs
    # opt_path(q, "Q-learning")

    return reward_list


def opt_path(q, title):
    """find the optimal path and visualize it"""
    s = start_state
    path = np.zeros(size, dtype=str)

    while s != goal_state:
        a = greedy(s, q)
        if a == 0:
            path[s[0]][s[1]] = '←'
        elif a == 1:
            path[s[0]][s[1]] = '↑'
        elif a == 2:
            path[s[0]][s[1]] = '↓'
        else:
            path[s[0]][s[1]] = '→'

        next_s, r = get_reward(s, a)
        s = next_s

    path[s[0]][s[1]] = 'G'

    plt.figure(figsize=(6, 2.5))
    plt.table(cellText=path, cellLoc='center', loc='upper left', bbox=[0, 0, 1, 1])
    plt.axis("off")
    plt.title("optimal path from " + title)
    plt.show()


if __name__ == "__main__":
    sarsa_r = sarsa(epsilon=0.5, alpha=0.5)
    qlearn_r = q_learning(epsilon=0.5, alpha=0.5)

    plt.plot(range(len(sarsa_r)), sarsa_r, label="Sarsa")
    plt.plot(range(len(qlearn_r)), qlearn_r, label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.ylim(-2000, 0)
    plt.legend(loc="lower right")
    plt.show()
