import numpy as np
import matplotlib.pyplot as plt
import matplotlib.table as tbl

size = 6
discount = 0
actions = [[0, -1], [-1, 0], [1, 0], [0, 1]]  # W, N, S, E
directions = ['<', '^', 'v', '>']
action_prob = 0.25
terminal_state = [[0, 1], [5, 5]]


def draw_table(T, title):
    """visualize the gridworld"""
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = tbl.table(ax, cellText=T, cellLoc='center', loc='upper left', bbox=[0, 0, 1, 1])
    ax.add_table(tb)
    plt.title(title)
    plt.show()


def get_reward(state, action):
    x = state[0] + action[0]
    y = state[1] + action[1]

    if x < 0 or x >= size or y < 0 or y >= size:  # out of gridworld
        next_state = state  # state unchanged
    else:
        next_state = [x, y]
    reward = -1

    return next_state, reward


def policy_evaluation():
    V = np.zeros((size, size))  # initialized to 0
    it = 0

    while True:
        delta = 0
        old_V = V.copy()

        for i in range(size):
            for j in range(size):
                V[i, j] = 0
                S = [i, j]

                # value of terminal state is 0
                if S not in terminal_state:
                    for a in actions:
                        next_S, r = get_reward(S, a)
                        V[i, j] += action_prob * (r + discount * old_V[next_S[0], next_S[1]])

                    delta = max(delta, abs(V[i, j] - old_V[i, j]))

        it += 1
        if delta < 1e-4:
            break

    draw_table(np.round(V, decimals=2), '$v_{\pi}$ from Policy Evaluation')
    print(it)


def policy_iteration():
    V = np.zeros((size, size))
    policy = np.ones((size, size, 4))  # 1 means action in the policy

    it_pi = 0  # policy iteration counters
    while True:
        # policy evaluation
        it_pe = 0  # policy evaluation counters
        while True:
            delta = 0
            old_V = V.copy()

            for i in range(size):
                for j in range(size):
                    V[i, j] = 0
                    S = [i, j]

                    if S not in terminal_state:
                        indexes = np.argwhere(policy[i, j] == 1)
                        indexes = indexes.flatten().tolist()
                        for index in indexes:  # for actions in policy
                            next_S, r = get_reward(S, actions[index])
                            V[i, j] += r + discount * old_V[next_S[0], next_S[1]]

                        V[i, j] /= len(indexes)  # average value
                        delta = max(delta, abs(V[i, j] - old_V[i, j]))

            it_pe += 1
            if delta < 1e-4 or it_pe >= 7:
                break

        print("policy evaluation loops:" + str(it_pe))
        draw_table(np.round(V, decimals=2), "$v_{\pi %d}$ from Policy Iteration" % it_pi)

        # policy improvement
        policy_stable = True
        for i in range(size):
            for j in range(size):
                old_action = policy[i, j].copy()
                S = [i, j]

                if S not in terminal_state:
                    values = []
                    for a in actions:
                        next_S, r = get_reward(S, a)
                        values.append(r + discount * V[next_S[0], next_S[1]])

                    policy[i, j] = np.zeros(4)
                    indexes = np.argwhere(values == np.max(values))  # maybe more than one action has the max value
                    indexes = indexes.flatten().tolist()

                    for index in indexes:
                        policy[i, j, index] = 1  # every feasible action is set as 1

                    if not (policy[i, j] == old_action).all():
                        policy_stable = False

        it_pi += 1

        # visualize the directions in the policy
        p = np.array(np.zeros((size, size)), dtype=str)
        for i in range(size):
            for j in range(size):
                p[i, j] = ''
                for k in range(4):
                    if policy[i, j, k]:
                        p[i, j] += directions[k]

        for x, y in terminal_state:
            p[x, y] = 'T'

        draw_table(p, "$\pi_{%d}$ from Policy Iteration" % it_pi)

        if policy_stable:
            break

    draw_table(V, "$v_{*}$ from Policy Iteration")
    draw_table(p, "$\pi_{*}$ from Policy Iteration")


def value_iteration():
    V = np.zeros((size, size))  # initialized to 0

    it = 0
    while True:
        delta = 0
        old_V = V.copy()

        for i in range(size):
            for j in range(size):
                V[i, j] = 0
                S = [i, j]

                # value of terminal state is 0
                if S not in terminal_state:
                    values = []
                    for a in actions:
                        next_S, r = get_reward(S, a)
                        values.append(r + discount * old_V[next_S[0], next_S[1]])

                    V[i, j] = max(values)
                    delta = max(delta, abs(V[i, j] - old_V[i, j]))

        it += 1
        if delta < 1e-4:
            break

    print(it)

    # conduct the policy
    p = np.array(np.zeros((size, size)), dtype=str)

    for i in range(size):
        for j in range(size):
            S = [i, j]
            p[i, j] = ''

            if S not in terminal_state:
                values = []
                for a in actions:
                    next_S, r = get_reward(S, a)
                    values.append(r + discount * V[next_S[0], next_S[1]])

                indexes = np.argwhere(values == np.max(values))  # maybe more than one action has the max value
                indexes = indexes.flatten().tolist()

                for index in indexes:
                    p[i, j] += directions[index]
            else:
                p[i, j] = 'T'

    draw_table(p, "$\pi_{*}$ from Value Iteration")

    draw_table(np.round(V, decimals=2), '$v_{\pi}$ from Policy Evaluation')


policy_evaluation()
policy_iteration()
value_iteration()
