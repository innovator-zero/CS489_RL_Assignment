import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import math
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v0')
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.shape[0]

MAX_EPI = 5000
MAX_STEP = 200
UPDATE_EVERY = 5


def init_layer(layer):
    nn.init.normal_(layer.weight, mean=0.0, std=0.1)
    nn.init.constant_(layer.bias, 0.0)


def record(name, global_epi, res_queue, ret):
    with global_epi.get_lock():
        global_epi.value += 1
    res_queue.put(ret)
    print('Name: %s Episode: %d, Reward Sum: %.1f' % (name, global_epi.value, ret))


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.95, 0.999), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # state initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ACNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(ACNet, self).__init__()
        # shared network
        self.fc = nn.Linear(state_size, 256)
        # net for policy (actor)
        self.mu = nn.Linear(256, action_size)
        self.sigma = nn.Linear(256, action_size)

        # net for V (critic)
        self.value = nn.Linear(256, 1)

        for l in [self.fc, self.mu, self.sigma, self.value]:
            init_layer(l)

        self.distribution = torch.distributions.Normal

    def forward(self, x):
        fc = F.relu6(self.fc(x))
        mu = 2 * torch.tanh(self.mu(fc))
        sigma = F.softplus(self.sigma(fc)) + 0.001
        v = self.value(fc)
        return mu, sigma, v

    def loss(self, state, action, R):
        self.train()
        mu, sigma, value = self.forward(state)
        TD_error = R - value
        critic_loss = TD_error.pow(2)

        dist = self.distribution(mu, sigma)
        log_prob = dist.log_prob(action)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(dist.scale)  # entropy of policy
        actor_loss = -(log_prob * TD_error.detach() + 0.005 * entropy)
        total_loss = (critic_loss + actor_loss).mean()
        return total_loss

    def select_action(self, state):
        # select action from normal distribution
        self.eval()
        mu, sigma, _ = self.forward(state)
        dist = self.distribution(mu.view(1, ).detach(), sigma.view(1, ).detach())
        a = dist.sample().numpy()
        return a


class Worker(mp.Process):
    def __init__(self, name, g_epi, opt, g_net, res_queue):
        super(Worker, self).__init__()
        self.name = 'worker%i' % name
        self.global_epi = g_epi
        self.opt = opt
        self.local_net = ACNet(STATE_SIZE, ACTION_SIZE)
        self.global_net = g_net
        self.res_queue = res_queue
        self.gamma = 0.9
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        step = 1
        while self.global_epi.value < MAX_EPI:
            state = self.env.reset()
            state_buffer, action_buffer, reward_buffer = [], [], []
            ret = 0

            for t in range(MAX_STEP):
                action = self.local_net.select_action(torch.FloatTensor(state.reshape(1, -1)))  # shape:[1, 1]
                next_state, reward, done, _ = self.env.step(action.clip(-2, 2))  # range of action:[-2, 2]

                if t == MAX_STEP - 1:
                    done = True

                ret += reward
                state_buffer.append(state.reshape(1, -1))  # shape: [3, ]->[1, 3]
                action_buffer.append(action.reshape(1))  # shape: [1, 1]->[1, ]
                reward_buffer.append((reward + 8.1) / 8.1)

                if step % UPDATE_EVERY == 0 or done:
                    self.train(done, next_state, state_buffer, action_buffer, reward_buffer)
                    state_buffer, action_buffer, reward_buffer = [], [], []

                    if done:
                        record(self.name, self.global_epi, self.res_queue, ret)
                        break

                state = next_state
                step += 1

        self.res_queue.put(None)

    def train(self, done, next_s, s_buf, a_buf, r_buf):
        if done:
            R = 0
        else:
            R = self.local_net.forward(torch.FloatTensor(next_s.reshape(1, -1)))[-1].detach().item()  # v(s)

        R_buffer = []
        for r in r_buf[::-1]:  # reverse buffer, t-1 to t_start
            R = r + self.gamma * R
            R_buffer.append(R)
        R_buffer.reverse()  # reverse again

        loss = self.local_net.loss(torch.FloatTensor(np.vstack(s_buf)), torch.FloatTensor(np.array(a_buf)),
                                   torch.FloatTensor(np.array(R_buffer).reshape(-1, 1)))
        # s_buf shape: n * [1, 3] -> [n, 3]
        # a_buf shape: n * [1, ] -> [n, 1]
        # R_buf shape: n * 1 -> [n, 1]
        self.opt.zero_grad()
        loss.backward()

        # use local net to update global net
        for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        self.opt.step()

        # copy global parameters
        self.local_net.load_state_dict(self.global_net.state_dict())


class A3C():
    def __init__(self, w_nums):
        self.global_net = ACNet(STATE_SIZE, ACTION_SIZE)
        self.global_net.share_memory()
        self.opt = SharedAdam(self.global_net.parameters(), lr=1e-4, betas=(0.95, 0.999))
        self.global_epi = mp.Value('i', 0)  # shared variable
        self.res_queue = mp.Queue()
        self.worker_nums = w_nums

    def start(self):
        workers = [Worker(i, self.global_epi, self.opt, self.global_net, self.res_queue) for i in
                   range(self.worker_nums)]
        for w in workers:
            w.start()

        res = []
        while True:
            r = self.res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break  # reach max episodes

        for w in workers:
            w.join()

        torch.save(self.global_net.state_dict(), 'model3.pth')

        return res


def draw(res):
    r = []
    stride = 50
    for i in range(len(res) // stride):
        r.append(np.mean(res[i * stride:(i + 1) * stride]))

    plt.plot([i * stride for i in range(len(r))], r)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards')
    plt.grid()

    plt.savefig('5.png')
    plt.show()


def test():
    net = ACNet(STATE_SIZE, ACTION_SIZE)
    net.load_state_dict(torch.load('model3.pth'))
    state = env.reset()
    while True:
        env.render()
        action = net.select_action(torch.FloatTensor(state.reshape(1, -1)))
        next_state, reward, done, _ = env.step(action.clip(-2, 2))
        state = next_state


if __name__ == '__main__':
    flag = 0
    if flag:
        a3c = A3C(8)
        res = a3c.start()
        draw(res)
    else:
        test()
