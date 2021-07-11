import torch
from torch import nn
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, 64, bias=True),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(64, action_size, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class DuelingNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64, bias=True),
            nn.ReLU()
        )
        self.adv = nn.Linear(64, action_size, bias=True)
        self.val = nn.Linear(64, 1, bias=True)

    def forward(self, x):
        x = self.fc(x)
        adv = self.adv(x)
        val = self.val(x).expand(adv.size())
        x = val + adv - adv.mean().expand(adv.size())
        return x


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


func = {"DQN": QNet, "Double_DQN": QNet, "Dueling_DQN": DuelingNet}


class Agent:
    def __init__(self, net_name='DQN'):
        self.env = gym.make('MountainCar-v0')
        # replay buffer
        self.memory = deque(maxlen=5000)
        # env parameter
        self.state_size = self.env.observation_space.shape[0]  # 2
        self.action_size = self.env.action_space.n  # 3

        # network
        self.net_name = net_name
        self.model = func[net_name](self.state_size, self.action_size).to(device)
        self.target_model = func[net_name](self.state_size, self.action_size).to(device)
        self.update_target = 10  # update target network every 10 episodes

        # training
        self.batch_size = 64
        self.train_start = 1000
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()

        # DQN parameter
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        set_requires_grad(self.target_model, requires_grad=False)

    def action(self, state):
        # epsilon-greedy selection
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)

        state = torch.Tensor(state).to(device)
        return torch.max(self.model.forward(state), 0)[1].item()

    def greedy_action(self, state):
        # greedy selection
        state = torch.Tensor(state).to(device)
        return torch.max(self.model.forward(state), 0)[1].item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, 0 if done else 1])

    def train(self):
        if len(self.memory) < self.train_start:
            return

        batch = random.sample(self.memory, self.batch_size)

        state = torch.FloatTensor([b[0] for b in batch]).to(device)  # (batch_size, 2)
        action = torch.LongTensor([b[1] for b in batch]).unsqueeze(1).to(device)  # (batch_size, 1)
        reward = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1).to(device)  # (batch_size, 1)
        next_state = torch.FloatTensor([b[3] for b in batch]).to(device)  # (batch_size, 2)
        done = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1).to(device)  # (batch_size, 1)

        if self.net_name == "DQN" or self.net_name == "Dueling_DQN":
            max_q = torch.max(self.target_model.forward(next_state), 1)[0].view(-1, 1)
            target = reward + self.gamma * max_q * done
        else:  # Double DQN
            max_q_idx = torch.max(self.model.forward(next_state), 1)[1].view(-1, 1)  # Q
            target = reward + self.gamma * self.target_model.forward(next_state).gather(1, max_q_idx) * done  # Q'

        loss = self.loss_func(self.model.forward(state).gather(1, action), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_models(self):
        torch.save(self.model.state_dict(), 'models/' + self.net_name)
        torch.save(self.target_model.state_dict(), 'models/' + self.net_name + "_target")

    def load_models(self):
        self.model.load_state_dict(torch.load('models/' + self.net_name))
        self.target_model.load_state_dict(torch.load('models/' + self.net_name + "_target"))


def train_model(episodes, net_name="DQN"):
    if net_name == "Noisy_DQN":
        agent = Agent(net_name=net_name)
    else:
        agent = Agent(net_name=net_name)

    # agent.load_models()
    reward_list = []
    step = 0

    for epi in tqdm(range(1, episodes + 1), ascii=True, unit='episode'):
        state = agent.env.reset()
        sum_reward = 0
        my_reward = 0

        # update target network
        if epi % agent.update_target == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        while True:
            step += 1
            action = agent.action(state)
            next_state, reward, done, info = agent.env.step(action)
            sum_reward += reward  # real reward from env

            reward = 100 * ((math.sin(3 * next_state[0]) * 0.0025 + 0.5 * next_state[1] * next_state[1]) - (
                    math.sin(3 * state[0]) * 0.0025 + 0.5 * state[1] * state[1]))

            my_reward += reward
            agent.store(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            if done:
                break

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon_min, agent.epsilon)

        print('sum_reward: {} , my_reward: {}'.format(round(sum_reward, 3), round(my_reward, 3)))
        reward_list.append(sum_reward)

    agent.save_models()

    return reward_list


def train_all(name_list):
    reward_lists = []

    for name in name_list:
        reward_list = train_model(episodes=2000, net_name=name)
        reward_lists.append(reward_list)

    return reward_lists


def draw_compare(reward_lists, name_list, name=None):
    for i, reward_list in enumerate(reward_lists):
        r = []
        for idx in range(len(reward_list) // 50):
            r.append(np.mean(reward_list[idx * 50:(idx + 1) * 50]))

        plt.plot([i * 50 for i in range(len(r))], r, label=name_list[i])

    plt.xlabel("Episodes")
    plt.ylabel("Sum of Rewards")
    plt.grid()
    plt.legend()

    if name is not None:
        plt.savefig(name + ".png")
    plt.show()


n_list = ["DQN", "Double_DQN", "Dueling_DQN"]
r_lists = train_all(n_list)
draw_compare(r_lists, n_list, name="train")


# ====test====
def demo(net_name, render=False):
    sum_reward = 0.0
    agent = Agent(net_name=net_name)
    agent.epsilon = 0

    agent.load_models()
    state = agent.env.reset()

    while True:
        if render:
            agent.env.render()
        action = agent.action(state)
        next_state, reward, done, info = agent.env.step(action)

        sum_reward += reward
        state = next_state

        if done:
            agent.env.close()
            break

    return sum_reward


def test(test_num, net_name):
    reward_list = []
    for _ in range(test_num):
        reward_list.append(demo(net_name=net_name, render=False))
    return reward_list


def test_all(name_list, test_num):
    reward_lists = []
    for name in name_list:
        reward_lists.append(np.array(test(test_num=test_num, net_name=name)))
    return reward_lists


r_lists_test = test_all(n_list, test_num=1000)
draw_compare(r_lists_test, n_list, name="test")
