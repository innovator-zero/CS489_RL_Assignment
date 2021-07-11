import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt


class DQN:
    def __init__(self, env):
        self.env = env
        # replay buffer
        self.buffer = deque(maxlen=10000)

        self.discount = 1
        self.epsilon = 1
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.995
        self.update_target = 10

        self.batch_size = 64
        self.train_start = 1000
        self.state_size = self.env.observation_space.shape[0]  # 2
        self.action_size = self.env.action_space.n  # 3

        self.lr = 0.001
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = models.Sequential()
        model.add(layers.Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))

        model.compile(optimizer=optimizers.Adam(lr=self.lr), loss='mean_squared_error')
        return model

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if random.random() <= self.epsilon:
            return self.env.action_space.sample()  # random action

        return np.argmax(self.model.predict(state)[0])

    def train(self):
        if len(self.buffer) < self.train_start:
            return

        batch = random.sample(self.buffer, self.batch_size)

        Input = np.zeros((self.batch_size, self.state_size))
        Target = np.zeros((self.batch_size, self.action_size))

        for i in range(self.batch_size):
            state, action, reward, next_state, done = batch[i]

            # only introduce loss for the action taken
            target = self.model.predict(state)[0]  # shape=[1, 3]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount * np.max(self.target_model.predict(next_state)[0])

            Input[i] = state  # state.shape=[1, 2]
            Target[i] = target

        self.model.fit(Input, Target, batch_size=self.batch_size, verbose=0)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")

    episode = 1000

    dqn = DQN(env=env)
    r = []
    counter = 0

    for epi in range(episode):
        cur_state = env.reset().reshape(1, dqn.state_size)
        score = 0
        done = False
        while not done:
            # env.render()
            action = dqn.act(cur_state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape(1, 2)
            # add to replay buffer
            dqn.store(cur_state, action, reward, next_state, done)
            # train network

            counter += 1
            if counter % 256 == 0:
                dqn.train()

            score += reward
            cur_state = next_state

        if epi % dqn.update_target == 0:
            # update target network
            dqn.target_train()

        print("Episode: {} Reward_sum: {}".format(epi, score))
        r.append(score)

    plt.plot(range(episode), r)
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards")
    plt.show()
