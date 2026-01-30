from tensorflow.keras import layers
from tensorflow.keras import optimizers

import gym
import numpy as np
from tensorflow import keras
from collections import deque
import random
import tensorflow as tf
import gc
import time
print(tf.test.is_gpu_available())
tf.compat.v1.debugging.set_log_device_placement(True)


class Agent:
    def __init__(self, state_size, action_size):
        self.LEARNING_RATE = 0.001
        self.EPSILON = 1.0
        self.EPSILON_DECAY = 0.97
        self.MIN_EPSILON = 0.01
        self.DISCOUNT = 0.95

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(state_size, activation="relu"))
        self.model.add(keras.layers.Dense(24, activation="relu"))
        self.model.add(keras.layers.Dense(48, activation="relu"))
        self.model.add(keras.layers.Dense(action_size, activation="linear"))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
                                                           loss="mse")

        self.memory = deque(maxlen=2000)

    def act(self, state):
        action = 0

        if state is None:
            action = 1
        else:
            if np.random.rand() <= self.EPSILON:
                action = random.randint(0, 2)
            else:
                action = self.model.predict(state)
                keras.backend.clear_session()
                gc.collect()
                action = np.argmax(action)

        return action

    def remember(self, state, reward, action, next_state, done):
        self.memory.append((state, reward, action, next_state, done))

    def retrain(self, minibatch_size):
        if minibatch_size < len(self.memory):
            minibatch = random.sample(self.memory, minibatch_size)
            now = time.time()
            for i in range(100):
                print("Start time:", str(now))
            for state, reward, action, next_state, done in minibatch:
                target = reward

                if not done:
                    target = reward + self.DISCOUNT*np.amax(self.model.predict(next_state)[0])
                    keras.backend.clear_session()
                    gc.collect()

                if not (state is None):
                    target_f = self.model.predict(state)
                    keras.backend.clear_session()
                    gc.collect()
                    target_f[0][action] = target
                    self.model.fit(state, target_f, epochs=1, verbose=0)

            print("Elapsed:", str(time.time() - now))

        if self.EPSILON > self.MIN_EPSILON:
            self.EPSILON *= self.EPSILON_DECAY

    def load(self, name):
        self.model.build(input_shape=(1, 2))
        self.model.load_weights(name)

    def save(self):
        self.model.save_weights(str(episode + 6700) + ".h5")


class Environment:
    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        self.reward = 0
        self.max_height = -999
        self.record_pos = -9

    def make_move(self, action):
        state, reward, done, info = self.env.step(action)

        if state[0] > self.max_height:
            if state[0] > self.record_pos:
                self.record_pos = state[0]
                self.max_height = self.record_pos
                self.reward = 50
            else:
                self.max_height = state[0]
                self.reward = 10

        if state[0] == 0.5:
            self.reward = 100

        return state, reward, done

    def get_state_size(self):
        return self.env.observation_space.shape[0]

    def get_action_size(self):
        return self.env.action_space.n

    def reset(self):
        self.env.reset()
        self.reward = 0
        self.max_height = -999

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def get_max_position(self):
        return self.record_pos


environment = Environment()
agent = Agent(environment.get_state_size(), environment.get_action_size())
agent.load("6700.h5")
for episode in range(1000):
    i = 0
    state = None
    environment.reset()
    done = False
    previous_height = 0
    while not done:
        if episode % 100 == 0:
            environment.render()

        action = agent.act(state)
        next_state, reward, done = environment.make_move(action)

        next_state = np.array(next_state)
        next_state = np.reshape(next_state, [1, environment.get_state_size()])

        if state is not None:
            agent.remember(state, reward, action, next_state, done)

        state = next_state

    print("Max height:", environment.get_max_position(), "episode:", episode)
    if episode % 100 == 0 and episode != 0:
        agent.save()
    agent.retrain(32)

environment.close()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95 #Or any other fraction != 1
session = tf.compat.v1.Session(config=config)