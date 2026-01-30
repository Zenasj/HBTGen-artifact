from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from model_based import ModelBasedAgent
import gym
import numpy as np
from types import MethodType
import random
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import datetime
import pickle

sample_period = 100 # How often to look at coverage for plotting
def experiment(env, agent, timesteps, render=False):
    global sample_period
    ob_space_dim = env.observation_space.shape[0]
    state = env.reset()
    state = np.array([state])
    scores =  []
    for t in range(1, timesteps+1):
        # Act
        action, v_pred = agent.act(state)
        # Step
        state_next, _reward, _terminal, _info = env.step(action)
        state_next = np.reshape(state_next, [1, ob_space_dim])
        # Think
        agent.think(state, action, state_next, v_pred)
        # Render
        if render:
            env.render()
        # Gather data
        if t % sample_period == 0:
            score = agent.covered_volume() * 100
            scores.append(score)
    return scores


#  def several_experiments(env, agent, n_experiments=18, timesteps=7000, color='blue'):
def several_experiments(env, agent, n_experiments=3, timesteps=800, color='blue'):
    global sample_period
    scores = np.zeros((n_experiments, int(timesteps/sample_period)))
    for i in range(n_experiments):
        with tf.Session() as sess:
            agent.reset()
            sess.run(tf.global_variables_initializer())
            # Scores
            scores[i, :] = experiment(env, agent, timesteps)
            print("Iter %s, timesteps=%s: coverage = %s" % (i, timesteps, scores[i, -1]))
        tf.reset_default_graph()

if __name__ == '__main__':
    ENV_NAME = "AcrobotForever-v1"
    env = gym.make(ENV_NAME)

    EPSILON = 0.1

    agent = ModelBasedAgent(env)
    agent.exploration_rate = EPSILON
    several_experiments(env, agent)

import random
import gym
import acrobot_forever
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import time
import tensorflow as tf

class ModelBasedAgent:
    def __init__(self, env, act=None):
        from types import MethodType
        if not act is None:
            self.act = MethodType(act, self)

        # Custom initializer
        from keras import backend as K
        def network_init(shape, dtype=None):
            return np.random.random(shape) * 10 - 5

        self.advantage_learning = None
        self.env = env
        self.states = []
        self.exploration_rate = 0.0
        self.n_actions = env.action_space.n
        self.ob_dim = env.observation_space.shape[0]

        # Model network n: (s, a) -> ds
        self.model_net = Sequential()
        self.model_net.add(Dense(12, input_shape=(self.ob_dim + 1,), activation="relu"))
        self.model_net.add(Dense(12, activation="relu"))
        self.model_net.add(Dense(self.ob_dim, activation="linear"))
        self.model_net.compile(loss="mse", optimizer=Adam(lr=0.001))

    def reset(self):
        self.__init__(self.env)

    def _predict_ds(self, state, action):
        sa = np.array([np.concatenate((state[0], [action]))])
        return self.model_net.predict(sa, 1)[0]

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.n_actions), None

        # Sample a desired ds (delta state)
        target_dstate = np.random.normal(0.0, 1.0, self.ob_dim)
        # Make predictions for each possible action
        dstate = [self._predict_ds(state, a) for a in range(self.n_actions)]
        # Find best action by measuring how close ds are to target_ds (angle)
        def dist(ds1, ds2):
            dot = np.dot(ds1, ds2) / (np.linalg.norm(ds1) * np.linalg.norm(ds2))
            return 1 - dot    # from 0 (perfect match) to 2 (anti-parallel)
        ds_distance = [dist(target_dstate, ds) for ds in dstate]

        # Finally, return the action which results in a state change in a direction most similar to desired direction
        return np.argmin(ds_distance), None


    def think(self, state, action, state_next, _):
        self.states.append(state_next[0])

        # Train the model
        sa = np.array([np.concatenate((state[0], [action]))])
        self.model_net.fit(sa, state_next - state, verbose=0)

    # Approximate the covered area by convex hull
    def covered_volume(self):
        states = np.array(self.states)
        states = (states + self.env.observation_space.low)/(self.env.observation_space.high - self.env.observation_space.low)
        return ConvexHull(states).volume