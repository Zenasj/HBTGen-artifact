from tensorflow import keras
from tensorflow.keras import layers

import random
import numpy as np

import tensorflow as tf


class Actor(tf.keras.Model):
    def __init__(self, action_size, fc1_units=400, fc2_units=300):
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=fc1_units, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units=fc2_units, activation=tf.nn.relu)
        self.fc3 = tf.keras.layers.Dense(action_size, activation=tf.nn.tanh)

    def call(self, state):
        return self.fc3(self.fc2(self.fc1(state)))


class Agent():

    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(action_size)

    def act(self, state, add_noise=True):
        action = self.actor_local(state)
        return tf.clip_by_value(action, -1, 1)


agent = Agent(state_size=5, action_size=1, random_seed=2)

@tf.function
def ddpg():
    import numpy as np
    state = np.arange(5).astype(np.float64).reshape((1, -1))
    state = agent.act(state)
    np.isfinite(state)
    return state

scores = ddpg()

#with tf.device("/gpu:0"):
#    scores = ddpg()
#    print("run is successful")