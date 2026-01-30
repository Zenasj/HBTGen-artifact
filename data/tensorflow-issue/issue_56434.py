import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from __future__ import division

from PIL import Image
import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, Visualizer

import matplotlib.pyplot as plt

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

env_name = 'SpaceInvaders-v0'
env = gym.make(env_name)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, LeakyReLU, Input, Dense, Dropout, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, Activation, concatenate, Add

def create_q_model():
    INPUT_SHAPE = (84, 84)
    WINDOW_LENGTH = 4
    inputs = Input(shape=(WINDOW_LENGTH,) + INPUT_SHAPE)
    layer0 = Permute((3,2,1))(inputs)

    layer = Conv2D(32, (8,8), strides=(4,4), activation="relu")(layer0)
    layer = Conv2D(64, (4,4), strides=(2,2), activation="relu")(layer)
    layer = Conv2D(64, (3,3), strides=(1,1), activation="relu")(layer)

    x = Flatten()(layer)
    x = Dense(256)(x)
    x = Dense(nb_actions, activation = 'linear')(x)

    return keras.Model(inputs=inputs, outputs=x)

model = create_q_model()
model.summary()

import cv2

class AtariProcessor(Processor):

    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

from rl.callbacks import Callback


import os
from rl.callbacks import Callback

from tensorflow.keras.optimizers import SGD, RMSprop

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

exploration_steps = 150000
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.,
                              value_min=.05,
                              value_test=.01,
                              nb_steps=exploration_steps)

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               memory=memory,
               processor=processor,
               nb_steps_warmup=5000,
               enable_double_dqn = True,
               enable_dueling_network = True,
               dueling_type = 'avg',
               gamma=.99,
               target_model_update=1e-03,
               delta_clip=1.,
               batch_size=32,
               train_interval=4)

learning_rate = 2.5e-3

dqn.compile(RMSprop(lr=learning_rate), metrics=['mae'])

weights_filename = 'dqn_{}_weights.h5f'.format(env_name)
checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
log_filename = 'dqn_{}_log.json'.format(env_name)
callbacks = [FileLogger(log_filename, interval=10000)]

dqn.fit(env, callbacks=callbacks, nb_steps=100, log_interval=10000, visualize=False, verbose=2) # Intended low nb_steps

dqn.save_weights(weights_filename, overwrite=True)
dqn.save_weights("another", overwrite=True) #This line fails!!