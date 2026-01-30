import random
from tensorflow import keras

import gym
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
from tensorflow.keras import layers

env = gym.make('NChain-v0')


def q_learning_keras(env, num_episodes=1000):
    # create the keras model
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(batch_input_shape=(1, 5)))
    model.add(layers.Dense(10, activation='sigmoid'))
    model.add(layers.Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # now execute the q learning
    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
            new_s, r, done, _ = env.step(a)
            target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
            target_vec = model.predict(np.identity(5)[s:s + 1])[0]
            target_vec[a] = target
            model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum / 1000)
    plt.plot(r_avg_list)
    plt.ylabel('Average reward per game')
    plt.xlabel('Number of games')
    plt.show()
    for i in range(5):
        print("State {} - action {}".format(i, model.predict(np.identity(5)[i:i + 1])))


if __name__ == "__main__":
    q_learning_keras(env)