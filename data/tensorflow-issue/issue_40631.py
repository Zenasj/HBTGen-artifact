import math
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import tqdm
import yfinance as yf


# tf.compat.v1.disable_eager_execution()
ticker = "AAPL"
df_full = yf.Ticker("{}".format(ticker)).history("max").reset_index()
df = df_full.copy()["Close"]
data = df.copy()


class DQN:
    def __init__(self, data, lookBack=30,
                 gamma=0.95, epsilon=0.5,
                 epsilonMin=0.01, epsilonDecay=0.8,
                 learningRate=0.001, money=10000):
        # NOT IMPORTANT
        self.lookBack = lookBack
        self.initialMoney = money
        self.actionSize = 3

        self.data = data
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate

        self.memory = []

    def buildModel(self):
        keras.backend.clear_session()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(256, input_shape=[self.lookBack],
                                     activation="relu"))
        model.add(keras.layers.Dense(self.actionSize))

        self.optimizer = keras.optimizers.RMSprop(lr=self.learningRate,
                                                  epsilon=0.1,
                                                  rho=0.99)
        self.lossFunc = keras.losses.mean_squared_error
        model.compile(loss="mse", optimizer=self.optimizer)
        self.model = model

    def getAction(self, state):
        # NOT IMPORTANT
        if random.random() <= self.epsilon:
            return random.randrange(self.actionSize)
        else:
            return np.argmax(self.model.predict(state)[0])

    def createDataset(self):
        # NOT IMPORTANT
        tmp = self.data.copy()
        tmp = tmp.diff(1).dropna().values
        shape = tmp.shape[:-1] + (tmp.shape[-1] - self.lookBack + 1,
                                  self.lookBack)
        strides = tmp.strides + (tmp.strides[-1],)
        self.dataset = np.lib.stride_tricks.as_strided(tmp, shape=shape,
                                                       strides=strides)

    def getReward(self, action, currentPrice):
        # NOT IMPORTANT
        return 0

    # @tf.function
    def updateWeights(self):
        # IMPORTANT
        tf.print(len(self.memory))
        if len(self.memory) >= self.batchSize:
            endIndex = len(self.memory)
            startIndex = endIndex - self.batchSize
            batchData = []
            for i in range(startIndex, endIndex):
                batchData.append(self.memory[i])
            X = np.zeros((self.batchSize, self.lookBack))
            Y = np.zeros((self.batchSize, self.actionSize))
            states = np.array([item[0] for item in batchData])
            newStates = np.array([item[3] for item in batchData])
            Q = self.model(states)
            QNext = self.model(newStates)
            for i in range(len(batchData)):
                state, action, reward, nextState = batchData[i]
                target = Q[i]
                target[action] = reward
                target[action] += self.gamma * np.max(QNext[i])

                X[i] = state
                Y[i] = target
            self.model.train_on_batch(X, Y)
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay

    def train(self, epochs=200, logFreq=1):
        # IMPORTANT
        for epoch in range(epochs):
            self.profit = 0
            self.money = self.initialMoney
            for timeStep in tqdm.tqdm(range(self.lookBack, len(self.data)-1)):
                currentPrice = data[timeStep]
                currentState = self.dataset[timeStep-self.lookBack]
                nextState = self.dataset[timeStep-self.lookBack+1]

                action = self.getAction(currentState.reshape(1, -1))

                reward = self.getReward(action, currentPrice)

                self.memory.append((currentState, action, reward, nextState))

                self.updateWeights()


test = DQN(data)
test.createDataset()
test.buildModel()
test.train(100)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import tqdm
import yfinance as yf


tf.compat.v1.disable_eager_execution()
ticker = "AAPL"
df_full = yf.Ticker("{}".format(ticker)).history("max").reset_index()
df = df_full.copy()["Close"]
data = df.copy()


class DQN:
    def __init__(self, data, lookBack=30,
                 gamma=0.95, epsilon=0.5,
                 epsilonMin=0.01, epsilonDecay=0.8,
                 learningRate=0.001, money=10000):
        # NOT IMPORTANT
        self.lookBack = lookBack
        self.initialMoney = money
        self.actionSize = 3
        self.batchSize = 32

        self.data = data
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate

        self.memory = []

    def buildModel(self):
        keras.backend.clear_session()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(256, input_shape=[self.lookBack],
                                     activation="relu"))
        model.add(keras.layers.Dense(self.actionSize))

        self.optimizer = keras.optimizers.RMSprop(lr=self.learningRate,
                                                  epsilon=0.1,
                                                  rho=0.99)
        self.lossFunc = keras.losses.mean_squared_error
        model.compile(loss="mse", optimizer=self.optimizer)
        self.model = model

    def getAction(self, state):
        # NOT IMPORTANT
        if random.random() <= self.epsilon:
            return random.randrange(self.actionSize)
        else:
            return np.argmax(self.model.predict(state)[0])

    def createDataset(self):
        # NOT IMPORTANT
        tmp = self.data.copy()
        tmp = tmp.diff(1).dropna().values
        shape = tmp.shape[:-1] + (tmp.shape[-1] - self.lookBack + 1,
                                  self.lookBack)
        strides = tmp.strides + (tmp.strides[-1],)
        self.dataset = np.lib.stride_tricks.as_strided(tmp, shape=shape,
                                                       strides=strides)

    def getReward(self, action, currentPrice):
        # NOT IMPORTANT
        return 0

    @tf.function
    def updateWeights(self):
        # IMPORTANT
        tf.print(len(self.memory))
        if len(self.memory) >= self.batchSize:
            endIndex = len(self.memory)
            startIndex = endIndex - self.batchSize
            batchData = []
            for i in range(startIndex, endIndex):
                batchData.append(self.memory[i])
            X = np.zeros((self.batchSize, self.lookBack))
            Y = np.zeros((self.batchSize, self.actionSize))
            states = np.array([item[0] for item in batchData])
            newStates = np.array([item[3] for item in batchData])
            Q = self.model(states)
            QNext = self.model(newStates)
            for i in range(len(batchData)):
                state, action, reward, nextState = batchData[i]
                target = Q[i]
                target[action] = reward
                target[action] += self.gamma * np.max(QNext[i])

                X[i] = state
                Y[i] = target
            self.model.train_on_batch(X, Y)
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay

    def train(self, epochs=200, logFreq=1):
        # IMPORTANT
        for epoch in range(epochs):
            self.profit = 0
            self.money = self.initialMoney
            for timeStep in tqdm.tqdm(range(self.lookBack, len(self.data)-1)):
                currentPrice = data[timeStep]
                currentState = self.dataset[timeStep-self.lookBack]
                nextState = self.dataset[timeStep-self.lookBack+1]

                action = self.getAction(currentState.reshape(1, -1))

                reward = self.getReward(action, currentPrice)

                self.memory.append((currentState, action, reward, nextState))

                self.updateWeights()


test = DQN(data)
test.createDataset()
test.buildModel()
test.train(100)

X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
Y = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
with tf.GradientTape() as tape:
    Q = self.model(states)
    QNext = self.model(newStates)
    for i in range(len(states)):
        state, action, reward, _ = states[i], actions[i], rewards[i], newStates[i]
        target = tf.Variable(Q[i], name="temp")
        nextQ = QNext[i][tf.math.argmax(QNext[i])]
        newVal = reward + self.gamma * nextQ
        target[action].assign(newVal)

        X.write(i, state)
        Y.write(i, target)
    loss = self.lossFunc(self.model(X.stack()), Y.stack())
    grads = tape.gradient(loss, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))