# tf.random.uniform((B, 30), dtype=tf.float32) ← input shape inferred to be (batch, lookBack=30)

import tensorflow as tf
from tensorflow import keras
import numpy as np


class MyModel(tf.keras.Model):
    def __init__(self, lookBack=30, actionSize=3, gamma=0.95,
                 epsilon=0.5, epsilonMin=0.01, epsilonDecay=0.8,
                 learningRate=0.001, batchSize=32):
        super().__init__()
        self.lookBack = lookBack
        self.actionSize = actionSize
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.batchSize = batchSize

        # Build the internal model
        self.model = keras.models.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(self.lookBack,)),
            keras.layers.Dense(self.actionSize)
        ])
        self.optimizer = keras.optimizers.RMSprop(learning_rate=self.learningRate,
                                                  epsilon=0.1,
                                                  rho=0.99)
        self.lossFunc = tf.keras.losses.MeanSquaredError()

        # Memory as a python list will not work with tf.function well.
        # We'll keep it as a python list outside tf.function for simplicity.
        # Note: for production use, consider using tf.Tensor or tf.data.Dataset.
        self.memory = []

    @tf.function
    def updateWeights(self, batchData):
        """
        Performs a training step on a batch of data:
        batchData shape: tuple of (states, actions, rewards, nextStates)
        - states: tf.float32 tensor shape (batch, lookBack)
        - actions: tf.int32 tensor shape (batch,)
        - rewards: tf.float32 tensor shape (batch,)
        - nextStates: tf.float32 tensor shape (batch, lookBack)
        """
        states, actions, rewards, nextStates = batchData

        with tf.GradientTape() as tape:
            # Predict Q values for current states and next states
            Q = self.model(states)                 # shape (batch, actionSize)
            QNext = self.model(nextStates)         # shape (batch, actionSize)

            # Create targets tensor to train on
            # Because tensors are immutable, we use tensor_scatter_nd_update
            batch_indices = tf.range(tf.shape(states)[0])

            # Gather max Q for next states
            maxQNext = tf.reduce_max(QNext, axis=1)

            # Calculate new Q values for the taken actions
            # target = reward + gamma * maxQNext
            targetQ = rewards + self.gamma * maxQNext

            # Current Q values for actions taken: shape (batch,)
            indices = tf.stack([batch_indices, actions], axis=1)

            # Scatter the updated Q values into Q to create target tensor
            updatedQ = tf.tensor_scatter_nd_update(Q, indices, targetQ)

            # Compute loss and gradients
            loss = self.lossFunc(updatedQ, Q)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Decay epsilon after each update
        if self.epsilon > self.epsilonMin:
            self.epsilon = self.epsilon * self.epsilonDecay
        return loss

    def getAction(self, state):
        """
        Epsilon-greedy action selection.
        state: (1, lookBack) numpy array or tensor
        Returns action as int.
        """
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.actionSize)
        else:
            q_values = self.model(state)
            return int(tf.argmax(q_values[0]).numpy())

    def addMemory(self, experience):
        """
        Add experience tuple:
        (state (np array), action (int), reward (float), nextState (np array))
        """
        self.memory.append(experience)

    def sampleBatch(self):
        """
        Samples a batch from memory and converts to tensors.
        If not enough samples, returns None.
        """
        if len(self.memory) < self.batchSize:
            return None

        batchData = self.memory[-self.batchSize:]
        states = np.array([item[0] for item in batchData], dtype=np.float32)
        actions = np.array([item[1] for item in batchData], dtype=np.int32)
        rewards = np.array([item[2] for item in batchData], dtype=np.float32)
        nextStates = np.array([item[3] for item in batchData], dtype=np.float32)
        return (tf.convert_to_tensor(states),
                tf.convert_to_tensor(actions),
                tf.convert_to_tensor(rewards),
                tf.convert_to_tensor(nextStates))

    def train_step(self, currentState, action, reward, nextState):
        """
        Add a single experience to memory and update weights if batch is ready.
        """
        self.addMemory((currentState, action, reward, nextState))
        batch = self.sampleBatch()
        if batch is not None:
            loss = self.updateWeights(batch)
            return loss
        else:
            return None


def my_model_function():
    """
    Returns a new instance of MyModel initialized with default parameters.
    """
    return MyModel()

def GetInput():
    """
    Returns a random tensor input matching the expected input shape for MyModel.
    Shape: (batch_size=1, lookBack=30)
    dtype: tf.float32
    """
    return tf.random.uniform((1, 30), dtype=tf.float32)

