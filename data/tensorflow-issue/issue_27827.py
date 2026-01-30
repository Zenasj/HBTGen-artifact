from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf


class KerasTest:
    def __init__(self, state_space, action_space, lr):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr

        inputs = Input(shape=(84, 84, 4))
        rewards = Input(shape=(1,))

        x = Conv2D(32, kernel_size=[8, 8], padding='valid', strides=[4, 4], activation=None)(inputs)
        x = BatchNormalization(trainable=True, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, kernel_size=[4, 4], padding='valid', strides=[2, 2], activation=None)(x)
        x = BatchNormalization(trainable=True, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, kernel_size=(4, 4), padding='valid', strides=(2, 2), activation=None)(x)
        x = BatchNormalization(trainable=True, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.05)(x)
        logits = Dense(self.action_space, activation=None)(x)

        self.model = Model(inputs=[inputs, rewards], outputs=logits)

        def policy_loss(r):
            def loss(labels, logits):
                policy = tf.nn.softmax(logits)
                entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)
                log = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
                p_loss = log * tf.stop_gradient(r)
                p_loss = p_loss - 0.01 * entropy
                total_loss = tf.reduce_mean(p_loss)
                return total_loss

            return loss

        self.model.compile(optimizer=Adam(lr=lr), loss=policy_loss(rewards))
        self.model.summary()

    def get_probs(self, s):
        s = s[np.newaxis, :]
        probs = self.model.predict([s, np.array([1])])
        probs = probs.squeeze()
        probs = self.softmax(probs)
        return probs

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def update_policy(self, s, r, a):
        self.model.train_on_batch([s, r], a)

import tensorflow as tf
from tensorflow.python.keras.layers import *
import numpy as np

tf.enable_eager_execution()
print(tf.executing_eagerly())


class EagerTest:
    def __init__(self, state_space, action_space, lr):
        self.action_space = action_space
        self.lr = lr

        inputs = Input(shape=(84, 84, 4))
        rewards = Input(shape=(1,))

        x = Conv2D(32, kernel_size=[8, 8], padding='valid', strides=[4, 4], activation=None)(inputs)
        x = BatchNormalization(trainable=True, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, kernel_size=[4, 4], padding='valid', strides=[2, 2], activation=None)(x)
        x = BatchNormalization(trainable=True, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, kernel_size=(4, 4), padding='valid', strides=(2, 2), activation=None)(x)
        x = BatchNormalization(trainable=True, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.05)(x)
        logits = Dense(self.action_space, activation=None)(x)

        self.model = tf.keras.Model(inputs=[inputs, rewards], outputs=logits)

        def policy_loss(r):
            def loss(labels, logits):
                policy = tf.nn.softmax(logits)
                entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)
                log = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
                p_loss = log * tf.stop_gradient(r)
                p_loss = p_loss - 0.01 * entropy
                total_loss = tf.reduce_mean(p_loss)
                return total_loss

            return loss

        self.model.compile(optimizer=tf.train.AdamOptimizer(lr), loss=policy_loss(rewards))
        self.model.summary()

    def get_probs(self, s):
        s = s[np.newaxis, :]
        probs = self.model([s, np.array([1])]).numpy()
        probs = probs.squeeze()
        probs = self.softmax(probs)
        return probs

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def update_policy(self, s, r, a):
        self.model.train_on_batch([s, r], a)

import tensorflow as tf
from tensorflow.python.keras.layers import *

import numpy as np

tf.enable_eager_execution()
print(tf.executing_eagerly())


class EagerSeqTest:
    def __init__(self, state_space, action_space, lr):
        self.state_space = state_space
        self.action_space = action_space

        self.model = tf.keras.Sequential()

        self.model.add(InputLayer(input_shape=(84, 84, 4)))

        # Conv
        self.model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), name='conv1'))
        self.model.add(BatchNormalization(trainable=True, epsilon=1e-5, name='batch_norm1'))
        self.model.add(ReLU(name='conv_1_out'))

        self.model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), name='conv2'))
        self.model.add(BatchNormalization(trainable=True, epsilon=1e-5, name='batch_norm2'))
        self.model.add(ReLU(name='conv_2_out'))

        self.model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), name='conv3'))
        self.model.add(BatchNormalization(trainable=True, epsilon=1e-5, name='batch_norm3'))
        self.model.add(ReLU(name='conv_3_out'))

        self.model.add(Flatten(name='flatten'))

        # Fully connected
        self.model.add(Dense(units=512, activation='relu', name='fc1'))
        self.model.add(Dropout(rate=0.4, name='dr1'))
        self.model.add(Dense(units=256, activation='relu', name='fc2'))
        self.model.add(Dropout(rate=0.3, name='dr2'))
        self.model.add(Dense(units=64, activation='relu', name='fc3'))
        self.model.add(Dropout(rate=0.03, name='dr3'))

        # Logits
        self.model.add(Dense(units=self.action_space, activation='linear', name='logits'))

        self.model.summary()

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    def get_probs(self, s):
        s = s[np.newaxis, :]
        logits = self.model(s)
        probs = tf.nn.softmax(logits).numpy().squeeze()
        return probs

    def update_policy(self, s, r, a):
        with tf.GradientTape() as tape:
            loss = self.calc_loss(s, r, a)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def calc_loss(self, s, r, a):
        logits = self.model(s)
        policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=a, logits=logits)
        policy_loss = tf.reduce_mean(policy_loss * tf.stop_gradient(r))
        return policy_loss