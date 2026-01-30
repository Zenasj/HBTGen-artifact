import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

def build_G(batchnorm_momentum=0.9):
    input_layer = Input(shape=(2,))
    for i in range(4):
        X = Dense(64)(input_layer)
        X = Activation('relu')(X)
    output_layer = Dense(2)(X)
    model = Model(input_layer, output_layer)
    return model


def build_D(batchnorm_momentum=0.9):
    input_layer = Input(shape=(2,))
    for i in range(4):
        X = Dense(64)(input_layer)
        if batchnorm_momentum:
            X = BatchNormalization(momentum=batchnorm_momentum)(X)
        X = Activation('relu')(X)
    output_layer = Dense(1, activation='sigmoid')(X)
    model = Model(input_layer, output_layer)
    model.compile(Adam(lr=0.001, beta_1=0.5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'],)
    return model


def build_GAN(G, D):
    D.trainable=False
    input_layer = Input(shape=(2,))
    X = G(input_layer)
    output_layer = D(X)
    model = Model(input_layer, output_layer)
    model.compile(Adam(lr=0.0002, beta_1=0.5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'],)
    return model


def get_noise(num):
    return np.random.random((num, 2)) * 2 - 1


def get_samples(num):
    return np.random.normal(0, 1, (num, 2))


BATCHNORM_MOMENTUM = 0.9
# BATCHNORM_MOMENTUM = None
G = build_G()
D = build_D(batchnorm_momentum=BATCHNORM_MOMENTUM)
GAN = build_GAN(G, D)


EPOCHS = 100
BATCH_SIZE = 32
BATCHES_PER_EPOCH = 100
g_loss = []
g_accuracy = []
d_real_loss = []
d_real_accuracy = []
d_fake_loss = []
d_fake_accuracy = []
for epoch in range(EPOCHS):
    g_running_loss = 0
    g_running_accuracy = 0
    d_real_running_loss = 0
    d_real_running_accuracy = 0
    d_fake_running_loss = 0
    d_fake_running_accuracy = 0
    for batch in range(BATCHES_PER_EPOCH):
        real = get_samples(BATCH_SIZE)
        fake = G.predict(get_noise(BATCH_SIZE))
        l, a = D.train_on_batch(real, np.ones((BATCH_SIZE, 1)))
        d_real_running_loss += l
        d_real_running_accuracy += a
        l, a = D.train_on_batch(fake, np.zeros((BATCH_SIZE, 1)))
        d_fake_running_loss += l
        d_fake_running_accuracy += a

        l, a = GAN.train_on_batch(get_noise(BATCH_SIZE), np.ones((BATCH_SIZE, 1)))
        g_running_loss += l
        g_running_accuracy += a

        print(f'Epoch {epoch+1} [{batch+1}/{BATCHES_PER_EPOCH}]: '
              f'G={g_running_loss/(batch+1):.4f} [{g_running_accuracy/(batch+1):.2%}]; '
              f'Dr={d_real_running_loss/(batch+1):.4f} [{d_real_running_accuracy/(batch+1):.2%}]; '
              f'Df={d_fake_running_loss/(batch+1):.4f} [{d_fake_running_accuracy/(batch+1):.2%}]; '
              f'Overall={(g_running_loss + (d_real_running_loss + d_fake_running_loss) / 2) / (batch+1):.4f}',
              end='\r'
              )
    g_loss.append(g_running_loss / BATCHES_PER_EPOCH)
    g_accuracy.append(g_running_accuracy / BATCHES_PER_EPOCH)
    d_real_loss.append(d_real_running_loss / BATCHES_PER_EPOCH)
    d_real_accuracy.append(d_real_running_accuracy / BATCHES_PER_EPOCH)
    d_fake_loss.append(d_fake_running_loss / BATCHES_PER_EPOCH)
    d_fake_accuracy.append(d_fake_running_accuracy / BATCHES_PER_EPOCH)
    print()

g_loss = np.array(g_loss)
d_real_loss = np.array(d_real_loss)
d_fake_loss = np.array(d_fake_loss)
d_loss = (d_real_loss + d_fake_loss) / 2
plt.plot(np.arange(len(g_loss)) + 1, g_loss, label='G Loss')
plt.plot(np.arange(len(d_loss)) + 1, d_loss, label='D Loss')
plt.plot(np.arange(len(d_loss)) + 1, g_loss + d_loss, label='GAN Objective')
plt.plot([1, len(d_loss)], [np.log(4), np.log(4)], label='GAN Objective Theoretical Minimum')
plt.plot([1, len(d_loss)], [np.log(4)/2, np.log(4)/2], label='Equilibrium')
plt.title("With BatchNorm" if BATCHNORM_MOMENTUM else "Without BatchNorm")
plt.legend(loc='lower left')
plt.show()

g_accuracy = np.array(g_accuracy)
d_real_accuracy = np.array(d_real_accuracy)
d_fake_accuracy = np.array(d_fake_accuracy)
d_accuracy = (d_real_accuracy + d_fake_accuracy) / 2
plt.plot(np.arange(len(g_accuracy)) + 1, g_accuracy, label='G accuracy')
plt.plot(np.arange(len(d_accuracy)) + 1, d_accuracy, label='D accuracy')
plt.plot([1, len(d_accuracy)], [1, 1], label='100%')
plt.title("With BatchNorm" if BATCHNORM_MOMENTUM else "Without BatchNorm")
plt.ylim(-0.05, 1.05)
plt.legend(loc='lower left')
plt.show()