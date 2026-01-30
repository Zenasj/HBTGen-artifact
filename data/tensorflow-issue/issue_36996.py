import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

with strategy.scope():
    # generator_model
    generator_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=noise_shape),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(784, activation='tanh'),
        tf.keras.layers.Reshape(img_shape)
    ])
    generator_model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
    generator_model.summary()

    # discriminator_model
    discriminator_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    discriminator_model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                                loss='binary_crossentropy')
    discriminator_model.summary()

    # combined_model
    z = tf.keras.layers.Input(shape=noise_shape)
    discriminator_model.trainable = False 
    valid = discriminator_model(generator_model(z)) 
    combined_model = tf.keras.models.Model(z, valid) 
    combined_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    combined_model.summary()

# train
for epoch in range(10000):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs = x_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    gen_imgs = generator_model.predict(noise.astype(np.float32))

    # train discriminator_model
    d_loss_real = discriminator_model.fit(imgs.astype(np.float32), np.ones((batch_size, 1)))
    d_loss_fake = discriminator_model.fit(gen_imgs.astype(np.float32), np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # train combined_model
    noise = np.random.normal(0, 1, (batch_size * 2, 100))
    valid_y = np.array([1] * batch_size * 2)  
    g_loss = combined_model.fit(noise.astype(np.float32), valid_y)

d_loss_real = model.fit(x_train.astype(np.float32), np.ones((x_train.shape[0],1)))

d_loss_real = model.fit(x_train.astype(np.float32), np.ones((x_train.shape[0],1)).astype(np.float32))