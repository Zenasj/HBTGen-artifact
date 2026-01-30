from tensorflow.keras import optimizers

import sys

from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 3
STEPS_PER_EPOCH = 10
BATCH_SIZE = 64
SAMPLES = BATCH_SIZE * STEPS_PER_EPOCH


inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(4*4096, activation='relu', name='dense_1')(inputs)
x = layers.Dense(4*4096, activation='relu', name='dense_2')(x)
x = layers.Dense(4*4096, activation='relu', name='dense_3')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train[:SAMPLES].reshape(SAMPLES, 784).astype('float32') / 255
y_train = y_train[:SAMPLES].astype('float32')

class PrintStatsCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        print('\nON_BATCH_END: loss: {loss:.4f} acc: {accuracy:.4f}'.format_map(logs), file=sys.stderr)
    def on_epoch_end(self, epoch, logs=None):
        print('\nON_EPOCH_END: loss: {loss:.4f} acc: {accuracy:.4f}'.format_map(logs), file=sys.stderr)

# Specify the training configuration (optimizer, loss, metrics)
model.compile(optimizer=keras.optimizers.SGD(lr=0.1),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          batch_size=BATCH_SIZE,
          callbacks=[PrintStatsCallback()])