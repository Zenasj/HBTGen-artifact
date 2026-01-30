import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

print("TF version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

BATCH_SIZE = 48
N_CLASSES = 27
INPUT_SHAPE = (240, 320, 3)

class DataGenerator(tf.keras.utils.Sequence):
    'Generates random data'

    def __init__(self, batch_size=32, dim=INPUT_SHAPE, n_classes=10):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 10000000

    def __getitem__(self, index):
        'Generate one batch of data'
        X = np.random.uniform(size=([self.batch_size] + list(self.dim)))
        y = np.random.randint(0, self.n_classes, size=(self.batch_size, 1))
        return X, y


train_ds = DataGenerator(batch_size=BATCH_SIZE, n_classes=N_CLASSES)
valid_ds = DataGenerator(batch_size=BATCH_SIZE, n_classes=N_CLASSES)

base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, pooling='avg',
                                                               weights=None, input_shape=INPUT_SHAPE)
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),
    base_model,
    tf.keras.layers.Dense(N_CLASSES, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,) + INPUT_SHAPE)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

hist = model.fit(
    train_ds,
    epochs=5, steps_per_epoch=100,
    validation_data=valid_ds,
    validation_steps=10).history