from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

class ToyModel(tf.keras.Model):
    """A simple linear model."""

    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(5)

    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32, name="input_func")])
    def call(self, x):
        return self.l1(x)

def toy_dataset():
    inputs = tf.range(10.)[:, None]
    outputs = inputs * 5. + tf.range(5.)[None, :]
    # TODO: switch `tuple` to `dict`.
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    return dataset.repeat(10).batch(2).shuffle(buffer_size=5)


dataset = toy_dataset()
optimizer = tf.keras.optimizers.Adam(0.1)

model = ToyModel()
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(dataset, epochs=5)

# Inference works
inputs = np.array([[0, 5]], dtype=np.float32).T
print(model(inputs))

# Export to Saved Model
model_path = "/tmp/saved_model"
tf.saved_model.save(model, model_path)

# Load model
saved_model = tf.saved_model.load(model_path)

# Inference from Saved Model
#inputs = np.array([[0, 5]], dtype=np.float32).T
#saved_model(inputs)