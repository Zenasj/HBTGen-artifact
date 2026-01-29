# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from Fashion MNIST images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the model architecture from the issue:
        # Flatten input 28x28 -> 784, 
        # Dense 128 units with relu activation,
        # BatchNormalization,
        # Dense 10 units without activation (logits output).
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(10)  # No activation here (logits)

    def call(self, inputs, training=False):
        # inputs assumed to be shape (B, 28, 28) with float32 normalized [0,1]
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.batchnorm(x, training=training)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an untrained model instance resembling the one in the issue
    # No weights loading because original code used graph sessions and checkpoints not directly portable here.
    model = MyModel()
    # Build model by passing dummy input (required to create weights in TF2 eager)
    _ = model(tf.zeros((1, 28, 28), dtype=tf.float32))
    return model

def GetInput():
    # Return a random tensor input that matches model input: batch of 1, 28x28, float32 normalized to [0,1]
    # In original code, train_images / 255.0 -> normalized floats; here generate uniform float input
    return tf.random.uniform(shape=(1, 28, 28), minval=0.0, maxval=1.0, dtype=tf.float32)

