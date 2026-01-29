# tf.random.uniform((B,)) ← inferred input shape is a vector, size unspecified (binary classification input assumed)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        # Basic feedforward binary classifier similar to issue example
        self.dense1 = tf.keras.layers.Dense(128)
        self.act = tf.keras.layers.Activation('relu')
        self.out_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        # For demonstration, store input shape for building
        self._input_shape = input_shape

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.act(x)
        return self.out_layer(x)

def my_model_function():
    # Assume input shape is a 1D tensor (feature vector) with size 20 as reasonable guess
    input_shape = (20,)
    model = MyModel(input_shape)
    # Build model by calling once (optional but useful for clarity)
    model.build(input_shape=(None,) + input_shape)
    # Compile model with the same parameters from the issue
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Generate a batch of random inputs matching the model's expected input shape (batch size 32)
    input_shape = (20,)
    batch_size = 32
    # Random uniform floats in [0,1)
    return tf.random.uniform((batch_size,) + input_shape, dtype=tf.float32)

