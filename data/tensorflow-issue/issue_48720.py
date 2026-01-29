# tf.random.uniform((20, 64), dtype=tf.float32)  ← Input shape inferred from training input: 20 frames * 64 pixels each flattened to vector of length 1280 (but model input is flattened 20x64)
# Actually inputs shape used in training is (num_samples, 20*64) per sample, so input shape is (1280,).
# But the model processes each sample as a flat vector of shape (1280,)
# For tf.keras.Model, input shape will be (1280,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Following the Sequential model layers from the issue:
        # Dense(50, relu) -> Dense(15, relu) -> Dense(2, softmax)
        self.dense1 = tf.keras.layers.Dense(50, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(15, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        # inputs shape: (batch_size, 1280)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.output_layer(x)
        return x

def my_model_function():
    # Instantiate and return the MyModel instance
    model = MyModel()
    # Build the model by calling it on a dummy input with batch size 1, to initialize weights
    dummy_input = tf.zeros((1, 1280), dtype=tf.float32)
    model(dummy_input)
    return model

def GetInput():
    # Return a random tensor input matching expected input shape (batch size 20, 1280 features)
    # From the training code, the model input is a flat vector of 1280 per example.
    # We return a batch of inputs, say 20 samples to simulate the training batch size.
    return tf.random.uniform((20, 1280), dtype=tf.float32)

