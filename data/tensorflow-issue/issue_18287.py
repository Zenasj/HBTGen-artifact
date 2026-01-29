# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape from example: input_dim=10 for model

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple dense layer matching the described example
        self.dense = tf.keras.layers.Dense(4, input_shape=(10,))

    def call(self, inputs, training=False):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Instantiate the model and compile it as done in the example
    model = MyModel()
    # Using RMSProp optimizer with learning rate 0.001 and MSE loss, matching the original snippet
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def GetInput():
    # Create a random input tensor matching the expected shape (batch_size, 10)
    # Using batch size 32 as a common default; dtype float32 for compatibility
    return tf.random.uniform(shape=(32, 10), dtype=tf.float32)

