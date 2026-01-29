# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape corresponds to flattened MNIST images (28*28=784)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the create_model() definition from the issue:
        # Sequential: Dense(512, relu) -> Dropout(0.2) -> Dense(10, softmax)
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,))
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        
        # Compile components not strictly needed here, but logic aligns with issue's model.compile
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Returns a fresh instance of MyModel; does not load weights by default,
    # to match the create_model() logic in the issue.
    return MyModel()

def GetInput():
    # Generate a batch of random inputs matching (batch_size, 784) for MNIST flattened images
    # As batch size is not fixed in the issue, we choose 32 arbitrarily.
    return tf.random.uniform((32, 784), minval=0, maxval=1, dtype=tf.float32)

