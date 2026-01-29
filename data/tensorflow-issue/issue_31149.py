# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from example model input_shape=[1]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # To simulate reproducible initializers without manually setting seed each time,
        # we set a fixed seed here and use it for all kernel initializers.
        seed = 1
        self.dense1 = tf.keras.layers.Dense(
            8,
            activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
            input_shape=(1,)
        )
        self.dense2 = tf.keras.layers.Dense(
            8,
            activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
        self.dense3 = tf.keras.layers.Dense(
            1,
            activation='linear',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    # Return an instance of MyModel with reproducible initializers via fixed seeds
    return MyModel()

def GetInput():
    # Return a random tensor input matching the input expected by MyModel
    # Format: batch size B, feature dimension 1 (as per input_shape=[1])
    B = 5  # arbitrary batch size for demonstration
    return tf.random.uniform((B, 1), dtype=tf.float32)

