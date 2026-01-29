# tf.random.uniform((1, 28000), dtype=tf.float32) ← Inferred input shape from issue examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        size = 28000
        
        # Following issue discussion, wrap kernel initializer in tf.function to reduce memory usage in TF 2.x
        glorot_uniform_init = tf.keras.initializers.glorot_uniform()
        # Wrap the __call__ method of the initializer instance to keep the benefit of tf.function while preserving init behavior
        self.wrapped_initializer = tf.function(glorot_uniform_init, autograph=False)

        # Create Dense layer with the wrapped initializer to prevent large memory spikes on init
        self.dense = tf.keras.layers.Dense(
            units=size,
            kernel_initializer=self.wrapped_initializer
        )

    def call(self, inputs, training=False):
        return self.dense(inputs)


def my_model_function():
    # Instantiate MyModel
    return MyModel()


def GetInput():
    # Generate input tensor matching (batch_size=1, input_dim=28000) with float32 type
    return tf.random.uniform((1, 28000), dtype=tf.float32)

