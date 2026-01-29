# tf.random.uniform((B, 8), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.out = tf.keras.layers.Dense(output_dim)
        # Creating a Mean metric instance but not adding it as a metric or using it in calls
        # can cause issues in some TF 2.0 versions, but here we keep it to match the original example.
        self.my_metric = tf.keras.metrics.Mean()

    def call(self, inputs):
        # Note: The original issue mentions if you add the line below:
        # self.add_metric(self.my_metric(5.)), the model becomes stateful.
        # Here we comment it out to replicate "harmless but unused" behavior.
        # self.add_metric(self.my_metric(5.))
        return self.out(inputs)

def my_model_function():
    # Return an instance of MyModel with output_dim=1 as in the example
    return MyModel(1)

def GetInput():
    # Return a random float32 input tensor with shape (batch_size, 8).
    # Assumption: batch_size 32 (matching training_set_size multiples in example)
    return tf.random.uniform((32, 8), dtype=tf.float32)

