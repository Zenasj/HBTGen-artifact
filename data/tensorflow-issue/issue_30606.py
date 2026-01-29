# tf.random.uniform((B, 1), dtype=tf.float32)  ← Input shape is a batch of scalars (single features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Single dense layer model representing f(x) = 2x - 1 approximately after training
        self.dense = tf.keras.layers.Dense(units=1, input_shape=(1,), activation=None)

    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel.
    # This model requires training to achieve expected function;
    # weights are randomly initialized here as no checkpoint loading is included.
    return MyModel()

def GetInput():
    # Return a random tensor input with shape [batch_size, 1] for the model
    # Assuming a batch size of 4 for demonstration.
    return tf.random.uniform((4, 1), minval=-10.0, maxval=10.0, dtype=tf.float32)

