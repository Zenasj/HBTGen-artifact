# tf.random.uniform((B, 1, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using a GRUCell inside an RNN layer with unroll=True.
        # Input shape is (batch_size, time_steps=1, features=1)
        self.rnn_layer = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(10),
            unroll=True,
        )
    
    def call(self, inputs):
        # Forward pass through the RNN layer
        return self.rnn_layer(inputs)


def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # The input expected is a 3D tensor (batch_size, time_steps, features)
    # According to the example, shape is (B, 1, 1)
    # Use float32 dtype as standard for Keras models
    batch_size = 2  # arbitrary batch size
    time_steps = 1  # fixed, required for unroll=True
    features = 1
    return tf.random.uniform(shape=(batch_size, time_steps, features), dtype=tf.float32)

