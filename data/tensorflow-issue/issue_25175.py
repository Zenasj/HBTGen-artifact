# tf.random.uniform((B, 28, 28), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using Sequential with explicit input shape to avoid the deprecated "deferred mode" bug
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),  # Must specify input_shape to fix the dropout training issue
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    def call(self, inputs, training=False):
        # Pass the training flag to enable correct Dropout behavior:
        # When training=True, Dropout is active; when training=False, it's bypassed
        return self.model(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel with a properly initialized dropout layer
    # No custom weights needed as this is a demo model following common MNIST tutorial architecture
    return MyModel()

def GetInput():
    # Produce a batch of random float inputs shaped like MNIST images: (batch_size, 28, 28)
    # Use dtype float32 consistent with typical keras preprocessing
    batch_size = 4  # arbitrary small batch size for testing
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

