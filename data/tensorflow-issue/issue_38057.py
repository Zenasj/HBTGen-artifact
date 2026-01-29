# tf.random.uniform((B, n_timestep, 6), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the layers from the provided Sequential model:
        # Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timestep,6))
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='valid')
        # Conv1D(filters=32, kernel_size=3, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='valid')
        # Dropout(0.5)
        self.dropout = tf.keras.layers.Dropout(0.5)
        # MaxPooling1D(pool_size=2) - layer that caused issues on ESP32
        self.maxpool = tf.keras.layers.MaxPooling1D(pool_size=2)
        # Flatten
        self.flatten = tf.keras.layers.Flatten()
        # Dense(100, activation='relu')
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        # Dense(4, activation='softmax')
        self.dense2 = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.dropout(x, training=training)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # From the original model input shape: (n_timestep, 6)
    # n_timestep is not explicitly given, so we assume a common timestep, e.g., 100 for demonstration.
    n_timestep = 100
    # Batch size can be arbitrary, use 1
    batch_size = 1
    # Input tensor shape: (batch_size, n_timestep, 6), dtype float32 to match Conv1D input
    return tf.random.uniform(shape=(batch_size, n_timestep, 6), dtype=tf.float32)

