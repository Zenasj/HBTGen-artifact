# tf.random.uniform((B, window_size // subseq_size, subseq_size, C), dtype=tf.float32)
import tensorflow as tf

# Assumptions on input shape:
# From the code snippet: input_shape=(window_size // subseq_size, subseq_size)
# This looks like a 3D input (time steps, subsequence length, channels)
# The example uses Conv1D layers, which expect (batch, steps, channels).
# Also, keras TimeDistributed(Flatten()) is applied after Conv1D + MaxPooling1D,
# so input channels inferred as 1 for simplicity.
# We'll fix subseq_size=10, window_size=100 for example, and channels=1.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the architecture from the issue:
        # Conv1D layers with filters and kernel_size=2, and activation
        # Using 'relu' as placeholder activation.
        self.conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu')
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.td_flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.lstm = tf.keras.layers.LSTM(500)
        self.dense1 = tf.keras.layers.Dense(100)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # inputs shape: (batch, steps, subseq_size, channels=1)
        # Since Conv1D expects 3D input (batch, steps, channels),
        # we need to merge the last two dims or reshape:
        # The original code uses input_shape=(window_size // subseq_size, subseq_size)
        # for Conv1D input, so likely inputs are 3D: (batch, steps, channels).
        # But the reported input in the comment assumes 4D shape for tf.random.uniform.
        # We'll squeeze the last dim if it is 1 to get 3D for Conv1D.

        # If input rank is 4, combine last two dims as channels for Conv1D input:
        # However, the original code uses Conv1D layers expecting (batch, steps, channels).
        # So we collapse subseq_size and channels into one channels dimension.
        shape = tf.shape(inputs)
        if inputs.shape.rank == 4:
            # Merge last two dims: subseq_size * channels
            batch = shape[0]
            steps = shape[1]
            subseq = shape[2]
            channels = shape[3]
            x = tf.reshape(inputs, (batch, steps, subseq * channels))
        else:
            x = inputs  # Already 3D

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        # The original model applies TimeDistributed(Flatten()), which expects 4D input.
        # After MaxPooling1D, x is 3D (batch, steps, channels).
        # To apply TimeDistributed, we reshape to 4D by introducing a dummy dimension:
        x_shape = tf.shape(x)
        x = tf.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2]))
        x = self.td_flatten(x)  # This flattens last two dims

        # After TimeDistributed(Flatten()), shape (batch, steps, flattened_features)
        x = self.lstm(x)

        x = self.dense1(x)
        out = self.dense2(x)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Use window_size=100, subseq_size=10 as example, channels=1
    window_size = 100
    subseq_size = 10
    batch_size = 8  # arbitrary batch size
    channels = 1

    # Input shape: (batch, steps=window_size//subseq_size, subseq_size, channels)
    steps = window_size // subseq_size
    input_tensor = tf.random.uniform(
        (batch_size, steps, subseq_size, channels),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )
    return input_tensor

