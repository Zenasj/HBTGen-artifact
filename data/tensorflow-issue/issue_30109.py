# tf.random.uniform((BATCH_SIZE, 5, 20), dtype=tf.float32) ← Input shape inferred from time_steps=5, sample_width=20

import tensorflow as tf

time_steps = 5
sample_width = 20
kernel_size = 3
num_filters = 5
num_classes = 5

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The inner model that will be applied to each time step independently:
        # This mimics the "sample_model" in the original code:
        self.reshape1 = tf.keras.layers.Reshape((sample_width, 1))
        # BatchNorm with momentum set to 0.5 to avoid training-validation mismatch described in the issue
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.conv = tf.keras.layers.Conv1D(num_filters, kernel_size, padding='same')
        self.reshape2 = tf.keras.layers.Reshape((num_filters * sample_width,))
        # TimeDistributed wrapper will be replaced by functional logic using tf.map_fn for easier control in subclassed model
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, time_steps, sample_width)
        # Process each time step independently via tf.map_fn to replicate TimeDistributed behavior
        def step_fn(step_input):
            # step_input shape: (batch_size, sample_width)
            x = self.reshape1(step_input)
            x = self.batch_norm(x, training=training)
            x = self.conv(x)
            x = self.reshape2(x)
            return x
        # map_fn iterates over the time dimension (axis=1)
        # We transpose inputs to (time_steps, batch_size, sample_width) to map over time_steps
        # then transpose back after.
        # Alternatively, map over axis=1 using tf.map_fn with fn applied to each time slice.
        # tf.map_fn defaults to mapping over the first dimension, so we transpose.
        x = tf.transpose(inputs, perm=[1, 0, 2])  # (time_steps, batch_size, sample_width)
        encoded = tf.map_fn(step_fn, x)
        # encoded shape: (time_steps, batch_size, num_filters * sample_width)
        encoded = tf.transpose(encoded, perm=[1, 0, 2])  # (batch_size, time_steps, num_filters * sample_width)
        # Now apply dense layer to each time step output (Dense supports 3D input applied last dimension-wise)
        # Result shape: (batch_size, time_steps, num_classes)
        out = self.dense(encoded)
        return out

def my_model_function():
    # Returns an instance of MyModel with initialized weights
    return MyModel()

def GetInput():
    # Returns a random tensor input shape matching (batch_size, time_steps, sample_width)
    # batch size arbitrarily chosen as 32
    batch_size = 32
    return tf.random.uniform((batch_size, time_steps, sample_width), dtype=tf.float32)

