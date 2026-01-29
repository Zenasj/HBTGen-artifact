# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape for MNIST grayscale images batch

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the reported architecture:
        # Conv2D layer with 32 filters, kernel size 3, and ReLU activation
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu')
        # MaxPooling2D layer (default pool size 2x2)
        self.pool = tf.keras.layers.MaxPooling2D()
        # The original code commented out Flatten, but TimeDistributed operates over time dimension
        # Here we assume input shape is (batch, height, width, channels)
        # To work with TimeDistributed Dense, we need a "time" dimension.
        # Given the issue, the `TimeDistributed` layer is applied directly after Conv2D+Pool,
        # so we interpret the spatial dimension as "time steps"
        #
        # We'll reshape the output spatially into a sequence for TimeDistributed Dense:
        #
        # After Conv2D+Pool, output shape: (batch, new_height, new_width, filters)
        #
        # We'll treat new_height as "time" and new_width * filters as "features"
        #
        # So we will reshape before applying TimeDistributed.
        #
        # Since original example uses TimeDistributed(Dense(64)), we'll do the same.
        self.time_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='softmax'))

    def call(self, inputs):
        # inputs: (batch, 28, 28, 1)
        x = self.conv(inputs)          # (batch, 26, 26, 32)
        x = self.pool(x)               # (batch, 13, 13, 32)
        # Reshape spatial dim to time steps:
        # Treat height dimension as time steps, flatten width and channels as features
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]    # 13
        feature_dim = tf.shape(x)[2] * tf.shape(x)[3]  # 13 * 32 = 416 (dynamic)
        x = tf.reshape(x, (batch_size, time_steps, feature_dim))  # (batch, 13, 416)
        x = self.time_dense(x)         # (batch, 13, 64) - 64 features per time step
        return x

def my_model_function():
    # Instantiate and compile the model the way it's done in the repo snippet
    model = MyModel()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a random tensor that matches the input shape expected by MyModel
    # (batch_size, 28, 28, 1) - grayscale MNIST images batch
    batch_size = 8   # arbitrary batch size
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

