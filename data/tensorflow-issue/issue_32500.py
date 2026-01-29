# tf.random.uniform((20, 42, 3), dtype=tf.float32) ← Inferred input shape: batch size 20, flattened board size 6*7=42, one-hot depth 3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten layer expects input shaped (batch_size, 42, 3)
        # Flattening last two dimensions to feed Dense layer
        self.flatten = tf.keras.layers.Flatten(input_shape=(42, 3))
        # Dense layer outputs 7 units (e.g. Connect4 possible moves)
        self.dense = tf.keras.layers.Dense(7)
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense(x)
        return x

def my_model_function():
    # Instantiate the model and compile it with same params as in the original reproduced issue
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a one-hot encoded tensor of shape (20, 42, 3)
    # Here, 20 samples, flattened board length = 42, one-hot depth = 3
    # Randomly create integer indices [0..2] and one hot encode them as float32
    BATCH_SIZE = 20
    FLAT_SIZE = 6 * 7  # 42
    DEPTH = 3
    indices = tf.random.uniform((BATCH_SIZE, FLAT_SIZE), minval=0, maxval=DEPTH, dtype=tf.int32)
    one_hot_input = tf.one_hot(indices, depth=DEPTH, dtype=tf.float32)
    return one_hot_input

