# tf.random.uniform((100, 50, 50, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input shape: (50, 50, 1)
        # Since Conv2D does not support SparseTensor inputs directly,
        # we convert sparse input to dense first as a workaround.

        # Lambda layer to convert SparseTensor input to dense tensor
        self.sparse_to_dense = tf.keras.layers.Lambda(lambda x: tf.sparse.to_dense(x, validate_indices=False))

        # CNN model layers (replicating the Sequential model from the issue)
        self.conv1 = tf.keras.layers.Conv2D(10, kernel_size=(10, 10), activation="relu", input_shape=(50, 50, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(5, kernel_size=(5, 5), activation="relu")
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(50, activation="relu")
        self.dense2 = tf.keras.layers.Dense(2, activation="relu")

    def call(self, inputs, training=False):
        # Inputs expected as SparseTensor with shape (batch, 50, 50, 1)
        # Convert SparseTensor to dense
        x = self.sparse_to_dense(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def my_model_function():
    return MyModel()


def GetInput():
    # Generate a random sparse tensor with shape (100, 50, 50, 1)
    # to mimic the input used in the original issue.

    batch_size = 100
    height = 50
    width = 50
    channels = 1

    # For demo: randomly generate sparse indices
    # Fill at most 60 pixels per image same as original issue
    import numpy as np

    indices = []
    values = []

    for i in range(batch_size):
        n_fill = np.random.randint(1, 61)
        seen_positions = set()
        for _ in range(n_fill):
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            pos = (row, col)
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            indices.append([i, row, col, 0])
            values.append(np.random.rand())

    dense_shape = [batch_size, height, width, channels]
    indices_tf = tf.constant(indices, dtype=tf.int64)
    values_tf = tf.constant(values, dtype=tf.float32)
    sparse_tensor = tf.sparse.reorder(tf.sparse.SparseTensor(indices=indices_tf, values=values_tf, dense_shape=dense_shape))

    return sparse_tensor

