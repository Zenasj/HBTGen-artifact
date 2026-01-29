# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape inferred from usage: (batch_size, 7, 15, 4)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.units = 32
        self.positional_embedding = PositionalEmbedding(self.units, dropout_rate=0.1)

    def call(self, inputs, training=False):
        return self.positional_embedding(inputs, training=training)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        self.units = units

        self.projection = tf.keras.layers.Dense(
            units, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=False):
        # inputs shape: (batch, 7, 15, 4) per the example
        # Slice channels for projection and positional encoding as per original code logic
        # inputs[:, :, :, 3:] shape: (batch, 7, 15, 1)
        x = self.projection(inputs[:, :, :, 3:])  # Project last channel(s)

        A = inputs[:, :, :, 2]  # shape: (batch, 7, 15)
        # Add positional encoding for A
        x = x + self.positional_encoding(A, self.units)

        B = inputs[:, :, :, 0]
        C = inputs[:, :, :, 1]
        x = x + self.positional_encoding(B, self.units)
        x = x + self.positional_encoding(C, self.units)

        return self.dropout(x, training=training)

    def positional_encoding(self, position, d_model, n=10000):
        # position shape: (batch, 7, 15)
        # We expand dims so angle_rads shape: (batch, 7, 15, d_model)
        position = tf.cast(position, tf.float32)
        angle_rates = 1 / tf.pow(
            n,
            (2 * (tf.range(d_model, dtype=tf.float32) // 2)) / d_model
        )
        # Broadcast multiply: (batch, 7, 15, 1) * (d_model,)
        angle_rads = tf.expand_dims(position, -1) * angle_rates

        # Apply sin to even indices in the array; 2i
        # and cos to odd indices; 2i+1
        # TensorFlow tensors don't support direct assignment, so we split and recombine:
        sines = tf.sin(angle_rads[:, :, :, 0::2])
        cosines = tf.cos(angle_rads[:, :, :, 1::2])

        # Now interleave sines and cosines back into the last dimension
        # One way: create a tensor of zeros and fill slices:
        pos_encoding = tf.TensorArray(dtype=tf.float32, size=d_model)

        # Fill even indices
        for i in range(0, d_model, 2):
            pos_encoding = pos_encoding.write(i, sines[:, :, :, i//2])
        # Fill odd indices
        for i in range(1, d_model, 2):
            pos_encoding = pos_encoding.write(i, cosines[:, :, :, (i-1)//2])

        pos_encoding = pos_encoding.stack()  # shape: (d_model, batch, 7, 15)
        pos_encoding = tf.transpose(pos_encoding, perm=[1, 2, 3, 0])  # (batch, 7, 15, d_model)

        return pos_encoding


def my_model_function():
    # Create and return the model instance
    return MyModel()


def GetInput():
    # Return a random tensor with shape (batch_size=8, height=7, width=15, channels=4)
    # dtype float32 to match tf.random.normal used in code example
    return tf.random.uniform((8, 7, 15, 4), dtype=tf.float32)

