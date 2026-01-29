# tf.random.uniform((B, 2), dtype=tf.float32)  # B = batch size dynamic, input shape (None, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate the model architecture from the issue:
        # Dense(5) with kernel_initializer='uniform', input shape (2,), batch size fixed to 1 during compilation,
        # BatchNormalization, PReLU, Dropout(0.5), Dense(1) with kernel_initializer='normal'
        # To support dynamic batch size, we omit fixed batch_size here; rely on dynamic batching at runtime.

        self.dense_1 = tf.keras.layers.Dense(
            5, kernel_initializer='uniform', input_shape=(2,), name='dense_1')
        self.batch_norm = tf.keras.layers.BatchNormalization(name='batch_normalization_1')
        self.prelu = tf.keras.layers.PReLU(name='p_re_lu_1')
        self.dropout = tf.keras.layers.Dropout(0.5, name='dropout_1')
        self.dense_2 = tf.keras.layers.Dense(
            1, kernel_initializer='normal', name='dense_2')

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        x = self.batch_norm(x, training=training)
        x = self.prelu(x)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)
        return x

def my_model_function():
    # Create and return an instance of MyModel.
    # We do not load weights here since original weights aren't provided in the issue,
    # but in practice weights could be loaded from a saved h5 or checkpoint.
    return MyModel()

def GetInput():
    # Return a random input tensor matching model input shape: (batch_size, 2)
    #
    # Due to the discussion around batch fixed to 1 during compilation,
    # but want to allow dynamic batch for normal TF usage,
    # we choose batch size = 1 as the simplest working example.
    #
    # The input dtype should be float32 as indicated by usage in numpy arrays.

    # Generate a batch of one 2-element vector with uniform random float32 values:
    return tf.random.uniform((1, 2), dtype=tf.float32)

