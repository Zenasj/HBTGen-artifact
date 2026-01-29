# tf.random.uniform((BATCH_SIZE, 3, 5), dtype=tf.float32) ← inferred input shape from INPUT_SHAPE=[3,5] and batch size 7

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The core issue discussed in the issue is shape mismatch from dense layers without LSTM.
        # The "working" version adds a small LSTM layer between Dense layers to produce correct shapes.
        #
        # INPUT_SHAPE = [3,5]: sequence length=3, feature dims=5
        # The model expects inputs of shape (batch, 3, 5).
        #
        # The user showed:
        #    Dense(100, activation='tanh', input_shape=(3, 5))
        #    LSTM(1, activation='tanh', return_sequences=False)
        #    Dense(3, activation='softmax')
        #
        # which gives output shape (batch, 3)
        # and works correctly for SparseCategoricalCrossentropy with labels shape (batch,)
        #
        # We replicate this model here.

        self.dense1 = tf.keras.layers.Dense(100, activation="tanh", input_shape=(3,5))
        self.lstm = tf.keras.layers.LSTM(1, activation="tanh", return_sequences=False)
        self.dense2 = tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.001))

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.lstm(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate input of correct shape and dtype to feed into MyModel
    # batch size is 7 as per issue examples, input shape [3,5]
    B = 7
    H = 3
    W = 5
    # Input dtype float32 as stated in issue and code
    return tf.random.uniform((B, H, W), dtype=tf.float32)

