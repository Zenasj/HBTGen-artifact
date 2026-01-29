# tf.random.uniform((batch_size, sequence_len, num_feature), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumptions based on issue: input shape (sequence_len, num_feature) but sequence_len and num_feature unknown
        # We'll use placeholder constants for their values to allow model construction and test
        self.sequence_len = 20      # Assumed sequence length
        self.num_feature = 10       # Assumed feature dimensionality

        # Layers as described in the issue code snippet
        self.masking = tf.keras.layers.Masking(mask_value=0., input_shape=(self.sequence_len, self.num_feature))
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=50, return_sequences=True), name='BiLSTM-1'
        )
        self.dense = tf.keras.layers.Dense(units=3, activation='softmax', name='Softmax')

    def call(self, inputs, training=False):
        x = self.masking(inputs)  # Masking zeros
        x = self.bilstm(x)
        out = self.dense(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching model input shape: (batch_size, sequence_len, num_feature)
    # The batch size can be arbitrary; we pick 32 here
    batch_size = 32
    sequence_len = 20  # Should match the model's assumption above
    num_feature = 10   # Should match the model's assumption above

    # Generate uniform random float32 tensor as plausible input sequence data
    # To avoid triggering the masking layer masking entire sequence (all zeros),
    # input will contain some nonzero values
    import numpy as np
    # generate random floats from 0 to 1, shape (batch_size, sequence_len, num_feature)
    # ensure not all zeros to avoid the Masking layer masking all input too heavily
    data = np.random.uniform(low=0.1, high=1.0, size=(batch_size, sequence_len, num_feature)).astype('float32')
    return tf.convert_to_tensor(data)

