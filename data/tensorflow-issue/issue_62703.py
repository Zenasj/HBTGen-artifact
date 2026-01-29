# tf.random.uniform((B, T, 1), dtype=tf.float32) ← Input shape inferred from X_train.shape[1:] where T = sequence length, 1 feature dimension

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, neurons=128, l2_weight=0.001, dropout_rate=0):
        super().__init__()
        # Using Bidirectional LSTM layers similar to the Sequential model from the issue
        # Regularization with l2, dropout layers included
        self.bilstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(neurons, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_weight))
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.bilstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(neurons, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_weight))
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.bilstm3 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(neurons, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_weight))
        )
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.bilstm4 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(neurons, kernel_regularizer=tf.keras.regularizers.l2(l2_weight))
        )
        self.dropout4 = tf.keras.layers.Dropout(dropout_rate)

        self.dense = tf.keras.layers.Dense(units=10)  # Placeholder units, adjust after input shape or label classes known
        self.activation = tf.keras.layers.Activation('softmax')

    def build(self, input_shape):
        # The number of classes for the output Dense layer must be inferred from input data
        # But since build is called with input_shape, we can't get labels here.
        # To fully conform, let's set dense units to 10 for demonstration,
        # user can alter as needed based on y_train shape.
        # This is a compromise due to lack of exact num classes.
        self.dense.units = 10
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.bilstm1(inputs, training=training)
        x = self.dropout1(x, training=training)

        x = self.bilstm2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.bilstm3(x, training=training)
        x = self.dropout3(x, training=training)

        x = self.bilstm4(x, training=training)
        x = self.dropout4(x, training=training)

        x = self.dense(x)
        x = self.activation(x)
        return x


def my_model_function():
    # Construct model with default parameters from the issue
    # Here, we default to 10 output classes as a plausible value;
    # in practice, set it to the number of your labels.
    return MyModel(neurons=128, l2_weight=0.001, dropout_rate=0)


def GetInput():
    # From the issue, input shape is (batch_size, seq_len, 1 feature)
    # We do not have exact sequence length or batch size here,
    # so choose a plausible example: batch=32, seq_len=100 (common for RNN),
    # feature=1, dtype float32 as typical for Raman spectra or sequential data.
    batch_size = 32
    seq_len = 100  # Assumed sequence length; adjust if known
    feature_dim = 1
    return tf.random.uniform((batch_size, seq_len, feature_dim), dtype=tf.float32)

