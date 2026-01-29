# tf.random.uniform((B, 4), minval=1, maxval=366856, dtype=tf.int64)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the issue's example model architecture:
        # Embedding with very large vocab 366856 tokens
        # Embedding output dims chosen carefully due to memory:
        # We'll choose 100 as a safe embedding dimension from discussions.
        self.embedding = tf.keras.layers.Embedding(
            input_dim=366856, output_dim=100, input_length=4)
        # Bidirectional LSTM with 256 units:
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                256,
                dropout=0.5,
                recurrent_dropout=0.5,
                kernel_regularizer=tf.keras.regularizers.L2(0.001)))
        # Dense layer 128 units with relu activation:
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Final output dense with softmax activation over large output vocabulary:
        # The output vocab size in one example is huge 685731
        # We pick that size as it was in the example.
        self.output_dense = tf.keras.layers.Dense(685731, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)           # (B, 4, 100)
        x = self.bi_lstm(x, training=training)     # (B, 512) because bidirectional 256*2
        x = self.dense1(x)                   # (B, 128)
        output = self.output_dense(x)        # (B, 685731) probabilities
        return output

def my_model_function():
    # Return an instance of MyModel; weights are randomly initialized.
    return MyModel()

def GetInput():
    # Return input tensor shape (batch_size, 4)
    # Values are integer token IDs from [1, 366855]
    batch_size = 64  # Typical batch size from examples
    # The input length is 4, per embedding input_length=4
    return tf.random.uniform(
        (batch_size, 4), minval=1, maxval=366856, dtype=tf.int64)

