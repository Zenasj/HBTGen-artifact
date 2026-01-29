# tf.random.uniform((B, T), dtype=tf.int32) ← Input shape is (batch_size, variable_seq_len) integer token IDs for embedding lookup
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding for vocab size 6, embedding dim 64
        self.embedding = tf.keras.layers.Embedding(input_dim=6, output_dim=64, mask_zero=True)
        # Bidirectional LSTM with return_sequences=True (fixed based on issue context)
        lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        self.bidir_lstm = tf.keras.layers.Bidirectional(lstm)
        # Dense output layer producing single logit value
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        """
        inputs: Tensor of shape (batch_size, seq_len), dtype int32 or int64
        Returns tensor with shape (batch_size, 1) as logit prediction.
        """
        x = self.embedding(inputs)     # (B, T, 64)
        x = self.bidir_lstm(x)         # (B, T, 128) since bidirectional doubles units
        # Use last timestep output for prediction
        x = self.dense(x[:, -1, :])    # (B, 1)
        return x

def my_model_function():
    """
    Initializes and returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor compatible with MyModel.
    Shape: (batch_size=2, seq_len=5) with integer values in [1,5].
    tf.keras.layers.Embedding uses input_dim=6 (0 reserved for mask).
    """
    batch_size = 2
    seq_len = 5
    # Random integers from 1 to 5 inclusive, avoiding 0 since it's mask token
    input_tensor = tf.random.uniform(shape=(batch_size, seq_len), minval=1, maxval=6, dtype=tf.int32)
    return input_tensor

