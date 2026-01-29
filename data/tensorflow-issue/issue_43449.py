# tf.random.uniform((B, SEQ_LEN), dtype=tf.int32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, maxfeature=100, seq_len=32):
        super(MyModel, self).__init__()
        # Embedding input_dim and output_dim as per example
        self.embed = tf.keras.layers.Embedding(input_dim=maxfeature, output_dim=64)
        # Bidirectional LSTM with 64 units each direction
        self.blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(2)
    
    @tf.function
    def call(self, x, training=False):
        """
        Forward pass:
        x: int32 tensor shaped (batch_size, seq_len)
        Returns log softmax logits (batch_size, 2)
        """
        x = self.embed(x)
        x = self.blstm(x)
        x = self.d1(x)
        x = self.d2(x)
        return tf.nn.log_softmax(x)
        

def my_model_function():
    # Instantiate MyModel with default maxfeature=100 and seq_len=32
    model = MyModel()
    # Normally model variables get created on first call, but to ensure they're created,
    # we provide a dummy call with a batch of zeros of shape (1, seq_len)
    dummy_input = tf.constant([[0]*32], dtype=tf.int32)
    model(dummy_input)
    return model

def GetInput():
    """
    Generates a random input tensor simulating padded sequences.
    Shape (64, 32) as per batch_size=64, seq_len=32
    Values are int32 in range [0, maxfeature-1] (100 assumed)
    """
    batch_size = 64
    seq_len = 32
    maxfeature = 100
    # Generate random integers in [0, maxfeature) shape (batch_size, seq_len)
    return tf.random.uniform(shape=(batch_size, seq_len), minval=0, maxval=maxfeature, dtype=tf.int32)

