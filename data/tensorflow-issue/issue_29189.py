# tf.random.uniform((BATCH_SIZE, 3), dtype=tf.int64)
import tensorflow as tf

BATCH_SIZE = 64
MAX_SEQ_LEN = 3
VOCAB_SIZE = 100
EMBED_SIZE = 16
STATE_SIZE = 7

class MyModel(tf.keras.Model):
    """
    A Keras Model replicating the reported architecture:
    Embedding -> LSTM -> Dense with softmax activation.
    
    Due to the original issue described where tf.keras.layers.LSTM
    misbehaved with tf.distribute strategies (graph placement errors 
    due to cross-device reference edges), the recommended workaround was to replace
    tf.keras.layers.LSTM with tf.keras.layers.RNN wrapping a tf.keras.layers.LSTMCell.
    
    This model integrates that workaround by using RNN(LSTMCell).
    It is expected to be compatible with MirroredStrategy and XLA compiled functions.
    """
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE)
        # Use RNN + LSTMCell instead of direct LSTM layer to avoid device placement errors.
        self.lstm = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(STATE_SIZE),
            return_sequences=True
        )
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return x


def my_model_function():
    """
    Instantiate the MyModel.
    No pretrained weights or special initialization needed.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the expected input of the model:
    Shape: (BATCH_SIZE, MAX_SEQ_LEN) with dtype tf.int64.
    Values in range [0, VOCAB_SIZE).
    """
    # Generate random integers for token indices
    return tf.random.uniform(
        shape=(BATCH_SIZE, MAX_SEQ_LEN),
        minval=0,
        maxval=VOCAB_SIZE,
        dtype=tf.int64
    )

