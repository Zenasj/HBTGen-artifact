# tf.random.uniform((B, MAX_LEN), dtype=tf.int32)  # Input shape inferred from input_ids and attention_mask shape=(MAX_LEN,)

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import losses, metrics, optimizers

# Note: The original issue refers to TFRobertaModel from HuggingFace transformers library,
# which is not included here. We'll create a minimal stub to simulate the "base_model" output
# with the correct shape and dtype for demonstration purposes.

class DummyBaseModel(tf.keras.layers.Layer):
    """A dummy base model simulating the output of a transformer encoder (e.g. RoBERTa).

    It receives inputs of shape (batch_size, seq_len) and outputs a tensor of shape
    (batch_size, seq_len, hidden_size).
    """
    def __init__(self, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

    def call(self, inputs):
        # inputs is expected to be a dict with keys: 'input_ids' and 'attention_mask'
        input_ids = inputs['input_ids']  # shape (B, MAX_LEN)
        batch_size = tf.shape(input_ids)[0]
        seq_len = tf.shape(input_ids)[1]

        # Output a dummy tensor, e.g. random uniform, simulating hidden states
        return tf.random.uniform((batch_size, seq_len, self.hidden_size), dtype=tf.float32), None

class MyModel(tf.keras.Model):
    def __init__(self, max_len=512, lstm_units=128, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

        # Instead of real TFRobertaModel, use dummy base_model here for standalone code
        self.base_model = DummyBaseModel(hidden_size=768, name="base_model")

        # Bi-directional LSTM on top of base model output
        self.bi_lstm = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True), name="bidirectional_lstm")
        self.dropout = layers.Dropout(self.dropout_rate)

        # TimeDistributed Dense layers for start and end logits
        self.dense_start = layers.TimeDistributed(layers.Dense(1), name="time_distributed_start")
        self.flatten_start = layers.Flatten(name="flatten_start")
        self.softmax_start = layers.Activation('softmax', name='y_start')

        self.dense_end = layers.TimeDistributed(layers.Dense(1), name="time_distributed_end")
        self.flatten_end = layers.Flatten(name="flatten_end")
        self.softmax_end = layers.Activation('softmax', name='y_end')

    def call(self, inputs, training=False):
        # inputs is a dict with keys: 'input_ids' and 'attention_mask'
        base_out, _ = self.base_model({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']})
        # base_out shape: (batch_size, seq_len, hidden_size)

        x = self.bi_lstm(base_out)
        if training:
            x = self.dropout(x, training=training)

        # Start position output
        x_start = self.dense_start(x)  # (batch_size, seq_len, 1)
        x_start = self.flatten_start(x_start)  # (batch_size, seq_len)
        y_start = self.softmax_start(x_start)  # (batch_size, seq_len)

        # End position output
        x_end = self.dense_end(x)  # (batch_size, seq_len, 1)
        x_end = self.flatten_end(x_end)  # (batch_size, seq_len)
        y_end = self.softmax_end(x_end)  # (batch_size, seq_len)

        return y_start, y_end

def my_model_function():
    # Return an instance of the model with default max_len=512 as typical for RoBERTa
    return MyModel(max_len=512, lstm_units=128, dropout_rate=0.1)

def GetInput():
    # Generate a random input matching the model's expected input:
    # inputs dict with:
    #  - 'input_ids': tf.int32 tensor of shape (batch_size, max_len)
    #  - 'attention_mask': tf.int32 tensor of shape (batch_size, max_len)

    batch_size = 2
    max_len = 512

    input_ids = tf.random.uniform(
        shape=(batch_size, max_len),
        minval=0,
        maxval=30522,  # typical vocab size for RoBERTa
        dtype=tf.int32
    )
    attention_mask = tf.ones_like(input_ids, dtype=tf.int32)  # simulate all tokens attended

    return {'input_ids': input_ids, 'attention_mask': attention_mask}

