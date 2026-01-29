# tf.random.uniform((BATCH_SIZE, sequence_length, 768), dtype=tf.float32)
# Note: Input is a dict with "input_1" key of shape (batch_size, 3, seq_len)
# corresponding to (input_ids, input_mask, segment_ids) for BERT.

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self,
                 nb_filters=20,
                 FFN_units=20,
                 dropout_rate=0.3,
                 name="dcnn"):
        super(MyModel, self).__init__(name=name)
        
        # Load BERT layer from TF-Hub - non-trainable as per original code
        self.bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
            trainable=False,
            output_shape=[None, 768])  # embedding size
        
        # Three Conv1D layers with different kernel sizes
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="linear")
    
    def embed_with_bert(self, inputs):
        # inputs["input_1"] has shape (batch_size, 3, seq_len)
        # The BERT layer expects a list: [input_ids, input_mask, segment_ids]
        input_ids = inputs["input_1"][:, 0, :]    # (batch, seq_len)
        input_mask = inputs["input_1"][:, 1, :]   # (batch, seq_len)
        segment_ids = inputs["input_1"][:, 2, :]  # (batch, seq_len)
        
        # BERT layer returns (pooled_output, sequence_output)
        # We want sequence_output: token embeddings (batch, seq_len, 768)
        _, sequence_output = self.bert_layer([input_ids, input_mask, segment_ids])
        return sequence_output
    
    def call(self, inputs, training=False):
        # Embed input tokens with BERT
        x = self.embed_with_bert(inputs)  # shape (batch, seq_len, 768)
        
        # Apply Conv1D layers with different kernel sizes
        x1 = self.bigram(x)
        x1 = self.pool(x1)
        x2 = self.trigram(x)
        x2 = self.pool(x2)
        x3 = self.fourgram(x)
        x3 = self.pool(x3)
        
        # Concatenate pooled features
        merged = tf.concat([x1, x2, x3], axis=-1)  # shape (batch, 3*nb_filters)
        
        # Feed forward network
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training=training)
        output = self.last_dense(merged)
        
        return output

def my_model_function():
    # Create and return model instance with default parameters matching original example
    return MyModel()

def GetInput():
    # Generate example input dictionary consistent with model expected input shapes:
    # "input_1": int32 tensor of shape (batch_size=32, 3, seq_len=50) with token ids, masks, segments
    # "input_2" is unused in model call, omitted here.
    BATCH_SIZE = 32
    SEQ_LEN = 50  # Assumed fixed sequence length (padding truncated or padded to length 50)

    # We create dummy inputs:
    # input_ids - random ints in vocabulary range (assumed vocab size ~30522 for BERT)
    vocab_size = 30522
    input_ids = tf.random.uniform((BATCH_SIZE, SEQ_LEN), minval=0, maxval=vocab_size, dtype=tf.int32)
    # input_mask - 1s (all tokens attended)
    input_mask = tf.ones_like(input_ids, dtype=tf.int32)
    # segment_ids - zeros (single segment)
    segment_ids = tf.zeros_like(input_ids, dtype=tf.int32)

    # Stack to shape (batch_size, 3, seq_len)
    input_1 = tf.stack([input_ids, input_mask, segment_ids], axis=1)

    return {"input_1": input_1}

# The code is compatible with TensorFlow 2.20.0 XLA compilation.
# Example:
# @tf.function(jit_compile=True)
# def compiled(x):
#     model = my_model_function()
#     return model(x)

