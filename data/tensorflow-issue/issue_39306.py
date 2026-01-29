# tf.random.uniform((B, max_seq_length), dtype=tf.int32) for input_word_ids, input_mask, segment_ids (all int32 with shape (B, max_seq_length))

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self, max_seq_length=128, num_classes=2, dropout_rate=0.0001):
        super().__init__()
        # Assumptions:
        # - Using ALBERT Lite Base model from TF Hub compatible with TF2 Lite ("tokens" signature)
        # - max_seq_length corresponds to sequence length
        # - num_classes=2 for binary classification (pos/neg sentiment)
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Instantiate the ALBERT Lite KerasLayer from TF Hub (TF2 format)
        # Using "tokens" signature and output dictionary to get pooled_output
        self.albert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/albert_lite_base/1",
            signature="tokens",
            signature_outputs_as_dict=True,
            trainable=False,  # Per issue discussion, trainable True unsupported for TF1 Hub format
            name="albert_lite"
        )
        
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.classifier = tf.keras.layers.Dense(
            self.num_classes,
            name='output',
            dtype=tf.float32
        )
    
    def call(self, inputs, training=False):
        # inputs: tuple or list of 3 tensors: (input_word_ids, input_mask, segment_ids)
        if not (isinstance(inputs, (tuple, list)) and len(inputs) == 3):
            raise ValueError("Input must be a tuple/list of 3 tensors: (input_word_ids, input_mask, segment_ids)")

        input_word_ids, input_mask, segment_ids = inputs

        # ALBERT expects input as dict of named tensors
        albert_inputs = {
            "input_ids": input_word_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids
        }
        
        # Output is a dict, get 'pooled_output' from ALBERT embeddings
        pooled_output = self.albert_layer(albert_inputs)["pooled_output"]
        
        x = self.dropout(pooled_output, training=training)
        logits = self.classifier(x)
        return logits

def my_model_function():
    # Return an instance of MyModel with default params
    return MyModel()

def GetInput(batch_size=1, max_seq_length=128):
    # Return a tuple of three input tensors:
    # input_word_ids, input_mask, segment_ids with shape [batch_size, max_seq_length]
    # dtype=tf.int32 as expected by ALBERT
    # Use random integers for input_word_ids and binary mask for input_mask and segment_ids
    # For segment_ids (token type ids), values typically 0 or 1
    
    input_word_ids = tf.random.uniform(
        shape=(batch_size, max_seq_length),
        minval=0,
        maxval=30522,  # assume vocab size approx 30k (BERT base vocab)
        dtype=tf.int32
    )
    # Random mask with 0 or 1 values; simulate realistic masks
    input_mask = tf.random.uniform(
        shape=(batch_size, max_seq_length),
        minval=0,
        maxval=2,
        dtype=tf.int32
    )
    segment_ids = tf.random.uniform(
        shape=(batch_size, max_seq_length),
        minval=0,
        maxval=2,
        dtype=tf.int32
    )
    return (input_word_ids, input_mask, segment_ids)

