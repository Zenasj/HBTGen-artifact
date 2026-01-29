# tf.random.uniform((4, 16), dtype=tf.int32) ← Batch size = 4, sequence length = 16 (matching input_word_ids shape)

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load ALBERT base model from TF Hub, trainable=True as in original snippet
        # Note: The original URL was "https://tfhub.dev/tensorflow/albert_en_base/1"
        self.albert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/albert_en_base/1",
            trainable=True,
            name='albert_layer'
        )
        # Output dense layer projecting sequence output to 2 classes per token
        self.output_dense = layers.Dense(2)
    
    def call(self, inputs, training=False):
        # inputs is expected to be a dict with keys:
        # 'input_word_ids', 'input_mask', 'segment_ids'
        input_word_ids = inputs['input_word_ids']
        input_mask = inputs['input_mask']
        segment_ids = inputs['segment_ids']
        
        # ALBERT layer returns pooled_output and sequence_output
        pooled_output, sequence_output = self.albert_layer([input_word_ids, input_mask, segment_ids])
        output = self.output_dense(sequence_output)  # Shape: (batch_size, seq_len, 2)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a dict of inputs matching model expectation
    batch_size = 4
    max_seq_length = 16
    
    # Create dummy inputs
    input_word_ids = tf.constant(5, shape=(batch_size, max_seq_length), dtype=tf.int32)
    input_mask = tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32)
    segment_ids = tf.zeros(shape=(batch_size, max_seq_length), dtype=tf.int32)
    
    return {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids
    }

