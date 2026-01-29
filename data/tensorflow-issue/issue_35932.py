# tf.random.uniform((B, vector_len), dtype=tf.int32) x 3 (input_word_ids, input_masks, segment_ids)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, vector_len=200, pre_train_handle=None):
        super().__init__()
        self.vector_len = vector_len

        # Load pre-trained BERT layer from TF Hub, trainable
        # In the original code, this is hub.KerasLayer(pre_train_handle, trainable=True)
        # Here we simulate with a placeholder layer to enable compilation and input signatures.
        # We will assume the loaded layer expects three int32 inputs of shape (batch, vector_len)
        # and returns (pooled_output, sequence_output).
        if pre_train_handle is not None:
            import tensorflow_hub as hub
            self.pre_train_layer = hub.KerasLayer(pre_train_handle, trainable=True, name="bert_layer")
        else:
            # Placeholder "mock" for environments without TF Hub, simulate output shape
            self.pre_train_layer = tf.keras.layers.Lambda(
                lambda x: (tf.zeros((tf.shape(x[0])[0], 768)), tf.zeros((tf.shape(x[0])[0], self.vector_len, 768))),
                name="bert_layer_mock"
            )

        # Classifier Dense layer on top of pooled output (BERT CLS token)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        """
        inputs: list or tuple of three int32 tensors
          - input_word_ids: (B, vector_len)
          - input_masks: (B, vector_len)
          - segment_ids: (B, vector_len)
        """
        input_word_ids, input_masks, segment_ids = inputs

        # BERT layer returns tuple: (pooled_output, sequence_output)
        pooled_output, sequence_output = self.pre_train_layer([input_word_ids, input_masks, segment_ids], training=training)

        # Use pooled_output or CLS token representation for classification
        # pooled_output shape: (batch, hidden_size = 768)
        logits = self.classifier(pooled_output)
        return logits

def my_model_function():
    # Using typical vector_len for BERT input sequences (e.g. 128 or 200)
    # Provide official TF Hub handle for BERT base uncased
    pre_train_handle = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    model = MyModel(vector_len=128, pre_train_handle=pre_train_handle)

    # Compile model with optimizer, loss - same as original ModelBERT.compile setup
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Prepare dummy input tensors with:
    # shape: (batch_size=2, vector_len=128)
    batch_size = 2
    vector_len = 128

    # inputs are int32 tokens and masks, segment IDs, so we generate uniform ints in range [0, 30522),
    # the typical vocabulary size for uncased BERT vocab
    vocab_size = 30522

    input_word_ids = tf.random.uniform(shape=(batch_size, vector_len), minval=0, maxval=vocab_size, dtype=tf.int32)
    input_masks = tf.ones(shape=(batch_size, vector_len), dtype=tf.int32)  # Mask typically 1s for actual tokens
    segment_ids = tf.zeros(shape=(batch_size, vector_len), dtype=tf.int32) # Segment IDs, 0 for single sequence

    return (input_word_ids, input_masks, segment_ids)

