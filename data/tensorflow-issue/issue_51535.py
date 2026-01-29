# tf.random.uniform((B, 512), dtype=tf.int32) ← Input shape inferred from max_seq_length=512 and input_word_ids dtype=int32

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.max_seq_length = 512

        # Load the BERT hub layer, trainable
        # Note: the BERT hub URL and signature returns a tuple (pooled_output, sequence_output)
        self.bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", 
            trainable=True,
            name="bert_hub_layer"
        )
        self.dropout = tf.keras.layers.Dropout(0.3)
        # Final output layer for binary classification with sigmoid activation
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid", name="output")

    def call(self, inputs, training=False):
        # Inputs is expected as a dict with keys:
        # 'input_word_ids', 'input_mask', 'input_type_ids' each of shape (batch, 512)
        input_word_ids = inputs["input_word_ids"]
        input_mask = inputs["input_mask"]
        input_type_ids = inputs["input_type_ids"]

        # The BERT layer returns (pooled_output, sequence_output)
        pooled_output, _ = self.bert_layer([input_word_ids, input_mask, input_type_ids])
        x = self.dropout(pooled_output, training=training)
        output = self.classifier(x)
        return output


def my_model_function():
    # Return an instantiated MyModel
    return MyModel()


def GetInput():
    # Generate a dictionary input with random integers to mimic BERT input tensors
    # Each input is (batch_size, max_seq_length), dtype=tf.int32
    B = 2  # batch size, small number for testing
    max_seq_length = 512

    # BERT input_word_ids: integer token ids
    input_word_ids = tf.random.uniform(
        shape=(B, max_seq_length), minval=0, maxval=30522, dtype=tf.int32
    )
    # input_mask: 0 or 1 mask to mark real tokens (simulate all ones here)
    input_mask = tf.ones(shape=(B, max_seq_length), dtype=tf.int32)
    # input_type_ids: segment IDs, usually 0 or 1
    input_type_ids = tf.zeros(shape=(B, max_seq_length), dtype=tf.int32)

    return {
        "input_word_ids": input_word_ids,
        "input_mask": input_mask,
        "input_type_ids": input_type_ids,
    }

