# tf.random.uniform((B,), dtype=tf.string) ← Input is a batch of strings (text inputs)

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self, seq_length=512):
        super().__init__()
        # Load preprocessing model from TF-Hub
        self.preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        # Tokenize layer as KerasLayer wrapping preprocessor tokenization function
        self.tokenize = hub.KerasLayer(self.preprocessor.tokenize)
        # Pack inputs layer wraps preprocessor's bert_pack_inputs function with seq_length argument
        self.bert_pack_inputs = hub.KerasLayer(
            self.preprocessor.bert_pack_inputs,
            arguments={"seq_length": seq_length}
        )
        # Load BERT encoder from TF-Hub, trainable
        self.encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1",
            trainable=True
        )
        # Final dense layer for binary classification with sigmoid activation
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        # inputs: tensor of shape (batch_size,) of dtype string
        tokenized = self.tokenize(inputs)  # tokenizes each string to RaggedTensor
        encoder_inputs = self.bert_pack_inputs([tokenized])  # pack tokens with padding/truncation
        encoder_outputs = self.encoder(encoder_inputs)  # dictionary with keys like 'pooled_output'
        pooled_output = encoder_outputs["pooled_output"]  # shape (batch_size, H)
        output = self.output_layer(pooled_output)  # (batch_size, 1) sigmoid output
        return output

def my_model_function():
    return MyModel()

def GetInput():
    # Return a batch of random dummy strings as input
    # To simulate, just make batch of short ASCII strings
    # e.g., tf.constant(["Hello world", "This is a test", "tensorflow bert"])
    batch_size = 4
    dummy_texts = [
        "tensorflow is great",
        "this is a sample text input",
        "small bert model test",
        "another example sentence"
    ]
    return tf.constant(dummy_texts[:batch_size])

