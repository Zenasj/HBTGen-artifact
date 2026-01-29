# tf.random.uniform((batch_size, 256), dtype=tf.int64) and tf.random.uniform((batch_size, 256), dtype=tf.int64)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layers for two inputs
        self.emb_a = tf.keras.layers.Embedding(
            input_dim=20000,
            output_dim=128,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
        self.emb_b = tf.keras.layers.Embedding(
            input_dim=20000,
            output_dim=128,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
        # Conv1D layers
        self.conv_layers = [
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu") for _ in range(5)
        ]
        # Pooling and Dense layers
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name="output")

    def call(self, inputs, training=False):
        input_a, input_b = inputs["input_a"], inputs["input_b"]
        # Pass through embeddings
        emb_out_a = self.emb_a(input_a)
        emb_out_b = self.emb_b(input_b)
        x = emb_out_a + emb_out_b
        # Pass through conv layers
        for conv in self.conv_layers:
            x = conv(x)
        # Global max pooling and dense layers
        x = self.global_max_pool(x)
        x = self.dense1(x)
        output = self.output_layer(x)
        return output


def my_model_function():
    # Instantiate and return MyModel instance
    return MyModel()


def GetInput():
    # Provide random valid input dict pair for MyModel
    # Assumptions:
    # - batch size 32 for demonstration (model can work with any batch size)
    # - input_a and input_b expect integers in [0, 20000) with shape (batch_size, 256)
    batch_size = 32
    input_a = tf.random.uniform(
        shape=(batch_size, 256), minval=0, maxval=20000, dtype=tf.int64
    )
    input_b = tf.random.uniform(
        shape=(batch_size, 256), minval=0, maxval=20000, dtype=tf.int64
    )
    return {"input_a": input_a, "input_b": input_b}

