# tf.random.uniform((B, 512), dtype=tf.float32) ← Inferred input shape based on Universal Sentence Encoder outputs (512 dim embeddings)
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the same model architecture as in standalone and distributed codes
        self.dense1 = keras.layers.Dense(units=256, activation='relu', input_shape=(512,))
        self.dropout1 = keras.layers.Dropout(rate=0.5)
        self.dense2 = keras.layers.Dense(units=128, activation='relu')
        self.dropout2 = keras.layers.Dropout(rate=0.5)
        self.output_layer = keras.layers.Dense(units=3, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)


def my_model_function():
    # Return an instance of MyModel with weights randomly initialized
    return MyModel()


def GetInput():
    """
    Produce a batch of random input embeddings to simulate Universal Sentence Encoder outputs.
    Since the example dataset had 3 classes and messages batch size varied between 3 and 4,
    return a batch size of 4 with 512-d embedding vectors.
    """
    batch_size = 4
    emb_dim = 512
    # Use uniform distribution as placeholder input, dtype float32
    return tf.random.uniform((batch_size, emb_dim), dtype=tf.float32)

