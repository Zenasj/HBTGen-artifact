# tf.random.uniform((B,), dtype=tf.string) ← Input is a 1D batch of strings (text samples)

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # TensorFlow Hub embedding layer for English text, not trainable
        self.embed = hub.KerasLayer(
            "https://tfhub.dev/google/nnlm-en-dim50/1",
            output_shape=[50],
            input_shape=[],
            dtype=tf.string,
            trainable=False,
            name='embedding_layer'
        )
        # Simple classifier layers on top of the embedding
        self.dense1 = tf.keras.layers.Dense(16, activation='relu', name='dense_relu')
        self.dense2 = tf.keras.layers.Dense(3, activation='softmax', name='output')

    def call(self, inputs, training=False):
        # inputs: batch of strings (shape [B])
        x = self.embed(inputs)  # shape [B, 50]
        x = self.dense1(x)      # shape [B, 16]
        x = self.dense2(x)      # shape [B, 3] softmax output over 3 classes
        return x

def my_model_function():
    # Return an instance of MyModel with the pretrained tensorflow hub embed layer
    return MyModel()

def GetInput():
    # Generate a random batch of strings to feed the model
    # For demo/testing, use random ascii strings
    batch_size = 8  # Example batch size
    import numpy as np

    # Create random ASCII strings of length between 5 and 20 characters
    def random_ascii_string(length):
        # Generate random bytes in the ascii letter range (32-126)
        ascii_bytes = np.random.randint(32, 127, size=length).astype('uint8')
        return ascii_bytes.tobytes().decode('ascii')

    text_batch = [random_ascii_string(np.random.randint(5, 20)) for _ in range(batch_size)]
    # Convert to tensor of dtype string
    text_tensor = tf.constant(text_batch, dtype=tf.string)
    return text_tensor

