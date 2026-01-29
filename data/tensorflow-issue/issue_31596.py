# tf.random.uniform((None, 25), dtype=tf.int64) ← Input shape inferred from input_signature in SimpleModel.call

import tensorflow as tf

class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        # Initialize the embedding weights as a trainable variable
        self.shared_weights = self.add_weight(
            "weights",
            shape=(self.vocab_size, self.hidden_size),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                mean=0.0,
                stddev=self.hidden_size ** (-0.5)
            )
        )

    def call(self, input_):
        # Use tf.gather on the layer attribute variable self.shared_weights
        return tf.gather(self.shared_weights, input_)


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=20000, hidden_size=300):
        super(MyModel, self).__init__()
        self.embedding_layer = Embedding(vocab_size, hidden_size)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 25), dtype=tf.int64, name='input')])
    def call(self, input_):
        # Forward pass through embedding layer
        return self.embedding_layer(input_)


def my_model_function():
    # Return an instance of the MyModel
    # Uses default vocab_size=20,000 and hidden_size=300 as from the original example
    return MyModel()


def GetInput():
    # Generate a random integer tensor of shape (batch_size=10, sequence_length=25) 
    # of dtype int64, with values in [0, 100)
    return tf.random.uniform(shape=(10, 25), dtype=tf.int64, maxval=100)

