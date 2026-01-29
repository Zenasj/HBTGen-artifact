# tf.random.uniform((B, T), dtype=tf.int32) ← Here, B=batch size, T=sequence length (variable)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=10000, embedding_size=256, lstm_size=64):
        super(MyModel, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)

    # Warning/Note:
    # The original issue arises from using multiple inputs with nested tuples in input_signature,
    # which causes TensorFlow to expect flattened input vs nested input and triggers mismatch.
    #
    # Workaround:
    # Use a single input argument that is a list or tuple of tensors, and correspondingly adjust
    # the input_signature to accept a single argument of type list/tuple with TensorSpecs inside.
    #
    # This approach neatly packages multiple inputs in one container input, matching structure.
    #
    # Here, `call` accepts a *single* argument: a tuple (sequence, states), 
    # where states itself is a tuple of two tensors.
    # For input_signature, we specify one argument: a tuple of specs mirroring input structure.

    @tf.function(input_signature=[
        (
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='sequence'),
            (
                tf.TensorSpec(shape=[None, 64], dtype=tf.float32, name='states_1'),
                tf.TensorSpec(shape=[None, 64], dtype=tf.float32, name='states_2')
            )
        )
    ])
    def call(self, inputs):
        # inputs is a tuple: (sequence, states_tuple)
        # This matches the input_signature expecting a single tuple argument holding required tensors.

        # Unpack inputs
        sequence, states = inputs

        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))


def my_model_function():
    # Build and return instance with default params
    return MyModel()

def GetInput():
    # Return a valid random input matching expected input_signature and the call argument structure.
    # According to input_signature:
    # inputs: tuple(
    #    sequence: [batch_size, sequence_length] int32 tensor
    #    states: tuple of two tensors [batch_size, 64] float32 each
    # )
    batch_size = 4        # arbitrary
    sequence_length = 10  # arbitrary

    sequence = tf.random.uniform(
        shape=(batch_size, sequence_length), minval=0, maxval=9999, dtype=tf.int32)

    states_1 = tf.zeros([batch_size, 64], dtype=tf.float32)
    states_2 = tf.zeros([batch_size, 64], dtype=tf.float32)

    states = (states_1, states_2)

    return (sequence, states)

