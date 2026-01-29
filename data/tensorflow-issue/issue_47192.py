# tf.random.uniform((B, T, 1), dtype=tf.float32) ← Input shape: batch size B, sequence length T, 1 feature channel

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Bidirectional LSTM with 10 units as per original issue example
        self.rec_net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10))

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # Input sequences
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32)     # Mask in int32 (1 or 0)
    ])
    def call(self, x, mask):
        # Cast mask to boolean as expected by Keras layers
        bool_mask = tf.cast(mask, tf.bool)
        # Use the mask in the bidirectional LSTM call
        return self.rec_net(x, mask=bool_mask, training=False)

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Create a random input tensor with shape (3, 4, 1), batch 3, sequence 4, feature 1
    # This matches the example input in the issue reproducing code
    x = tf.random.uniform((3, 4, 1), dtype=tf.float32)
    # Construct a mask tensor with shape (3, 4, 1) with values 1 or 0 as int32
    mask = tf.convert_to_tensor(
        [
            [[1], [1], [1], [1]],
            [[1], [1], [1], [0]],
            [[1], [1], [0], [0]],
        ],
        dtype=tf.int32
    )
    # Return inputs as a tuple matching the model call signature
    return (x, mask)

