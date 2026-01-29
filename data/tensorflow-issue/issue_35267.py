# tf.random.uniform((B, 10), dtype=tf.float32), tf.random.uniform((B, 20), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two separate dense heads corresponding to 'output_a' and 'output_b'
        self.dense_a = tf.keras.layers.Dense(1, name='output_a')
        self.dense_b = tf.keras.layers.Dense(1, name='output_b')

    def call(self, inputs, training=False):
        # inputs expected as list or tuple of two tensors
        input_a, input_b = inputs
        out_a = self.dense_a(input_a)
        out_b = self.dense_b(input_b)
        # Return outputs as dict to support naming losses by output names
        return {'output_a': out_a, 'output_b': out_b}

def my_model_function():
    model = MyModel()
    # Compile with loss on output_b only; output_a loss set to None to make it ignored by training
    # Using dictionary with output names as keys
    model.compile(
        optimizer='sgd',
        loss={'output_a': None, 'output_b': 'mse'}  # None loss tells TF to ignore this head's loss
    )
    return model

def GetInput():
    # Generate a tuple of two input tensors
    # Shapes: (batch_size, 10) for input_a and (batch_size, 20) for input_b
    # Use float32 dtype consistent with typical model inputs
    batch_size = 8  # a small batch for input, can be modified
    input_a = tf.random.uniform((batch_size, 10), dtype=tf.float32)
    input_b = tf.random.uniform((batch_size, 20), dtype=tf.float32)
    return (input_a, input_b)

