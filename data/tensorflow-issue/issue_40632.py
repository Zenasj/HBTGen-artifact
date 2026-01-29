# tf.random.uniform((N, N), dtype=tf.float32) for each input, with 2 inputs concatenated along last axis

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, output_dim=32, n_inputs=2):
        super(MyModel, self).__init__()
        self.output_dim = output_dim
        self.n_inputs = n_inputs
        self.W_list = []

    def build(self, input_shapes):
        # input_shapes is list of shapes for each input tensor, each shape (batch_size, input_dim)
        self.input_dim = input_shapes[0][1]

        # Create separate weights for each input; do NOT concatenate here to preserve gradient tracking
        self.W_list = [
            self.add_weight(
                name=f'W_{i}',
                shape=(self.input_dim, self.output_dim),
                initializer='random_normal',
                trainable=True
            ) for i in range(self.n_inputs)
        ]

    def call(self, inputs):
        # inputs: list of tensors (batch_size, input_dim)
        # Concatenate inputs along last axis to get combined (batch_size, input_dim * n_inputs)
        supports = tf.concat(inputs, axis=-1)

        # Concatenate weights to (input_dim * n_inputs, output_dim)
        W = tf.concat(self.W_list, axis=0)

        # Multiply supports with weights
        return tf.matmul(supports, W)


def my_model_function():
    # Instantiate MyModel with 2 inputs and output dim 32 as in the example
    return MyModel(output_dim=32, n_inputs=2)


def GetInput():
    # According to example: batch_size N=100, each input tensor shape (N, N)
    N = 100
    # Generate 2 random input tensors shaped (N, N)
    # Use tf.random.uniform to produce float32 tensors compatible with model
    return [tf.random.uniform((N, N), dtype=tf.float32) for _ in range(2)]

