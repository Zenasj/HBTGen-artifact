# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from Keras Input(shape=(10,))

import tensorflow as tf

class MyDense(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.num_units = num_units

    def build(self, input_shape):
        # kernel shape: [input_dim, num_units * 2, num_units]
        kernel_shape = [input_shape[-1], self.num_units * 2, self.num_units]
        bias_shape = [self.num_units]

        self.kernel = self.add_weight(
            "kernel", shape=kernel_shape, trainable=True,
            initializer="glorot_uniform")
        self.bias = self.add_weight(
            "bias", shape=bias_shape, trainable=True,
            initializer="zeros")
        super(MyDense, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, input_dim)
        # kernel: (input_dim, num_units*2, num_units)
        # einsum performs: batch x input_dim, multiplied with input_dim x (2*num_units) x num_units 
        # result shape: (batch_size, 2*num_units, num_units)
        output = tf.einsum("ac,cde->ade", inputs, self.kernel) + self.bias
        return output

class MyModel(tf.keras.Model):
    def __init__(self, num_units=15):
        super(MyModel, self).__init__()
        self.my_dense = MyDense(num_units)

    def call(self, inputs):
        # Inputs shape (batch_size, 10)
        # Output shape (batch_size, 30, 15) since 2*num_units=30
        return self.my_dense(inputs)

def my_model_function():
    # Return an instance of MyModel with num_units=15 as in original code
    return MyModel(num_units=15)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel:
    # shape: (batch_size, 10), use batch size 1 for simplicity
    return tf.random.uniform(shape=(1, 10), dtype=tf.float32)

