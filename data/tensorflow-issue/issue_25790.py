# tf.random.uniform((B, 20, 20, C), dtype=tf.float32) ← input shape inferred from Dense layer input_shape=(20, 20)

import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        # input_shape is tf.TensorShape and its dims are accessible as ints or tf.Dimension
        # Use integer floor division to avoid TypeError
        input_dim = input_shape[-1]
        output_dim = input_shape[-1] // 2  # floor division to avoid incompatibility
        
        self.kernel = self.add_weight(
            shape=(input_dim, output_dim),
            name='kernel',
            initializer='ones',
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.linalg.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = shape[-1] // 2
        return tuple(shape)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Sequential model equivalent:
        # Dense(8, input_shape=(20, 20)) + MyLayer
        # Note: input_shape excludes batch dimension
        self.dense = tf.keras.layers.Dense(8)
        self.my_layer = MyLayer()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.my_layer(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random float32 tensor input matching the model input (batch_size, 20, 20)
    # Use a batch size of 1 for simplicity
    return tf.random.uniform((1, 20, 20), dtype=tf.float32)

