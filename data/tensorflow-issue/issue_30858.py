# tf.random.uniform((B, 32), dtype=tf.float32)
import tensorflow as tf

class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer='random_normal',
            trainable=True,
            name="W")
        self.b = self.add_weight(
            shape=(units,),
            initializer='zeros',
            trainable=True,
            name='b')

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class NestedLinear(tf.keras.layers.Layer):
    def __init__(self, input_dim=32):
        super(NestedLinear, self).__init__()
        # Initialize child Linear layer with input_dim
        self.linear_1 = Linear(units=32, input_dim=input_dim)

    def call(self, inputs):
        return self.linear_1(inputs)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Create base Linear layer and nested Linear layer
        # Note: input_dim=32 assumed from the original code
        self.linear = Linear(units=32, input_dim=32)
        self.nested_linear = NestedLinear(input_dim=32)

    def call(self, inputs):
        # Compute outputs of both linear layers
        out_linear = self.linear(inputs)
        out_nested = self.nested_linear(inputs)

        # Compare outputs element-wise within a tolerance
        # This reflects a potential comparison of both models' forward outputs
        tol = 1e-5
        diff = tf.abs(out_linear - out_nested)
        compare = diff < tol

        # Return boolean tensor indicating closeness, shape (batch_size, 32)
        return compare


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random float32 input of shape (batch_size, 32)
    # Batch size chosen as 4 for example, consistent with input shape (None, 32)
    batch_size = 4
    return tf.random.uniform(shape=(batch_size, 32), dtype=tf.float32)

