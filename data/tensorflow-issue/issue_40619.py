# tf.random.normal((B, input_dim)) ← Input shape is (batch_size, input_dim), input_dim=100 inferred from example

import tensorflow as tf

class Dense(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_size, **kwargs):
        super(Dense, self).__init__(**kwargs)
        # Initialize weights and biases as Variables
        self.w = self.add_weight(
            shape=(input_dim, output_size),
            initializer=tf.random_normal_initializer(),
            trainable=True,
            name="w"
        )
        self.b = self.add_weight(
            shape=(output_size,),
            initializer=tf.zeros_initializer(),
            trainable=True,
            name="b"
        )

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Following the example: input_dim=100, sizes=[30, 30]
        self.input_dim = 100
        self.sizes = [30, 30]

        # Build layers similar to MLP in example
        self.layers_list = []
        input_size = self.input_dim
        for size in self.sizes:
            self.layers_list.append(Dense(input_dim=input_size, output_size=size))
            input_size = size

    def call(self, x):
        # Forward pass through all Dense layers with ReLU activations
        for layer in self.layers_list:
            x = layer(x)
        return x


def my_model_function():
    # Instantiate and return MyModel
    return MyModel()


def GetInput():
    # Return a random normal input tensor of shape (batch_size, input_dim)
    # Using batch_size=8 as a reasonable default for testing
    batch_size = 8
    input_dim = 100  # inferred from the original example where input_size=100
    # Use tf.random.normal to match the original example style
    return tf.random.normal((batch_size, input_dim))

