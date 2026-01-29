# tf.random.normal((B, input_dim), dtype=tf.float32) ← Input shape inferred as batch size B and feature size input_dim=100 as in example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Build MLP composed of Dense layers per the fixed example
        # Here 'input_dim' is chained through layers
        sizes = [30, 30]
        input_dim = 100
        
        self.layers_list = []
        for size in sizes:
            self.layers_list.append(Dense(input_dim=input_dim, output_size=size))
            input_dim = size

    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x


class Dense(tf.Module):
    def __init__(self, input_dim, output_size, name=None):
        super(Dense, self).__init__(name=name)
        # Initialize weights and bias as trainable variables
        self.w = tf.Variable(tf.random.normal([input_dim, output_size]), name='w')
        self.b = tf.Variable(tf.zeros([output_size]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input with shape (batch_size, input_dim)
    # batch_size and input_dim are set to 4 and 100 (default input size)
    batch_size = 4
    input_dim = 100
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

