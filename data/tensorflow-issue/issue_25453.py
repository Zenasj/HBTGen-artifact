# tf.random.uniform((B, input_dim)) ← assuming input is a 2D tensor (batch_size, input_dim)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_outputs=10):
        super(MyModel, self).__init__()
        self.num_outputs = num_outputs
        # Initialize kernel variable here similarly to custom layer
        # Using add_weight as recommended in TF 2.x
        self.kernel = self.add_weight(
            name='kernel',
            shape=(None, self.num_outputs),  # will be set dynamically in build
            initializer='random_normal',
            trainable=True,
            dtype=tf.float32
        )
        # We need to track input_dim to set kernel shape dynamically
        self.built_flag = False

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # Reinitialize kernel properly now that input_dim is known
        self.kernel.assign(tf.random.normal([input_dim, self.num_outputs]))
        self.built_flag = True
        super(MyModel, self).build(input_shape)

    def call(self, inputs):
        # If kernel shape is not set correctly (e.g., first call), recreate kernel properly
        if not self.built_flag:
            input_dim = tf.shape(inputs)[-1]
            self.kernel = tf.Variable(
                tf.random.normal([input_dim, self.num_outputs]),
                trainable=True,
                dtype=tf.float32,
                name='kernel'
            )
            self.built_flag = True
        return tf.linalg.matmul(inputs, self.kernel)

    # Custom method to compute gradients manually, mimicking the _backprop example
    # y_delta: gradient flowing back from next layer (same shape as output)
    # x: original input to the forward pass (same shape as inputs)
    def _backprop(self, y_delta, x):
        # Gradient wrt weights is matmul of x^T and y_delta
        w_delta = tf.matmul(tf.transpose(x), y_delta)
        # Gradient wrt input is matmul of y_delta and kernel^T
        x_delta = tf.matmul(y_delta, tf.transpose(self.kernel))
        return w_delta, x_delta


def my_model_function():
    # Return an instance of MyModel with default num_outputs=10
    return MyModel(num_outputs=10)


def GetInput():
    # For demonstration, we assume input batch size 4 and input_dim 8
    batch_size = 4
    input_dim = 8
    # Return a random tensor input of shape (batch_size, input_dim)
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

