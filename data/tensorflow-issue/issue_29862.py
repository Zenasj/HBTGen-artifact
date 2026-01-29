# tf.random.poisson(lam=10, shape=(batch_size, 3), dtype=tf.int64) is used inside the model on input shape (batch_size, 3)
import tensorflow as tf

class StupidLayer(tf.keras.layers.Layer):
    def __init__(self, batch_size, x_shape, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.x_shape = x_shape

    def build(self, input_shape):
        # Create a poisson random tensor of shape (batch_size, 3) with int64 dtype
        self.y = tf.random.poisson(
            lam=10, shape=(self.batch_size,) + self.x_shape, dtype=tf.int64
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs is int64, self.y is int64, multiply will be int64
        # Cast to float32 to match model output dtype
        return tf.cast(inputs * self.y, tf.float32)

class MyModel(tf.keras.Model):
    def __init__(self, batch_size=10, x_shape=(3,)):
        super().__init__()
        self.batch_size = batch_size
        self.x_shape = x_shape
        self.stupid_layer = StupidLayer(batch_size=batch_size, x_shape=x_shape)

    def call(self, inputs, training=False):
        # Expect input tensor of shape (batch_size, 3) and dtype int64.
        return self.stupid_layer(inputs)

def my_model_function():
    # Instantiate the model with default batch_size=10 and input shape (3,)
    model = MyModel(batch_size=10, x_shape=(3,))
    # Typical keras model compile call to match original, but user may compile after creation
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    return model

def GetInput():
    # Return random int64 tensor input of shape (batch_size, 3),
    # matching expected dtype and shape as used in example
    batch_size = 10
    x_shape = (3,)
    return tf.random.uniform(shape=(batch_size,) + x_shape, minval=0, maxval=50, dtype=tf.int64)

