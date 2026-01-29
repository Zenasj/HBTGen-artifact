# tf.random.uniform((1, 540, 960, 16), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct PReLU layer with shared_axes=(1, 2, 3) as in example
        # This means the alpha weights are shared across height, width, channels dimensions.
        self.prelu = tf.keras.layers.PReLU(shared_axes=(1, 2, 3))
        
        # Initialize the alpha weights similar to example:
        # Example initializes weights as a scalar 0.00040957872988656163
        # The shape of alpha in PReLU with shared_axes=(1,2,3) and input (540,960,16)
        # alpha shape will be (1,) and broadcasted to all inputs in axes 1,2,3.
        # So we set weights to np.array([0.0004095787], dtype=np.float32)
        init_alpha = np.array([0.00040957872988656163], dtype=np.float32)
        self.prelu.alpha.assign(init_alpha)
        # Note: Alternatively, we can set weights via set_weights as done in example:
        # self.prelu.set_weights([init_alpha])

    def call(self, inputs):
        # Forward pass through PReLU layer
        return self.prelu(inputs)


def my_model_function():
    # Return model instance with pre-initialized PReLU alpha weight
    model = MyModel()
    return model


def GetInput():
    # Return a random input tensor matching expected shape and dtype
    # The example used an input of shape (540, 960, 16) with batch size 1
    # The values can be arbitrary but float32
    return tf.random.uniform(shape=(1, 540, 960, 16), dtype=tf.float32)

