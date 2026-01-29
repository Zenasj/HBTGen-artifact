# tf.random.uniform((32, 32, 32, 3), dtype=tf.float32)  # inferred input shape and dtype

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A Keras Model demonstrating a pattern where a custom loss is added via add_loss.
    The loss is a mean absolute error between the Conv2D output and a tf.Tensor
    shifted by +1 (element-wise). The loss is integrated as part of the model via add_loss.
    
    This encapsulates both the "base model" and "custom loss" as part of one
    model object, following the workaround commented on in the issue for saving/loading.
    
    Inputs:
      - Tensor of shape (batch_size, 32, 32, 3), dtype float32
    Outputs:
      - Conv2D output tensor of shape (batch_size, 30, 30, 16)
    """
    def __init__(self):
        super().__init__()
        # Core convolution layer
        self.conv = tf.keras.layers.Conv2D(filters=16, kernel_size=3)

        # Lambda layer to compute mean absolute error loss between conv output and conv output + 1
        # This layer outputs the MAE values per example and is used via add_loss.
        self.loss_layer = tf.keras.layers.Lambda(
            lambda tensors: tf.keras.losses.mean_absolute_error(tensors[0], tensors[1])
        )

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        # Compute loss tensor: MAE between x and x+1
        loss_tensor = self.loss_layer([x, x + 1])
        # Add the computed loss to model losses
        self.add_loss(tf.reduce_mean(loss_tensor))
        return x

def my_model_function():
    """
    Returns an instance of the MyModel keras.Model
    """
    model = MyModel()
    # Compile with optimizer only, no explicit loss because loss is embedded via add_loss
    # This matches the pattern in the issue and related workaround.
    model.compile(optimizer='adam')
    return model

def GetInput():
    """
    Returns a random input tensor matching the expected model input: batch_size=32, 32x32 RGB image
    """
    return tf.random.uniform(shape=(32, 32, 32, 3), dtype=tf.float32)

