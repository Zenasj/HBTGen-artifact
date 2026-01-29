# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape inferred from example: [2, 256, 256, 8]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # BatchNormalization implemented manually with non-trainable moving mean and var
        shape = [8]  # feature/channel dimension
        
        # Trainable variables
        self.beta = tf.Variable(initial_value=tf.zeros(shape), trainable=True, name='beta')
        self.gamma = tf.Variable(initial_value=tf.ones(shape), trainable=True, name='gamma')
        
        # Non-trainable variables for moving statistics
        self.moving_mean = tf.Variable(initial_value=tf.zeros(shape), trainable=False, name='moving_mean')
        self.moving_var = tf.Variable(initial_value=tf.ones(shape), trainable=False, name='moving_var')

    def update_var(self, inputs):
        # Compute batch mean and variance across batch and spatial dims (assumed NHWC)
        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0,1,2], keepdims=False)
        batch_std = tf.math.sqrt(batch_var)
        
        # Update moving averages using assign (important to avoid loss of variable tracking)
        self.moving_mean.assign(self.moving_mean * 0.09 + batch_mean * 0.01)
        self.moving_var.assign(self.moving_var * 0.09 + batch_std * 0.01)
        
        return batch_mean, batch_std

    def call(self, inputs, training=False):
        if training:
            mean, var = self.update_var(inputs)
        else:
            mean, var = self.moving_mean, self.moving_var
        
        # Perform batch normalization using current mean and variance
        return tf.nn.batch_normalization(
            inputs, mean, var, offset=self.beta, scale=self.gamma, variance_epsilon=0.001
        )

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input matching [batch=2, height=256, width=256, channels=8]
    return tf.random.uniform((2, 256, 256, 8), dtype=tf.float32)

