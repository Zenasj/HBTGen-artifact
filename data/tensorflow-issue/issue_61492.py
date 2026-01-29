# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build functional model architecture from the issue
        self.inputs_layer = tf.keras.layers.InputLayer(input_shape=(28, 28, 1))
        self.conv2d = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
        
        # The submodel that replaces Conv2D, as per the clone_function logic:
        # Takes input tensor and applies Conv2D layer followed by Lambda (t * 2) and Add.
        # We'll instantiate this here for clarity and reuse.
        self.submodel_conv = None # Will be lazily created during call on input shape
        
    def build_submodel_conv(self, input_shape):
        # Input layer for the submodel
        inputs = tf.keras.Input(shape=input_shape[1:])
        x = self.conv2d(inputs)
        y = tf.keras.layers.Lambda(lambda t: t * 2)(x)
        outputs = tf.keras.layers.Add()([x, y])
        submodel = tf.keras.Model(inputs=inputs, outputs=outputs, name='conv2d_submodel')
        return submodel

    def call(self, inputs, training=None):
        # inputs: (B, 28, 28, 1)
        x = inputs
        # Because the issue involves infinite loop cloning when Conv2D is replaced 
        # by a functional submodel, we simulate this replacement here by
        # using the submodel instead of plain Conv2D.

        # Lazily create submodel matching input shape
        if self.submodel_conv is None:
            # input_shape includes batch dim, e.g. (None, 28, 28, 1)
            self.submodel_conv = self.build_submodel_conv(tf.shape(x))
        
        x = self.submodel_conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Input shape from the original issue: (batch_size, 28, 28, 1)
    batch_size = 1
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

