# tf.random.uniform((8, 32, 32, 3), dtype=tf.float32) ← inferred from the input batch size and shape in the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example model: single Conv2D layer with 8 filters and (3,3) kernel
        self.conv = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3))

    def call(self, inputs, training=False):
        return self.conv(inputs)

def null_fn(y_true, y_pred):
    # Custom loss function that simply returns 0, as per example
    return tf.constant(0.)

def my_model_function():
    # Instantiate the model, compile with custom loss as per example
    model = MyModel()
    # Compile with Adam optimizer and custom loss null_fn
    model.compile(optimizer='adam', loss=null_fn)
    return model

def GetInput():
    # Return random input tensor of shape (8, 32, 32, 3) matching the example batch and shape
    return tf.random.uniform((8, 32, 32, 3), dtype=tf.float32)

