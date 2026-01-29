# tf.random.uniform((B,)) ← Input shape: single-dimensional scalar tensor (shape=()) as no specific shape was given in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A variable initialized to 3.0 as per the original example
        self.var = tf.Variable(3.0, trainable=True)

    def call(self, inputs):
        # We cannot output tf.Variable directly as a model output in Keras,
        # but we can output a tensor derived from it using tf.identity
        scaled_input = inputs * self.var
        var_tensor = tf.identity(self.var)  # convert variable to tensor for output
        
        # The model returns both scaled input tensor and the variable tensor
        return scaled_input, var_tensor

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Inputs must be a tensor compatible with the model's forward pass expecting inputs that multiply 
    # with a scalar variable; from the demonstration inputs were scalar tensors
    # Since no input shape was explicitly provided, assuming scalar inputs (shape=()) with float32 dtype
    return tf.random.uniform((), dtype=tf.float32)

