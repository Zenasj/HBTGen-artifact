# tf.random.uniform((), dtype=tf.float32) ← input is a scalar float or tensor convertible scalar

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A minimal model demonstrating the behavior related to Tensor conversion
    that is discussed in the issue.

    This model applies tf.math.real on the input tensor. The input can be
    a raw Python float (or other Tensor-like object registered with 
    tf.convert_to_tensor) because tf.math.real expects a Tensor but ideally
    should accept Tensor-like objects that can be converted to Tensor.

    The original issue highlights that calling tf.math.real(1.) fails in TF 1.14
    because non-Tensors are not auto-converted inside tf.math.real function.

    This model ensures that the input is converted explicitly before applying 
    tf.math.real to avoid failures.
    """

    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Explicitly convert input to Tensor before calling tf.math.real to avoid errors
        # observed in TF 1.14, where tf.math.real did not autoconvert non-Tensors.
        tensor_input = tf.convert_to_tensor(inputs)
        real_part = tf.math.real(tensor_input)
        return real_part

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random scalar float compatible with the model input expectations.
    # According to the issue context, input can be any scalar float or
    # tensor-like object convertible to tensor.
    # We provide a scalar float which will be converted inside the model.
    #
    # Note: input shape is () scalar float
    return tf.random.uniform(shape=(), dtype=tf.float32)

