# tf.random.uniform((5, 5), minval=-128, maxval=128, dtype=tf.int32) cast to tf.int8 as input

import tensorflow as tf

class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def call(self, inp):
        # Model1 logic as per the issue:
        # expand_dims: add extra dim at axis=0
        # multiply inp with expanded version
        # take absolute value of the multiplication
        expanded = tf.expand_dims(inp, axis=0)
        multiplied = tf.multiply(inp, expanded)
        absed = tf.abs(multiplied)
        return absed,

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def call(self, inp):
        # Model2 logic as per the issue:
        # expand dims and multiply in swapped order compared to Model1
        # concat multiplied with itself on axis=0
        # return absed and concated (per original)
        expanded = tf.expand_dims(inp, axis=0)
        multiplied = tf.multiply(expanded, inp)
        concated = tf.concat([multiplied, multiplied], axis=0)
        absed = tf.abs(multiplied)
        return absed, concated

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model1 = Model1()
        self.model2 = Model2()

    @tf.function(jit_compile=True)
    def call(self, inp):
        # Run both models on the same input
        out1 = self.model1(inp)   # Tuple with one element (absed,)
        out2 = self.model2(inp)   # Tuple with two elements (absed, concated)

        # Compare first outputs (absed tensors) from both models using allclose logic
        # Since the issue deals with discrepancies especially under XLA compilation,
        # here we emit a boolean tensor indicating element-wise closeness within tolerance.
        # Using tolerances rtol=1e-3, atol=1e-3 as used in the issue's asserts.

        absed1 = out1[0]
        absed2 = out2[0]

        # Compute element-wise closeness boolean tensor
        # Use float conversion to avoid dtype mismatch issues in abs/subtract
        absed1_float = tf.cast(absed1, tf.float32)
        absed2_float = tf.cast(absed2, tf.float32)
        rtol = 1e-3
        atol = 1e-3
        diff = tf.abs(absed1_float - absed2_float)
        allowed_error = atol + rtol * tf.abs(absed2_float)
        is_close = diff <= allowed_error

        # Additionally return the concatenated tensor from model2 to reflect the full output
        concated = out2[1]

        # Return tuple: (element-wise closeness of absed outputs, concatenated tensor)
        # This fused output mirrors core parts of both models to facilitate discrepancy detection.
        return is_close, concated

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return input tensor matching original:
    # Shape: [5,5], dtype: int8, with values between -128 and 127
    # Mimic original data generation from issue
    raw_int32 = tf.random.uniform(shape=[5, 5], minval=-128, maxval=128, dtype=tf.int32)
    input_tensor = tf.cast(raw_int32, tf.int8)
    return input_tensor

