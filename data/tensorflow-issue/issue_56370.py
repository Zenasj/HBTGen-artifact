# tf.random.normal((2, 3, 4)) + x where x has shape [None, None] (batched, variable-length sequences)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # No additional layers; model adds random noise to input tensor
        # The input shape is expected to be [batch_size, seq_len] with float32 dtype

    def call(self, x):
        # x assumed to be a float32 tensor with shape [batch_size, sequence_length]
        # Add random noise of shape (2,3,4) + x (broadcasting will occur if shapes differ)
        # Since input could be variable shape, and random noise is fixed shape,
        # this is mostly a demonstration of tracing behavior, matching original example.
        noise = tf.random.normal((2,3,4))
        return noise + x

class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    # Define input signatures matching original issue's example
    F1_SIG = [tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='inputs_1')]
    F2_SIG = [tf.TensorSpec(shape=[None], dtype=tf.float32, name='inputs_2')]

    @tf.function(input_signature=F1_SIG)
    def f1(self, x):
        # Print statement to show tracing (would appear once in TF2.3, multiple times in TF2.4+)
        print("tracing f1")
        # Pass first element of x (tuple) to model, per original example
        return self.model(x[0])

    @tf.function(input_signature=F2_SIG)
    def f2(self, x):
        print("tracing f2")
        return self.model(x)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We combine both behaviors from the issue: two functions f1, f2 on different inputs
        self.internal_model = MyModelInner()

    def call(self, inputs):
        # inputs is expected to be a tuple or list with two elements used for f1 and f2,
        # but to unify into one model, we'll just call internal_model on inputs[0]
        # for demonstration as in original example.
        x1 = inputs[0]
        return self.internal_model(x1)

# Because the original issue's main focus is on retracing behavior of tf.function wrapping calls
# to an internal model, the actual model class is a simple custom keras.Model that adds random noise.

# For clarity and requirements of this task, keep only MyModel class that mirrors original CustomModel logic:

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, x):
        # x: float32 Tensor with shape [batch_size, variable_length]
        noise = tf.random.normal((2, 3, 4))
        return noise + x

def my_model_function():
    # Return an instance of the simple MyModel class
    return MyModel()

def GetInput():
    # Return a random input tensor matching expected input signature:
    # Based on F1_SIG: shape [None, None], dtype float32, so batch size and sequence length variable.
    # We pick something concrete for testing, e.g. batch=2, seq_len=4
    return tf.random.uniform((2, 4), dtype=tf.float32)

