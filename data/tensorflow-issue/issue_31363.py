# tf.constant shape=(), dtype=int32 (scalar tensor)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, initial=100):
        super().__init__()
        # We use a tf.Variable to hold state inside the model.
        # This aligns with the explanation in the issue:
        # "If you need to change the value of something inside a tf.function 
        # and you want that to persist either make it a return value or make it a tf.Variable"
        self.value = tf.Variable(initial, dtype=tf.int32)

    def call(self, inputs=None):
        # Increment the variable by 1 and return the updated value.
        self.value.assign_add(1)
        return self.value

def my_model_function():
    # Return an instance of MyModel with initial value 100
    return MyModel(initial=100)

def GetInput():
    # This model does not require any input tensor for its operation,
    # but to keep consistent with the signature, we return None.
    # The forward pass increments an internal variable without input.
    return None

