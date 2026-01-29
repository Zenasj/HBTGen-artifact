# tf.random.uniform((B,), dtype=tf.float32) ← Based on the example, input tensors are scalar float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Compose an Adder-like module as a tf.Module, exposing two methods:
        #  - add(x,y) = x + y^2 + 1
        #  - square(x) = x^2
        # Both methods decorated as tf.functions with input_signature allowing SavedModel multiple signatures export simulation.
        # Here we encapsulate these as tf.functions inside the model for demonstration.
        
        self.add_fn = tf.function(self._add, input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32)
        ])
        self.square_fn = tf.function(self._square, input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32)
        ])

    def _add(self, x, y):
        return x + y ** 2 + 1

    def _square(self, x):
        return x ** 2

    def call(self, inputs):
        # For demonstration, inputs is a dict with keys "x" and "y" where "y" may be optional.
        # If "y" is provided, compute add(x,y)
        # Otherwise, compute square(x).
        # This unify the two functions into a single forward call.

        x = inputs.get("x")
        y = inputs.get("y", None)

        if y is not None:
            # Use the add function logic (x + y^2 + 1)
            return self.add_fn(x, y)
        else:
            # Use the square function logic (x^2)
            return self.square_fn(x)

def my_model_function():
    # Return an instance of the MyModel class
    return MyModel()

def GetInput():
    # As the model expects a dict with keys "x" and optional "y", create inputs accordingly.
    # We create both scenarios: with and without 'y' to cover both signatures.
    # For simplicity, output with 'y' included as example input.
    x = tf.random.uniform((), dtype=tf.float32)  # scalar float32 tensor
    y = tf.random.uniform((), dtype=tf.float32)  # scalar float32 tensor
    return {"x": x, "y": y}

