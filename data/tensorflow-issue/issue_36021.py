# tf.random.uniform(()) ← This example uses a scalar input to match the saved Module's single variable use-case

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Mimicking the original Module holding a variable initialized to a fixed scalar (e.g., 9000)
        # This model has one trainable variable as in the original issue example
        self.variable = tf.Variable(9000.0, trainable=True, name='variable')

    @property
    def trainable_variables(self):
        # Explicitly expose trainable_variables as a property as in tf.Module to ensure compatibility
        return (self.variable, )

    def call(self, x):
        # Forward pass simply adds the variable to the input scalar (arbitrary operation)
        # This allows the variable to be used meaningfully in a TF graph / SavedModel context
        return x + self.variable

def my_model_function():
    # Return an instance of MyModel with the variable initialized
    return MyModel()

def GetInput():
    # Provide a single scalar float tensor (batch dimension omitted as original example is scalar)
    # This matches what the model expects for the call method
    return tf.constant(1.0, dtype=tf.float32)

