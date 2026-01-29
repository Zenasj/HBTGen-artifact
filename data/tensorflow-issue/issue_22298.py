# tf.random.uniform((10, 3), dtype=tf.float32) ← Input shape inferred from example x = tf.random_uniform([10, 3])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple architecture based on the example: one Dense layer with 10 units
        # This aligns with the example in the issue where tf.keras.Sequential with a Dense(10) was used.
        self.fc1 = tf.keras.layers.Dense(10, name='fc1')
    
    def call(self, inputs):
        # Forward pass through the single Dense layer
        return self.fc1(inputs)

def my_model_function():
    # Return an instance of MyModel
    # In the issue, the model was a keras.Model subclass or Sequential with Dense(10).
    # No weights are loaded here because the issue relates to variable scoping and init_from_checkpoint incompatibility.
    # User is recommended to use load_weights or set_weights as alternative.
    return MyModel()

def GetInput():
    # Return a random tensor input consistent with the example x = tf.random_uniform([10,3])
    # Use tf.random.uniform to match TensorFlow 2.x style.
    # Assume dtype float32 (default)
    return tf.random.uniform(shape=(10, 3))

