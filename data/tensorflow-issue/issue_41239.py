# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from the example: (batch_size, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two submodels similarly structured with a Dense layer outputting 10 units
        # This is to simulate the "mixing eager and non-eager models" scenario by defining two sub-models.
        self.model0 = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1,)),
            tf.keras.layers.Dense(10)
        ])
        self.model1 = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1,)),
            tf.keras.layers.Dense(10)
        ])
    
    def call(self, inputs):
        # Forward pass through both models
        out0 = self.model0(inputs)
        
        # The second submodel is instantiated but not assigned to any graph (mimicking legacy graph mode),
        # here, logically it is used for comparison but not executed directly on inputs.
        # Since TF 2.x does not support mixing eager and graph this way in a single call,
        # we simulate by computing model1 output independently (we do it here for demonstration).
        out1 = self.model1(inputs)
        
        # Compare outputs numerically with a tolerance to simulate the problem context
        # Note: In the original report, the error happened when mixing eager/non-eager,
        # here we just produce a difference tensor:
        diff = tf.abs(out0 - out1)
        
        # Return a boolean tensor indicating element-wise if the outputs are close within a tolerance
        is_close = tf.less_equal(diff, 1e-5)
        
        # Reduce across axis=-1 to check if all 10 output units are close per batch element
        all_close_per_sample = tf.reduce_all(is_close, axis=-1)
        
        return all_close_per_sample

def my_model_function():
    # Return an instance of MyModel (initialized with random weights)
    return MyModel()

def GetInput():
    # Create a random input tensor of shape (batch_size=4, 1) matching the example usage
    # Using tf.float32 dtype as typical for TF Keras models
    return tf.random.uniform((4, 1), dtype=tf.float32)

