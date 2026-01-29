# tf.zeros((1,1), dtype=tf.float32) ← Input shape inferred from example usage in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize weights with a default value
        # Using tf.Variable to illustrate the safer pattern mentioned in the discussion
        self.weights = tf.Variable(tf.zeros((4, 5), dtype=tf.float64), trainable=False)

    @tf.function
    def encoder_graph(self, x):
        # Return current weights tensor
        return self.weights
        
    def encoder_eager(self, x):
        # Return current weights tensor (eager execution)
        return self.weights

    def call(self, x):
        # The core issue raised is that changing self properties (self.weights)
        # does not trigger retracing and causes graph and eager modes to differ.
        # This example sets self.weights twice with different shapes and compares shapes.

        # Assign a new tensor to weights (simulate dynamic property change)
        self.weights.assign(tf.zeros((4,5), dtype=tf.float64))
        result1 = self.encoder_graph(x)
        # Change weights tensor shape and assign again
        self.weights.assign(tf.zeros((2,3), dtype=tf.float64))
        result2 = self.encoder_graph(x)

        # Compare shapes of encoder_graph output and encoder_eager output
        # This replicates the original example: returns boolean tensor
        are_shapes_equal = tf.reduce_all(tf.equal(tf.shape(result2), tf.shape(self.encoder_eager(x))))

        return are_shapes_equal

def my_model_function():
    return MyModel()

def GetInput():
    # According to the usage in the original reproduction:
    # input is tf.zeros((1,1)) passed as x to call and encoder methods
    return tf.zeros((1, 1), dtype=tf.float32)

