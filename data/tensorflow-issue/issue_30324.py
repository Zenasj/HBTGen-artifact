# tf.random.uniform((B=1, H=1, W=1, C=3000), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model replicates the common use case discussed in the issue:
    a simple single Dense layer model with input dimensionality 3000.
    The output is a dense transformed tensor.
    
    Since the issue describes multiple models (model_a and model_b) created for comparison
    and discusses memory leak and clearing session issues,
    we encapsulate two submodels and add a comparison between their outputs.
    
    This class holds two identical submodels and outputs whether their predictions are equal 
    within a numerical tolerance, per element.
    """

    def __init__(self):
        super().__init__()
        # Input shape 3000 per example
        # Two simple submodels with one Dense layer each
        self.model_a = tf.keras.Sequential([
            tf.keras.layers.Dense(3000, input_shape=(3000,))
        ])
        self.model_b = tf.keras.Sequential([
            tf.keras.layers.Dense(3000, input_shape=(3000,))
        ])

    def call(self, inputs, training=False):
        """
        Forward pass:
        - run inputs through both submodels
        - compare outputs elementwise within a tolerance
        - return a boolean tensor showing where outputs are close.
        
        This embeds the "compare and check" logic hinted in the issue's discussion,
        simulating the two models "being compared or discussed together".
        """
        out_a = self.model_a(inputs, training=training)
        out_b = self.model_b(inputs, training=training)
        
        # Use tf.math.abs difference to check closeness within tolerance
        tolerance = 1e-5
        close = tf.math.less_equal(tf.math.abs(out_a - out_b), tolerance)
        
        # Return boolean tensor indicating matching elements
        return close


def my_model_function():
    """
    Construct and return an instance of MyModel.
    This matches the typical keras model creation pattern discussed in the issue.
    """
    return MyModel()


def GetInput():
    """
    Generate a random tensor input matching MyModel expected input shape and type.
    Shape: (1, 3000) - batch size 1, feature dimension 3000
    
    The input shape and data type is aligned with the example inputs in the issue text,
    i.e. numpy normal distribution with size (1, 3000) converted to tf.float32 tensors.
    """
    # Using tf.random.uniform to generate float32 tensor with the correct shape
    # The problem states using tf.random.uniform((B,H,W,C), dtype=...)
    # Here, input is 2D: batch x features = (1, 3000)
    return tf.random.uniform((1, 3000), dtype=tf.float32)

