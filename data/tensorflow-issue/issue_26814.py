# tf.random.uniform((B, 1), dtype=tf.float32)  # Inferred input shape from ToyModel: (batch_size, 1)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Combines the key aspects of the ToyModel example from the issue discussion,
    adjusted for compatibility with saving/loading and inference in TF 2.x.

    This model is a simple linear model with one Dense layer of 5 units,
    reflecting the original ToyModel design.

    The call method is decorated with tf.function and an input_signature to
    facilitate saved model export and restore.

    This example includes the corrections implied by the issue discussion:
    - Proper decorating of call
    - Using a consistent signature for input
    """

    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(5)

    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32, name="input_func")])
    def call(self, x):
        return self.l1(x)

def my_model_function():
    """
    Instantiate and return an instance of MyModel.
    No pretrained weights are loaded here; weights are random by default.
    """
    return MyModel()

def GetInput():
    """
    Return a random input tensor compatible with MyModel's expected input shape.
    The model expects: (batch_size, 1), dtype float32.
    We'll generate a random uniform tensor with batch size 2 for demonstration.
    """
    batch_size = 2
    input_shape = (batch_size, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

