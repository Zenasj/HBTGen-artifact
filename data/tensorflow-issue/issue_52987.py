# tf.random.uniform((), dtype=tf.float32) ← The inputs are scalar float tensors as per the provided tf.function signatures

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We are replicating the scenario where a tf.function is defined with 
        # experimental_implements metadata, and attached to the model.
        # This function adds two scalar floats.
        self.experimental_implements = "name: \"addons:MaxUnpooling2D\""

        # Define a tf.function with input_signature matching the test function from the issue
        @tf.function(
            input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)],
            experimental_implements=self.experimental_implements
        )
        def test_fn(a, b):
            return a + b

        self._test_fn = test_fn

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)],
        experimental_implements="name: \"addons:MaxUnpooling2D\""
    )
    def call(self, a, b):
        # This allows the model() call to be passed scalar float inputs and return their sum,
        # consistent with the behavior described.
        return a + b

    # Provide access to the test function for checking _implements attribute or equivalence.
    @property
    def test(self):
        return self._test_fn

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of two scalar float tensors matching the input_signature of MyModel.call / test function
    # Scalars of dtype float32 as per the issue.
    a = tf.random.uniform((), dtype=tf.float32)
    b = tf.random.uniform((), dtype=tf.float32)
    return (a, b)

