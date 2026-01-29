# tf.random.uniform((B, 1), dtype=tf.float32) ← Assumed input shape is (batch_size, 1) based on Input(shape=(1,)) in the original issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two Dense layers with 1 unit each, mimicking the example from the issue
        self.layer1 = tf.keras.layers.Dense(1)
        self.layer2 = tf.keras.layers.Dense(1)
        
        # Add losses using the fixed approach to capture layer kernel correctly inside the lambda
        # Use closure helper to avoid late binding issue in lambdas inside loops as per the issue discussion
        def make_loss(layer):
            return lambda: tf.reduce_sum(layer.kernel)

        # Add losses properly to each layer
        self.layer1.add_loss(make_loss(self.layer1))
        self.layer2.add_loss(make_loss(self.layer2))
    
    def call(self, inputs):
        x1 = self.layer1(inputs)
        x2 = self.layer2(inputs)
        # Just output both layer outputs as a tuple, for demonstration
        # You could implement any logic here based on the issue context
        return x1, x2

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape (batch_size, 1) expected by MyModel
    # batch_size is inferred as 4 here arbitrarily for test, since issue example does not specify batch_size
    return tf.random.uniform((4, 1), dtype=tf.float32)

