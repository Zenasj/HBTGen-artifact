# tf.random.uniform((B, 5), dtype=tf.float32) ← Assumed input shape (batch_size, 5) as per the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A Dense layer with output units=10, input dim=5 inferred from example
        self.linear = tf.keras.layers.Dense(10)
        
    def call(self, x, cache=None, training=None):
        """
        Implements conditional logic similar to the issue description:
        - If training mode is active or cache is None: run linear layer on x
        - Otherwise, use tf.cond to check if cache is empty:
            - If cache shape[0]==0, run linear layer on x
            - Else, return the cache tensor
        
        This preserves the conditional logic layer, which causes different variable naming
        due to tf.cond name scopes in TF 2.0, reproducing the issue.
        """
        if training or cache is None:
            return self.linear(x)
        else:
            return tf.cond(
                tf.equal(tf.shape(cache)[0], 0),
                true_fn=lambda: self.linear(x),
                false_fn=lambda: cache
            )

def my_model_function():
    # Return an instance of MyModel without any special initialization
    return MyModel()

def GetInput():
    # Generate a random input tensor of shape (batch_size=3, 5 features)
    # dtype matches model expected input which is float32
    return tf.random.uniform((3, 5), dtype=tf.float32)

