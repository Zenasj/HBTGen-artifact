# tf.random.uniform((1, 1, 1), dtype=tf.float32) ← inferred input shape from x=tf.constant(3.0, shape=(1,1,1))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # From original MetaModel: two Dense layers, hidden1 with 5 units, out with 1 unit
        # The original code used input_shape=(1,) for hidden1 - input after squeeze likely shape (1,1)
        self.hidden1 = tf.keras.layers.Dense(5)
        self.out = tf.keras.layers.Dense(1)

    def call(self, x):
        # Use relu activation after hidden layer like in original code
        x = tf.nn.relu(self.hidden1(x))
        x = self.out(x)
        return x

def my_model_function():
    # Return an instance of MyModel with freshly initialized weights
    return MyModel()

def GetInput():
    # Provide an input tensor matching the expected input shape (1,1,1),
    # squeeze to (1,1) before feeding to model (which expects last dim = 1)
    # The model's layers expect input shape (?,1)
    # We'll generate a random float tensor with shape (1, 1, 1) 
    # to match original example and slice appropriately in usage
    return tf.random.uniform((1, 1, 1), dtype=tf.float32)

