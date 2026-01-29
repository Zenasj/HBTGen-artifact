# tf.random.uniform((4, 5), dtype=tf.float32) ← Input shape inferred from example np.random.rand(4,5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original reported example model:
        # Sequential model with Dense(8, input_shape=(5,)) then Dense(1)
        self.dense1 = tf.keras.layers.Dense(8)
        self.dense2 = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Create an instance of MyModel and compile akin to example
    model = MyModel()
    # Compile with Adam optimizer and binary crossentropy loss (as in the original example)
    model.compile(optimizer="Adam", loss="binary_crossentropy")
    return model

def GetInput():
    # Return random tensor with shape (4, 5) matching the input_shape=(5,) with batch size 4 as in the example
    return tf.random.uniform((4, 5), dtype=tf.float32)

