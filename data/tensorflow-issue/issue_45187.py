# tf.random.uniform((4, 2), dtype=tf.float32) ← Input shape inferred from x_train shape: (4 samples, 2 features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original model:
        # Sequential model with Dense(20), Dense(1), Softmax
        # Softmax expects at least 2D logits (classes), but original model's Dense(1) output shape is (batch, 1),
        # Softmax on single unit is unusual; replicate original behavior exactly.
        self.dense1 = tf.keras.layers.Dense(20)
        self.dense2 = tf.keras.layers.Dense(1)
        self.softmax = tf.keras.layers.Softmax()
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Instantiate MyModel and compile it like original code
    model = MyModel()
    # Compile with SGD optimizer and mse loss as in original snippet
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mse')
    return model

def GetInput():
    # The original input was shape (4,2) with float values 0 or 1.
    # We'll generate a random tensor with shape (4,2) with float32 dtype.
    # Using uniform random over [0,1) to be consistent but flexible.
    return tf.random.uniform((4, 2), dtype=tf.float32)

