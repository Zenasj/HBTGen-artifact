# tf.random.uniform((1, 5), dtype=tf.float32) ← input shape inferred from original model using (1,5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture inferred from given Sequential model:
        # Input shape (1, 5), Dense 10 units with sigmoid, Dense 2 units linear output
        self.dense1 = tf.keras.layers.Dense(10, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(2, activation='linear')

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Forward pass same as original Sequential model
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate MyModel
    model = MyModel()
    # Since original code compiled with optimizer and loss for training,
    # here we just return the instantiated model without compilation,
    # full user training loop should handle compile if needed.
    return model

def GetInput():
    # Return a random input tensor matching the shape (batch_size=1, input_dim=5)
    # The original model input shape is (1,5), batch size 1 and 5 features.
    # Use dtype float32 to match typical model expectations.
    return tf.random.uniform((1, 5), dtype=tf.float32)

