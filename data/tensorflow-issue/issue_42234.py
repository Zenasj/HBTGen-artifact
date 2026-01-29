# tf.random.uniform((B,)) ← The model here is a simple Sequential model with one Dense layer taking an input tensor of shape (batch_size, features).
# Since the original example was minimal and did not specify input shape, assume input shape = (5,), i.e. feature vector length 5 for demonstration.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Dense layer with 3 outputs and sigmoid activation as per original example
        self.dense = tf.keras.layers.Dense(3, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Compile the model with Nadam optimizer configured with a learning rate as in the issue example
    # Use tf.keras.optimizers.Nadam directly instead of dictionary to avoid the issue described
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001)
    # Compile with a dummy loss to complete the example (e.g., mean squared error)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    return model

def GetInput():
    # Generate a random input tensor matching the assumed input shape
    # batch size = 8, feature vector length = 5 (arbitrary reasonable choice)
    return tf.random.uniform((8, 5), dtype=tf.float32)

