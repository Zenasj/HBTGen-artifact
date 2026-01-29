# tf.random.uniform((B, 10), dtype=tf.float32) ← Inferred input shape based on reproduction code (batch size B, 10 features)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers similar to the example reproductions:
        # Dense(10) -> BatchNormalization -> Dense(1)
        self.dense1 = tf.keras.layers.Dense(10)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Forward pass with explicit "training" passed on for BatchNormalization
        x = self.dense1(inputs)
        x = self.batch_norm(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    # No pretrained weights or special params mentioned, so default init
    return MyModel()

def GetInput():
    # Generate a random input tensor with shape (1, 10) as used in repro examples
    # dtype float32 matches Tensorflow default for layers.Dense
    return tf.random.uniform((1, 10), dtype=tf.float32)

