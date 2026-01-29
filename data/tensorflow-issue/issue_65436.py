# tf.random.uniform((B, 10), dtype=tf.float32)  ← input shape inferred from issue: (None, 10)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two Dense layers as per original example
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(5, activation="softmax")
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
    def build(self, input_shape):
        # Explicitly build both Dense layers to inform Keras that model is built.
        # This is a workaround for model.summary() in TF 2.16.1+
        self.dense1.build(input_shape)
        # dense1 output shape = (batch_size, 32)
        self.dense2.build((input_shape[0], 32))
        super().build(input_shape)  # Mark the model as built

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Build the model with the expected input shape (batch unspecified)
    model.build(input_shape=(None, 10))
    return model

def GetInput():
    # Return a random float32 tensor of shape (batch_size=1, 10)
    return tf.random.uniform((1, 10), dtype=tf.float32)

