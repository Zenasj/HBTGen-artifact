# tf.random.uniform((B=1, 200), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the architecture as per original example: Input(200) -> Dense(64, relu) -> Dense(1, sigmoid)
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Return a compiled instance of MyModel
    model = MyModel()
    # Compile is optional if only inference used, included here for completeness
    model.compile()
    return model

def GetInput():
    # Produce a single input sample with shape (1, 200), float32 dtype to match model
    return tf.random.uniform((1, 200), dtype=tf.float32)

