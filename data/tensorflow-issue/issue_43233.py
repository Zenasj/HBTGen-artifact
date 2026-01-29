# tf.random.uniform((3, 3), dtype=tf.float32) ← Input shape inferred from example: batch=3, feature=3
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, input_shape=(3,))
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Batch size 3, feature size 3, matching the example input for the model.fit call
    return tf.random.uniform((3, 3), dtype=tf.float32)

