# tf.random.uniform((B, D)) where D is input feature dimension (inferred as flexible for example simplicity)
import tensorflow as tf

class FlexibleDense(tf.keras.layers.Layer):
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        # Using a Dense layer internally for illustration
        self.dense = tf.keras.layers.Dense(out_features)

    def call(self, inputs):
        return self.dense(inputs)

class MyModel(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_1 = FlexibleDense(out_features=3)
        self.dense_2 = FlexibleDense(out_features=2)

    def call(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the expected input of MyModel
    # Since FlexibleDense is a Dense layer, it expects input with last dim matching
    # input features dimension. Here we can assume input features dimension = 5.
    # Batch size can be any positive integer, e.g., 4.
    B, D = 4, 5  # Batch size, input feature dimension (arbitrary example)
    return tf.random.uniform((B, D), dtype=tf.float32)

