# tf.random.uniform((1000, 784), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define model layers matching the original example in the issue
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate the model and compile it as in the example
    model = MyModel()
    model.compile(
        loss="binary_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"]
    )
    return model

def GetInput():
    # Return input tensor matching the shape expected by MyModel:
    # [batch_size=1000, features=784], dtype float32
    # This mimics the original example's data input.
    return tf.random.uniform(shape=(1000, 784), maxval=15, dtype=tf.float32)

