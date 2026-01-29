# tf.random.uniform((B, 200), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original model architecture:
        # Input shape: (None, 200)
        # Layers: Dense 1024 tanh -> Dense 1024 tanh -> Dense 6 relu
        self.layer0 = tf.keras.layers.Dense(1024, activation='tanh', name="layer0", input_shape=(200,))
        self.layer1 = tf.keras.layers.Dense(1024, activation='tanh', name="layer1")
        self.final_layer = tf.keras.layers.Dense(6, activation='relu', name="final_layer")

    def call(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        output = self.final_layer(x)
        return output


def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Build the model by calling on a dummy input to initialize weights
    dummy_input = tf.zeros((1, 200), dtype=tf.float32)
    model(dummy_input)
    return model


def GetInput():
    # Generate a random float32 tensor of shape (batch_size=4, 200)
    # Batch size 4 is an arbitrary small batch size for demonstration
    return tf.random.uniform((4, 200), dtype=tf.float32)

