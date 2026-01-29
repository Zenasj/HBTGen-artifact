# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the original Sequential model from the issue:
        # Input shape: (224, 224, 3)
        # Layers: GlobalAveragePooling2D, Dense(1)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(1)

        # We create an input spec to define the expected input shape
        self._input_spec = tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)

    def call(self, inputs, training=False):
        x = inputs
        x = self.pool(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Instantiate the model and build it explicitly
    model = MyModel()
    # Build with batch size none and the known input shape
    model.build(input_shape=(None, 224, 224, 3))
    return model

def GetInput():
    # Return a random input tensor matching the expected input shape
    # We use batch size 16 to match the original example's batch size
    return tf.random.uniform((16, 224, 224, 3), dtype=tf.float32)

