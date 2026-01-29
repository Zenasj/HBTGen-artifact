# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

imsize = 28
color_channels = 1

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Encoder-decoder style autoencoder network as in minimal example
        self.network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(imsize, imsize, color_channels)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation='relu'),  # Adding activation to aid training
            tf.keras.layers.Dense(imsize * imsize * color_channels),
            tf.keras.layers.Reshape(target_shape=(imsize, imsize, color_channels)),
        ])

    def call(self, inputs):
        # Forward pass: encode and decode input to reconstruction logits
        logits = self.network(inputs)
        return logits

def my_model_function():
    """
    Instantiate and return MyModel instance, ready for training or saving.
    """
    return MyModel()

def GetInput():
    """
    Return a random tensor input matching model input shape:
    (batch_size, 28, 28, 1), dtype float32, values in [0,1)
    """
    BATCH_SIZE = 100  # same batch size as in original example
    return tf.random.uniform(
        (BATCH_SIZE, imsize, imsize, color_channels),
        minval=0.0, maxval=1.0,
        dtype=tf.float32
    )

