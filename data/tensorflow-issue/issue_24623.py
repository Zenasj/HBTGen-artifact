# tf.random.uniform((B, 128), dtype=tf.float32) ← Input is a batch of vectors with dimension 128

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Single dense layer with tanh activation, input dimension 128, output 128
        self.encoder_layer = tf.keras.layers.Dense(128, activation='tanh')

    def call(self, inputs, training=None, mask=None):
        encoded = self.encoder_layer(inputs)
        return encoded

def my_model_function():
    # Return an instance of MyModel
    # Note: to properly load weights from h5 with this subclassed model,
    # you must run at least one forward pass or fit call before loading weights,
    # e.g. model(GetInput()) or model.fit on dummy data to initialize weights.
    return MyModel()

def GetInput():
    # Return a random tensor input of shape (batch_size, 128), dtype float32
    # Batch size is arbitrary; 32 chosen as typical default
    return tf.random.uniform((32, 128), dtype=tf.float32)

