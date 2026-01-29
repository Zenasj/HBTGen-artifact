# tf.random.uniform((B, 1000), dtype=tf.float32) ← Input shape inferred from example input shape (10000, 1000)

import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.enc = tf.keras.layers.Dense(units=20, activation="relu", name='encoder_layer')

    def call(self, inputs):
        x = inputs
        x = self.enc(x)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dec = tf.keras.layers.Dense(units=1000, activation="sigmoid", name='decoder_layer')

    def call(self, inputs):
        x = inputs
        x = self.dec(x)
        return x

class MyModel(tf.keras.Model):
    def __init__(self, name="autoencoder", **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs):
        """
        Forward pass: encode inputs, then decode the latent representation to reconstruct input.
        """
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def train_step(self, data):
        """
        Custom train step applying mean squared error loss.
        """
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Compute mean squared error over the batch
            mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), axis=-1), axis=-1)
        gradients = tape.gradient(mse_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": mse_loss}


def my_model_function():
    """
    Returns an instance of MyModel with default initialization.
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor of shape (batch_size=32, 1000),
    matching the expected input shape of MyModel.

    The example used 10000 samples in the original snippet,
    but for a generic function we pick batch=32.
    
    Values are uniform floats in [0,1).
    """
    return tf.random.uniform(shape=(32, 1000), dtype=tf.float32)

