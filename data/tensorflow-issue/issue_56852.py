# tf.random.normal([1, 100, 64], dtype=tf.float32)  ← Inferred input shape and dtype from example usage in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumption:
        # - The encoder is a pretrained model loaded from SavedModel format, frozen/non-trainable.
        # - The decoder is a simple sequential model with a Dense(2) layer.
        # Since we can't load the actual model here, we simulate the architecture.
        
        # Simulated encoder: a simple layer that mimics output embedding of shape (None, None, emb_dim)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, 64)),
            tf.keras.layers.Dense(32, activation='relu', trainable=False),  # frozen encoder
            tf.keras.layers.Dense(16, activation='relu', trainable=False)
        ])
        self.encoder.trainable = False  # explicitly frozen
        
        # Decoder: single Dense layer outputting 2 units (like in original)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(2)
        ])
    
    def call(self, inputs, training=False):
        # Forward pass using frozen encoder then decoder
        emb = self.encoder(inputs, training=False)  # encoder frozen, no training mode
        outputs = self.decoder(emb)
        return outputs


def my_model_function():
    # Return an instance of MyModel
    # According to the original issue, the pretrained encoder is frozen
    # Returned model corresponds to the final assembled model
    return MyModel()


def GetInput():
    # Return a random tensor compatible with MyModel input
    # From original issue, input shape is [batch=1, timesteps=100, feats_dim=64]
    return tf.random.normal([1, 100, 64], dtype=tf.float32)

