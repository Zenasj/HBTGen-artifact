# tf.random.uniform((B, None, 224, 224, 3), dtype=tf.float32) ← inferred input: batch of variable-length sequences of images with shape 224x224x3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base model: MobileNet without top, pretrained on imagenet
        input_shape = (224, 224, 3)
        self.base_model = tf.keras.applications.MobileNet(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg',
        )
        
        # TimeDistributed wrapping the base model
        # The trick is that the base_model expects inputs of shape [batch, H, W, C].
        # TimeDistributed expects inputs of shape [batch, timesteps, H, W, C].
        # Since the base_model is a tf.keras.Model, wrapping with TimeDistributed causes issues
        # when saving in SavedModel format (runtime shape mismatch).
        # However, the functional calling still works.
        self.sequence_embedding = tf.keras.layers.TimeDistributed(self.base_model)
        
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, timesteps, 224, 224, 3)
        # output shape: (batch_size, timesteps, base_model_output_dim)
        return self.sequence_embedding(inputs)

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Create a random input tensor of shape (batch_size, timesteps, 224, 224, 3)
    # using float32 dtype consistent with MobileNet expectations.
    # Use batch_size=2 and timesteps=5 as reasonable defaults here.
    batch_size = 2
    timesteps = 5
    height = 224
    width = 224
    channels = 3
    return tf.random.uniform(
        (batch_size, timesteps, height, width, channels),
        minval=0, maxval=255, dtype=tf.float32
    )

