# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape typical for EfficientNet-B0: (batch_size, 224, 224, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load EfficientNetB0 from tf.keras.applications (TF 2.x compatible)
        # We exclude top layers since original model uses custom dense layers on top.
        self.backbone = tf.keras.applications.EfficientNetB0(include_top=False, pooling='avg', weights='imagenet')
        
        # Following original reported model:
        # Dense(120), Dense(120), Dense(2).
        self.dense1 = tf.keras.layers.Dense(120, activation='relu')
        self.dense2 = tf.keras.layers.Dense(120, activation='relu')
        self.dense3 = tf.keras.layers.Dense(2, activation=None)
        
    def call(self, inputs, training=False):
        # Input should be float32, normalized [0,1]
        x = self.backbone(inputs, training=training)  # shape: (B, 1280)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor that fits EfficientNetB0 input requirements:
    # Typically 224x224 RGB images, float32, values ~[0,1].
    batch_size = 4  # assumed batch size for example
    height = 224
    width = 224
    channels = 3
    # Use uniform floats between 0 and 1 as typical normalized input
    return tf.random.uniform((batch_size, height, width, channels), minval=0.0, maxval=1.0, dtype=tf.float32)

