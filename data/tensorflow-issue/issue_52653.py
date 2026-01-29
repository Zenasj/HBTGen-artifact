# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← Input shape inferred from MobileNetV2 input_shape=(224,224,3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate the MobileNetV2 base model without the top classification layer
        # As per the original model generation:
        # input_shape=(224, 224, 3), alpha=1.0, weights='imagenet', classes=6,
        # include_top=False, pooling='avg', dtype float32 for the Dense output
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            alpha=1.0,
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        # The final Dense layer with 6 classes (inferred from the issue)
        # Set dtype explicitly to float32 to match usage
        self.logits = tf.keras.layers.Dense(6, name='logits', dtype='float32')

    def call(self, inputs, training=False):
        # Forward pass through the base MobileNetV2 model
        x = self.base_model(inputs, training=training)
        # Linear classification layer producing logits
        outputs = self.logits(x)
        return outputs

def my_model_function():
    # Return an instance of the MyModel
    # Weights by default will be loaded from imagenet for base_model, logits random init
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching the input shape of MobileNetV2: (1, 224, 224, 3)
    # Use batch size 1 as the example inference batch size
    return tf.random.uniform(shape=(1, 224, 224, 3), minval=0, maxval=1, dtype=tf.float32)

