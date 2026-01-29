# tf.random.uniform((B=10, H=224, W=224, C=3), dtype=tf.float32) ← Input shape from the original example: (10, 224, 224, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base MobileNetV2 model with ImageNet weights, without top layer
        # Note: alpha=1.4 as per original code, pooling='avg'
        self.base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            alpha=1.4,
            input_shape=(224, 224, 3),
            pooling='avg',
        )
        # Wrap base_model in TimeDistributed layer to process sequences of length 10
        self.time_distributed = tf.keras.layers.TimeDistributed(self.base_model)

        # Final dense layer with sigmoid activation to produce output per time step
        self.prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 10, 224, 224, 3)
        x = self.time_distributed(inputs)
        # x shape: (batch_size, 10, base_model_output_features)
        out = self.prediction_layer(x)
        # Output shape: (batch_size, 10, 1)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model input shape
    # Batch size arbitrary, here 1 for simplicity
    # Shape: (batch_size=1, 10 frames, 224, 224, 3 channels)
    return tf.random.uniform(shape=(1, 10, 224, 224, 3), dtype=tf.float32)

