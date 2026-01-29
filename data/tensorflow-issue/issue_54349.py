# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Input shape and dtype inferred from the code: input shape=(224,224,3), dtype='float32'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model emulates the fusion of a preprocessing pipeline (Normalization + Rescaling)
        # with a ResNet50V2 backbone loaded with ImageNet weights,
        # mimicking the construction logic and preserving the dtype handling.
        # The mean and variance are provided explicitly to Normalization layer.

        # Normalization layer: per-channel mean and variance as given
        self.normalization = tf.keras.layers.Normalization(
            mean=[118.662, 119.194, 96.877],
            variance=[2769.232, 2633.742, 2702.492],
            axis=-1,
            dtype=tf.float32,
            name="normalization"
        )
        # Rescaling layer: scale inputs from [0, 255] to [-1, 1]
        self.rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(
            scale=1./127.5,
            offset=-1,
            dtype=tf.float32,
            name="rescaling"
        )

        # Load ResNet50V2 with pretrained imagenet weights, include_top=True
        base_resnet = tf.keras.applications.ResNet50V2(include_top=True, weights='imagenet', input_shape=(224,224,3))
        # We extract the output of the 'avg_pool' layer (the global pooling layer)
        transfer_layer = base_resnet.get_layer('avg_pool')
        self.resnet_submodel = Model(inputs=base_resnet.input, outputs=transfer_layer.output)

        # Freeze all layers initially
        for layer in self.resnet_submodel.layers:
            layer.trainable = False
        # Then set all trainable (mirroring the given code which sets trainable=True on all layers)
        for layer in self.resnet_submodel.layers:
            layer.trainable = True

        # Classification head Dense layer for 1000 classes with softmax activation
        # Also specifying dtype='float32' to ensure compatibility with mixed precision.
        self.classifier = Dense(1000, activation='softmax', dtype=tf.float32, name='species')

    def call(self, inputs, training=False):
        # Forward pass:
        # 1) Normalize inputs (subtract mean, divide by sqrt(variance))
        x = self.normalization(inputs)
        # 2) Rescale from normalized values to [-1,1] range
        x = self.rescaling(x)
        # 3) Forward through ResNet50V2 backbone
        x = self.resnet_submodel(x, training=training)
        # 4) Classification output
        x = self.classifier(x)
        return x

def my_model_function():
    # Return an instance of the model.
    # Loads pretrained weights automatically via ResNet50V2.
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input: shape (batch, 224, 224, 3), dtype float32
    # Batch size is assumed 1 as the original code does not specify batch
    # Random values in uint8 image range [0,255], cast to float32 for input.
    # This mirrors typical image input before normalization.
    batch_size = 1
    return tf.random.uniform(shape=(batch_size, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)

