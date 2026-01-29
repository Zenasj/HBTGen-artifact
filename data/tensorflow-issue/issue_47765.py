# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base MobileNetV2 model without pretrained weights and no top
        self.base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights=None, input_shape=(224, 224, 3)
        )
        self.base_model.trainable = True
        
        # Rescaling layer as in original model: scale input pixels from [0..255] to [-1..1]
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)
        
        # Global average pooling after base model features extraction
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        
        # Dense + Dropout layers as per original model
        self.dense1 = layers.Dense(512, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.classifier = layers.Dense(30, activation="softmax")

    def call(self, inputs, training=False):
        x = self.rescale(inputs)
        # The original issue was that training=True caused problems during TFLite conversion,
        # so here we keep training flag passed through call accordingly
        features = self.base_model(x, training=training)
        pooled = self.global_avg_pool(features)
        dense_out = self.dense1(pooled)
        dropped = self.dropout(dense_out, training=training)
        return self.classifier(dropped)

def my_model_function():
    # Return an instance of MyModel (untrained, weights=None as in original)
    return MyModel()

def GetInput():
    # Input shape: batch_size 1, height 224, width 224, channels 3
    # Value range: typical uint8 image values [0..255]; we generate float since rescale expects float
    return tf.random.uniform((1, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)

