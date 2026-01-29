# tf.random.uniform((32, 224, 224, 3), dtype=tf.float32) ← inferred input shape and dtype based on model training setup

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base model: MobileNetV2 without top layers, pretrained on imagenet
        self.base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        self.base_model.trainable = False

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(128, activation='relu')
        # Output number of classes is unknown at code time; 
        # Use placeholder 10 classes, user can adjust accordingly.
        # This must match number of classes used during training.
        self.pred_layer = Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=False)  # keep base_model in inference mode
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.pred_layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # Note: The final Dense layer default is 10 classes -- adjust if known.
    # Alternatively, allow passing num_classes if dynamic construction needed.
    return MyModel()

def GetInput():
    # Return a random tensor input matching input expected by MyModel.
    # Batch size of 32 to match original training batch size.
    return tf.random.uniform((32, 224, 224, 3), dtype=tf.float32)

