# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Input is batches of RGB images 224x224

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import vgg16

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3)):
        super().__init__()
        # Use VGG16 pretrained on ImageNet as base, exclude top FC layers
        # Freeze all layers except last 5 layers to allow fine-tuning
        with tf.device('/gpu:0'):
            base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
            for layer in base_model.layers[:-5]:
                layer.trainable = False
        
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            outputs = Dense(2, activation='softmax')(x)
            self.model = Model(inputs=base_model.input, outputs=outputs)
    
    @tf.function
    def call(self, inputs, training=False):
        # Forward pass through the VGG16-based model
        return self.model(inputs, training=training)


def my_model_function():
    # Create and return an instance of MyModel
    return MyModel(input_shape=(224, 224, 3))


def GetInput():
    # Return a random tensor input with batch size 8 of normalized images (float32 scaled 0-1)
    # Matches shape expected by MyModel
    # Typical input dtype for TF models is float32 with range [0, 1]
    batch_size = 8
    # Using random uniform here to simulate normalized RGB images
    return tf.random.uniform(shape=(batch_size, 224, 224, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

