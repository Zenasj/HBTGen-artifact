# tf.random.uniform((batch_size, 224, 224, 3), dtype=tf.float32) for images and (batch_size, 1, 96, 1366) for audio

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Assumptions:
        # There are multiple image inputs (8), each image size 224x224x3,
        # plus one audio input (1,96,1366).
        #
        # Since the original model details are not fully shared,
        # we assume a simple CNN backbone for images (shared weights),
        # then concatenating features and process audio separately,
        # finally combine all features for classification (e.g. softmax).
        #
        # This is a reasonable construction based on typical multi-input CNN+Audio modeling.
        
        # Image feature extractor: use a simple CNN applied per image input
        self.image_cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
        ])
        # Audio feature extractor (assume 1x96x1366 input channels last),
        # input shape per sample (1,96,1366) - treat as a single channel 2D signal
        self.audio_cnn = tf.keras.Sequential([
            tf.keras.layers.Permute((2,3,1)),  # reorder dims if needed; assuming input (batch,1,96,1366)
            tf.keras.layers.Reshape((96,1366,1)), # single channel for Conv2D
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
        ])
        # Final combined layers
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')  # assuming 10 classes for example
        ])

    def call(self, inputs, training=False):
        """
        inputs is a list:
          [img1, img2, ..., img8, audio]
        img_i each shape: (batch, 224, 224, 3)
        audio shape: (batch, 1, 96, 1366)
        
        Process each image individually through image_cnn (shared weights),
        concatenate all image features,
        process audio through audio_cnn,
        concatenate audio features with image features,
        apply classifier and return output.
        """
        image_features = []
        # The first 8 inputs are images
        for i in range(len(inputs) - 1):
            img = inputs[i]  # shape (batch, 224, 224, 3)
            feat = self.image_cnn(img)  # shape (batch, 128)
            image_features.append(feat)
        # concatenate image features: shape (batch, 8*128)
        img_concat = tf.concat(image_features, axis=-1)
        
        audio = inputs[-1]  # shape (batch, 1, 96, 1366)
        # to adapt the input shape for audio_cnn:
        # in call, input shape: (batch, 1, 96, 1366),
        # we reshape audio to (batch, 96, 1366, 1) assuming channels last
        batch_size = tf.shape(audio)[0]
        audio_reshaped = tf.reshape(audio, (batch_size, 96, 1366, 1))
        audio_feat = self.audio_cnn(audio_reshaped)  # shape (batch, 128)

        # concatenate image features + audio features
        combined = tf.concat([img_concat, audio_feat], axis=-1)
        # classifier to output predictions
        output = self.classifier(combined)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel.
    # 8 images each (batch_size,224,224,3), 1 audio input (batch_size,1,96,1366)
    batch_size = 4  # example batch size

    images = [
        tf.random.uniform((batch_size, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)
        for _ in range(8)
    ]
    # audio input with shape (batch, 1, 96, 1366)
    audio = tf.random.uniform((batch_size, 1, 96, 1366), dtype=tf.float32)

    inputs = images + [audio]
    return inputs

