# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32)  ← Input shape inferred from VGG16 input_shape=(224,224,3)

import tensorflow as tf

NUM_CLASSES = 2  # Assuming binary classification as per dogs vs cats example

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The VGG16 base model as a submodule, without top layers and with average pooling
        # Disable training for transfer learning
        self.vgg_base = tf.keras.applications.VGG16(
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg',
            weights='imagenet',
        )
        self.vgg_base.trainable = False  # Freeze VGG16 layers

        # Custom fully connected layers on top of VGG16 base
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu')
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.prediction = tf.keras.layers.Dense(NUM_CLASSES)

    def call(self, inputs, training=False):
        # Forward pass
        x = self.vgg_base(inputs, training=False)  # pretrained base, no training
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        logits = self.prediction(x)
        return logits

def my_model_function():
    # Return an instance of MyModel, no pretrained weights need loading here,
    # as the VGG16 weights are automatically loaded by keras.applications
    return MyModel()

def GetInput():
    # Return a random input tensor shaped as expected: batch x height x width x channels
    # Batch size chosen as 4 as an example
    batch_size = 4
    # Uniform random input simulating image pixels between 0 and 255 as floats
    return tf.random.uniform((batch_size, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)

