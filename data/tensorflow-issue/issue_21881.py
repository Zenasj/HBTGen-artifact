# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← The input shape is batch size unknown, 32x32 RGB images

import tensorflow as tf

# Because the core model in the shared code uses 'channels_first' format (NCHW),
# but input placeholder is NHWC (batch, 32, 32, 3),
# the model transposes input accordingly.

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Defining convolutional layers similar to the TF code with use_bias=True
        # and BN+ReLU activations (implemented here with tf.keras.layers.BatchNormalization + ReLU).

        # Note: For simplicity and compatibility with tf2.20, use standard Keras layers.
        # The original code used tensorpack and tflearn globals; 
        # here we replicate the architecture as closely as possible.

        # We'll implement all conv layers with kernel size 3, padding same, no bias disabling.

        # Define layers with names similar to original for clarity.

        conv_filters = [
            ('conv1_1', 66),
            ('conv1_2', 128),
            ('conv2_1', 128),
            ('conv2_2', 128),
            ('conv2_3', 192),
            # MaxPool here
            # Dropout 0.05 keep prob means dropout rate=0.05 => 0.95 kept
            ('conv4_1', 192),
            ('conv4_2', 192),
            ('conv4_3', 192),
            ('conv4_4', 192),
            ('conv4_5', 288),
            # MaxPool here
            # Dropout again 0.05 rate
            ('conv5_1', 288),
            ('conv5_2', 355),
            ('conv5_3', 432),
        ]

        # Using channels_first, so input shape after transpose is (B, 3, 32, 32)

        self.conv_blocks = []
        for name, filters in conv_filters:
            self.conv_blocks.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters,
                                           kernel_size=3,
                                           padding='same',
                                           use_bias=True,
                                           data_format='channels_first',
                                           name=name),
                    tf.keras.layers.BatchNormalization(axis=1, name=name + '_bn'),
                    tf.keras.layers.ReLU(name=name + '_relu'),
                ], name=name + '_block')
            )

        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first', name='pool2_1')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.05, name='dropout1')  # keep_prob=0.95 => rate=0.05

        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first', name='pool2_2')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.05, name='dropout2')

        # After last conv layers, apply global max pooling instead of global avg pooling.
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling2D(data_format='channels_first', name='global_max_pool')

        # Fully connected final layer:
        self.fc = tf.keras.layers.Dense(num_classes, name='linear')

    def call(self, inputs, training=False):
        # inputs shape: (B, 32, 32, 3) NHWC
        # transpose to channels_first: (B, 3, 32, 32)
        x = tf.transpose(inputs, [0, 3, 1, 2])

        # Pass through conv layers as per original sequence:
        # conv1.1, conv1.2, conv2.1, conv2.2, conv2.3
        for i in range(5):
            x = self.conv_blocks[i](x, training=training)

        x = self.pool1(x)
        x = self.dropout1(x, training=training)

        # conv4.1 to conv4.5 (5 layers)
        for i in range(5, 10):
            x = self.conv_blocks[i](x, training=training)

        x = self.pool2(x)
        x = self.dropout2(x, training=training)

        # conv5.1 to conv5.3 (3 layers)
        for i in range(10, 13):
            x = self.conv_blocks[i](x, training=training)

        # global max pool
        x = self.global_max_pool(x)  # shape (B, channels)

        logits = self.fc(x)  # shape (B, num_classes)

        return logits


def my_model_function():
    # Instantiate the model with 10 classes (CIFAR-10)
    model = MyModel(num_classes=10)
    # Normally weights are randomly initialized; pretrained weights not provided
    return model


def GetInput():
    # Return a random tensor matching input expected by MyModel
    # Shape: (batch_size, height=32, width=32, channels=3)
    # Using batch size 100 as per original batch size in example
    batch_size = 100
    # Using float32 input values in range [0, 1)
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)

