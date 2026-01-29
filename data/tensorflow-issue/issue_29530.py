# tf.random.uniform((B, 64, 64, 3), dtype=tf.float32) ← Input shape (64,64,3), batch size variable

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(64, 64, 3), num_classes=2, use_lambda=False):
        super().__init__()
        self.use_lambda = use_lambda

        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.act1 = tf.keras.layers.Activation("relu")
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)

        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.act2 = tf.keras.layers.Activation("relu")
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)

        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.act3 = tf.keras.layers.Activation("relu")
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, use_bias=False)
        self.bn4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.act4 = tf.keras.layers.Activation("relu")
        self.dropout = tf.keras.layers.Dropout(0.25)

        if use_lambda:
            # Lambda layer that doubles the input tensor (to simulate mini_cnn_with_lambda)
            self.lambda_layer = tf.keras.layers.Lambda(lambda x: 2.0 * x)
        else:
            self.lambda_layer = None

        self.logits = tf.keras.layers.Dense(num_classes, use_bias=True, name="logits")
        self.probas = tf.keras.layers.Activation("softmax", name="probas")

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn4(x, training=training)
        x = self.act4(x)
        x = self.dropout(x, training=training)

        if self.lambda_layer is not None:
            x = self.lambda_layer(x)

        logits = self.logits(x)
        probas = self.probas(logits)
        return probas


def my_model_function(use_lambda=False):
    # Return an instance of MyModel
    # Default input shape (64, 64, 3) and 2 classes as per issue example
    return MyModel(input_shape=(64, 64, 3), num_classes=2, use_lambda=use_lambda)


def GetInput():
    # Returns a random float32 tensor of shape (batch_size=1, 64, 64, 3)
    return tf.random.uniform(shape=(1, 64, 64, 3), dtype=tf.float32)

