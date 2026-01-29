# tf.random.uniform((B, 3, 229, 1), dtype=tf.float32)  # Inferred input shape from Input((3,229,1))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following layers from the provided model in the issue description
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(0.25)
        self.pool1 = tf.keras.layers.MaxPooling2D((1,2))

        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((1,2))
        
        # LSTM layer, units=256 as per original
        self.lstm = tf.keras.layers.LSTM(256)
        self.dense = tf.keras.layers.Dense(1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)

        # Reshape to (batch, time_steps, features) to feed into LSTM
        # Time dimension inferred as "width" dimension after pooling, features as channels * height
        shape = tf.shape(x)
        batch_size = shape[0]
        time_steps = shape[1] * shape[2]  # height * width dimension combined as time steps
        features = x.shape[-1]  # known channels dimension

        # As in original code,
        # x = tf.keras.layers.Reshape((-1, tf.keras.backend.int_shape(x)[-1] * tf.keras.backend.int_shape(x)[-2]))(x)
        # Here for dynamic shape, flatten dims 1 and 2 and keep channels as features
        x = tf.reshape(x, (batch_size, -1, features))

        x = self.lstm(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x


def my_model_function():
    # Instantiate and return the defined model
    # No pretrained weights given, so default initialized
    return MyModel()


def GetInput():
    # Generate random input tensor matching expected input shape: (batch_size, 3, 229, 1)
    # Use batch size 1 for simplicity
    # dtype set to tf.float32 matching typical float input in TensorFlow
    return tf.random.uniform((1, 3, 229, 1), dtype=tf.float32)

