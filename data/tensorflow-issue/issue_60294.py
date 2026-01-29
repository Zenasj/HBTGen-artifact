# tf.random.uniform((B, 128, 128, 3), dtype=tf.float32) ← Input shape matching the facial images resized to 128x128 RGB

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using a similar architecture to the training model described (train_face_id.py)
        # This model corresponds roughly to the conv-dropout-stack + dense,
        # consistent with input (128,128,3) and embedding 128-d output.

        # Conv2D and Dropout layers as described in the training script
        self.conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(128,128,3))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.drop1 = tf.keras.layers.Dropout(0.3)

        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.drop2 = tf.keras.layers.Dropout(0.3)

        self.conv3 = tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.drop3 = tf.keras.layers.Dropout(0.3)

        self.conv4 = tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.drop4 = tf.keras.layers.Dropout(0.3)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.drop5 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(128, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.drop4(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop5(x, training=training)
        output = self.dense2(x)
        return output


def my_model_function():
    # Instantiate the model and compile it with the triplet semi-hard loss and Adam optimizer,
    # mimicking the original training setup.
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.TripletSemiHardLoss(margin=3.0)
    )
    return model


def GetInput():
    # Return a batch of one random image tensor with shape (1, 128, 128, 3)
    # Float values typical for preprocessing (e.g. 0-1 scaled)
    return tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)

