# tf.random.uniform((1, 150, 150, 3), dtype=tf.float32) ← input shape inferred from img_width=150, img_height=150, 3 channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Replicating the Keras Sequential model architecture from the issue

        # Block 1
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        # Block 2
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        # Block 3
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # Forward pass replicating the compiled Sequential model
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)

        x = self.dense2(x)
        return x

def my_model_function():
    # Return a model instance, weights not loaded since no checkpoint info provided
    model = MyModel()
    # Compile model similarly as in the issue for consistency
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random float32 tensor with shape (1, 150, 150, 3), normalized to [0,1]
    # This mimics a batch of one image for the model input and matches training input size.
    return tf.random.uniform((1, 150, 150, 3), dtype=tf.float32)

