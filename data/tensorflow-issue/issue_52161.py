# tf.random.uniform((B, 150, 150, 3), dtype=tf.float32) ← Input shape inferred from model input_shape=(150,150,3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The Sequential CNN model described in the issue
        self.conv1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D(2, 2)
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2,2)
        self.conv3 = tf.keras.layers.Conv2D(128, (3,3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(2,2)
        self.conv4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(2,2)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        # Output layer for 3 classes with softmax activation
        self.dense2 = tf.keras.layers.Dense(3, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Instantiate the model and compile to match code from issue
    model = MyModel()
    # Compile using RMSprop and categorical_crossentropy (note: loss fixed here to 'categorical_crossentropy' string)
    model.compile(
        optimizer=tf.optimizers.RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a batch of random float32 tensors normalized between 0 and 1 to mimic ImageDataGenerator output
    # Batch size: 32 (assumed), image size: 150x150, channels:3
    batch_size = 32
    input_shape = (batch_size, 150, 150, 3)
    # Random float in [0,1], dtype=float32
    return tf.random.uniform(input_shape, dtype=tf.float32)

