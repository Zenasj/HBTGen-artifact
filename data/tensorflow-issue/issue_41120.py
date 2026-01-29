# tf.random.uniform((128, 300, 300, 3), dtype=tf.float32) ← inferred input shape from batch_size=128, target_size=(300,300), 3 channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the Sequential architecture from the issue
        
        self.conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3))
        self.pool1 = tf.keras.layers.MaxPooling2D(2,2)
        
        self.conv2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2,2)
        
        self.conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(2,2)
        
        self.conv4 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(2,2)
        
        self.conv5 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.pool5 = tf.keras.layers.MaxPooling2D(2,2)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        # Forward pass
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.pool5(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


def my_model_function():
    # Return an instance of MyModel with freshly initialized weights
    model = MyModel()
    # Compile with the loss and optimizer used in the original code
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def GetInput():
    # Generate a random input tensor with batch_size=128, 300x300 RGB image data in [0,1]
    # The original ImageDataGenerator rescales inputs by 1/255, so inputs are floats roughly in [0,1]
    return tf.random.uniform((128, 300, 300, 3), minval=0, maxval=1, dtype=tf.float32)

