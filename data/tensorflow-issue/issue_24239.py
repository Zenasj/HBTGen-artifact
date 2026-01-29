# tf.random.uniform((4, 150, 150, 3), dtype=tf.float32)  ← Assumed input shape (batch_size=4, height=150, width=150, channels=3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # Define the layers similar to the original Keras Sequential model given in the issue.
        # The original model uses ReLU activations, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, and sigmoid output.
        
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel, mimicking the original keras Sequential model architecture
    model = MyModel()
    # The original model was compiled with rmsprop optimizer, binary_crossentropy loss, and accuracy metric.
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Returns a random batch of input images: batch size 4, 150x150 RGB images with float32 dtype in [0,1] range
    # This matches the model's expected input shape.
    return tf.random.uniform((4, 150, 150, 3), minval=0., maxval=1., dtype=tf.float32)

