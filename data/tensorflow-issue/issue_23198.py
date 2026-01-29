# tf.random.uniform((B, 512, 512, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers, Model

# Based on the issue, the input shape is grayscale images resized to 512x512.
# The model is a CNN with Conv2D layers and Dense layers to classify 7 classes.
# The error was due to OOM on large inputs (512x512), typically caused by large feature maps or 
# flattening huge maps before dense layers. 
# So we implement the model as described but in a tf.keras.Model subclass format.
# We'll add MaxPooling to reduce spatial size and use standard 'same' padding.
# Assume float32 input.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D layers as in original Sequential model:
        # 32 filters, 3x3, relu, padding='same'
        self.conv1 = layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(512,512,1))
        self.conv2 = layers.Conv2D(32, (3,3), padding='valid', activation='relu')  # original had conv2d(32, 3,3) - assuming valid default
        self.pool1 = layers.MaxPooling2D((2,2))
        self.drop1 = layers.Dropout(0.5)
        
        self.conv3 = layers.Conv2D(64, (3,3), padding='valid', activation='relu')
        self.pool2 = layers.MaxPooling2D((2,2))
        self.drop2 = layers.Dropout(0.5)
        
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.drop3 = layers.Dropout(0.5)
        
        self.dense_out = layers.Dense(7, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop3(x, training=training)
        out = self.dense_out(x)
        return out

def my_model_function():
    # Return an instance of MyModel with compiled optimizer and loss as per original:
    model = MyModel()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a random input tensor matching the input shape (B, 512, 512, 1).
    # Use batch size 32 as used in original generator.
    # dtype float32 normalized [0,1] as per rescale=1./255.
    batch_size = 32
    height = 512
    width = 512
    channels = 1
    # Random uniform floats simulating normalized grayscale images.
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

