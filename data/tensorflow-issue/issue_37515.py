# tf.random.uniform((batch_size, 300, 300, 3), dtype=tf.float32) ← input shape inferred from datagenerator example PicX=300, PicY=300, 3 channels

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, pic_x, pic_y, color):
        self._BatchSize = batch_size
        self._dim = (pic_x, pic_y)
        self._Color = color
        
    def __len__(self):
        # Arbitrary length for example; consistent with example code
        return 100
        
    def create_random_form(self):
        # Creates a single synthetic image with a random shape and label
        img = Image.new('RGB', self._dim, (50,50,50))
        draw = ImageDraw.Draw(img)
        label = np.random.randint(3)
        # Coordinates roughly split image in halves/quadrants for shape position
        x0 = np.random.randint(int((self._dim[0]-5)/2)) + 1
        x1 = np.random.randint(int((self._dim[0]-5)/2)) + int(self._dim[0]/2)
        y0 = np.random.randint(int((self._dim[1]-5)/2))
        y1 = np.random.randint(int((self._dim[1]-5)/2)) + int(self._dim[1]/2)
        if label == 0:
            draw.rectangle((x0,y0,x1,y1), fill=self._Color)
        elif label == 1:
            draw.ellipse((x0,y0,x1,y1), fill=self._Color)
        else:
            draw.polygon([(x0,y0), (x0,y1), (x1,y1)], fill=self._Color)
        return img, label
        
    def __getitem__(self, index):
        # Generate one batch of images and labels as numpy arrays normalized [0,1]
        X = np.empty((self._BatchSize, *self._dim, 3), dtype=np.float32)
        y = np.empty((self._BatchSize), dtype=np.int32)
        for i in range(self._BatchSize):
            img, label = self.create_random_form()
            X[i] = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            y[i] = label
        return X, y

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple classification model matching the minimal reproducible example
        # Input shape is (300, 300, 3)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(600, activation='relu')
        self.dense2 = tf.keras.layers.Dense(150, activation='relu')
        self.out = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)

def my_model_function():
    # Return a compiled instance of MyModel for training
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

def GetInput():
    # Return a random tensor input to MyModel - shape (batch_size, 300, 300, 3)
    # Use batch_size=32 as a typical batch size for example
    batch_size = 32
    # Values in [0,1] float32 - matching DataGenerator output normalized images
    return tf.random.uniform((batch_size, 300, 300, 3), dtype=tf.float32)

