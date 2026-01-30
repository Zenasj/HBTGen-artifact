import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model): 
    def __init__(self, **kwargs): 
        super().__init__(**kwargs) 
        self.conv1 = [] 
        self.conv1.append([tf.keras.layers.Conv2D(8, 3, name="conv1")]) 
        self.conv2 = [] 
        self.conv2.insert(0, [tf.keras.layers.Conv2D(16, 3, name="conv2")]) 

    def call(self, inputs): 
        x = inputs 
        x = self.conv1[0][0](x) 
        x = self.conv2[0][0](x) 
        return x


m = MyModel()
m.build((None, None, None, 3))
m.summary()

for w in m.trainable_weights: 
    print(w.name)