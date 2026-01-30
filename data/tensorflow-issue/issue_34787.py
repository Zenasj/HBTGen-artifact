from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras import *

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

import tensorflow.keras.backend as K

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        self.a = None
  
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                        shape=[int(input_shape[-1]), self.num_outputs])
  
    def call(self, inputs):
        self.a = K.dot(inputs, self.kernel)
        return K.dot(inputs, self.kernel)

layer1 = MyDenseLayer(64)
input_tensor = Input(shape=(28, 28))
x = layers.Flatten()(input_tensor)
x = layer1(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor,  output_tensor)

model.summary()

adam = tf.keras.optimizers.Adam()
model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

for epoch in range(2):
    model.fit(x_train, y_train, batch_size=32, epochs=1)

layer = model.layers[2]
print('--------------------------------------------------')
print(layer.a)

layer.a.numpy()

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        self.a = tf.Variable([0]*num_outputs, dtype=tf.dtypes.float32)
  
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                        shape=[int(input_shape[-1]), self.num_outputs])
  
    def call(self, inputs):
        self.a.assign(K.dot(inputs, self.kernel))
        return K.dot(inputs, self.kernel)

layer1 = MyDenseLayer(64)
input_tensor = Input(shape=(28, 28))
x = layers.Flatten()(input_tensor)
x = layer1(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor,  output_tensor)

model.summary()

adam = tf.keras.optimizers.Adam()
model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

for epoch in range(2):
    model.fit(x_train, y_train, batch_size=32, epochs=1)

layer = model.layers[2]
print('--------------------------------------------------')
print(layer.a)

layer.a.numpy()