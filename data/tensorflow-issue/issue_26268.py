import tensorflow as tf
tf.enable_v2_behavior()
from tensorflow import keras

class MyModel(keras.Model):
    def call(self, x):
        if x > 0:
            return x + 1
        else:
            return x - 1
            
m = MyModel()
m(tf.constant(0))  # This works, returns -1 as expected
m.compile(loss='mse', optimizer='sgd')
m.fit(tf.constant(0), tf.constant(1)) # This fails

m = MyModel(dynamic=True)

model.compile()
model.run_eagerly = True