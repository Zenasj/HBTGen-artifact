import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MLP(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(1)
    @tf.function
    def call(self, inputs):
        a, b, mode = inputs
        if mode == 'predict':
            return a
        
        return self.dense(a + b),b

if __name__ == '__main__':
    a = tf.ones((2, 10))
    b = tf.ones((2, 10))
    mode = 'train'
    mode = tf.cast(mode, tf.string)
    model = MLP()
    print(model((a, b, mode)))
    model.save('test_model/1/')