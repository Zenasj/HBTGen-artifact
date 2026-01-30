import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.layers.Layer):

    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = tf.keras.layers.Dense(10)

    def call(self, x, cache=None, training=None):
        if training or cache is None:
            return self.linear(x)
        else:
            return tf.cond(
                tf.equal(tf.shape(cache)[0], 0),
                true_fn=lambda: self.linear(x),
                false_fn=lambda: cache)

x = tf.zeros([3, 5])
model = MyModel()
_ = model(x, training=True)
print(model.trainable_variables)

x = tf.zeros([3, 5])
cache = tf.ones([0, 5])
model = MyModel()
_ = model(x, cache=cache, training=False)
print(model.trainable_variables)

tf.get_default_graph().get_name_scope() + "/"