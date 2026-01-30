import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        tf.print("tf.executing_eagerly() =", tf.executing_eagerly())
        return inputs


def get_model():
    inp = tf.keras.layers.Input(shape=(1,))
    out = MyLayer(8)(inp)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.summary()
    return model


def train():
    model = get_model()
    model.compile(optimizer="adam", loss="mae")
    x_train = [2, 3, 4, 1, 2, 6]
    y_train = [1, 0, 1, 0, 1, 1]
    model.fit(x_train, y_train)


if __name__ == '__main__':
    train()

tf.config.experimental_run_functions_eagerly(True)

from tensorflow.keras import layers


class Linear(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'),
                         trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                              dtype='float32'),
                         trainable=True)

  def call(self, inputs):
    print("tf.executing_eagerly() =", tf.executing_eagerly())
    return tf.matmul(inputs, self.w) + self.b

x = tf.ones((1, 2)) # returns tf.executing_eagerly() = True
#x = tf.keras.layers.Input(shape=(2,)) #tf.executing_eagerly() = False
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y) 
#output in graph mode: Tensor("linear_9/Identity:0", shape=(None, 4), dtype=float32)
#output in Eager mode: tf.Tensor([[-0.03011466  0.02563028  0.01234017  0.02272708]], shape=(1, 4), dtype=float32)

from tensorflow import keras
from tensorflow.keras import layers
class CustomDense(layers.Layer):
  def __init__(self, units=32):
    super(CustomDense, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    print("tf.executing_eagerly() =", tf.executing_eagerly())
    return tf.matmul(inputs, self.w) + self.b


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)

tf.compat.v1.enable_v2_behavior()