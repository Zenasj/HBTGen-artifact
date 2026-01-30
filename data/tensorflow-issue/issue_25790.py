from tensorflow.keras import layers

use_tf_keras = True
if use_tf_keras:
    from tensorflow.keras.layers import Layer, Dense
    from tensorflow.keras import backend as K
    from tensorflow.keras import Sequential
else:
    from keras.layers import Layer, Dense
    from keras import backend as K
    from keras import Sequential


class MyLayer(Layer):

  def build(self, input_shape):
    print(type(input_shape))
    input_dim = input_shape[-1]
    output_dim = input_shape[-1] / 2

    self.kernel = self.add_weight(shape=(input_dim, output_dim),
                                  name='kernel',
                                  initializer='ones')
    super().build(input_shape)

  def call(self, inputs):
    return K.dot(inputs, self.kernel)

  def compute_output_shape(self, input_shape):
    input_shape = list(input_shape)
    input_shape[-1] = input_shape[-1] // 2
    return input_shape

model = Sequential()
model.add(Dense(8, input_shape=(20, 20)))
model.add(MyLayer())