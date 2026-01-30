import tensorflow as tf
from tensorflow.linalg import matmul
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import numpy as np

class MinimalRNNCell(tfk.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states=None, constants=None, *args, **kwargs):
        prev_output = states[0]
        print("constants: ", constants[0].name)
        h = matmul(inputs, self.kernel) + constants[0]
        output = h + matmul(prev_output, self.recurrent_kernel)
        return output, [output]

    def get_config(self):
        return dict(super().get_config(), **{'units': self.units})

cell = MinimalRNNCell(32)
x = tfk.Input((None, 5), name='x')
z = tfk.Input((1,), name='z')
layer = tfk.layers.RNN(cell, name='rnn')
y = layer(x, constants=[z])

model = tfk.Model(inputs=[x, z], outputs=[y])
model.compile(optimizer='adam', loss='mse')
model.predict([np.array([[[0,0,0,0,0]]]), np.array([[0]])])
model.save('tmp.model')

class MinimalRNNCell(tfk.layers.Layer):
  ... # same as above

#  Create an RNN subclass to bypass the shape inference logic
class RNN(tfk.layers.RNN):
  pass

cell = MinimalRNNCell(32)
x = tfk.Input((None, 5), name='x')
z = tfk.Input((1,), name='z')
layer = RNN(cell, name='rnn')  #  This now uses RNN instead of tfk.layers.RNN
y = layer(x, constants=[z])

model = tfk.Model(inputs=[x, z], outputs=[y])
model.compile(optimizer='adam', loss='mse')
model.predict([np.array([[[0,0,0,0,0]]]), np.array([[0]])])
model.save('tmp_model')

# To load the model, the custom cell must be passed into `custom_objects`
loaded = tfk.models.load_model('tmp_model',custom_objects={'MinimalRNNCell': MinimalRNNCell})