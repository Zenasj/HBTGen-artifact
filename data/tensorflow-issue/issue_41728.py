import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class GRUCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = tf.TensorShape([units])
        self.output_size = tf.TensorShape([units])
        super(GRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[-1]

        self.w_r = self.add_weight(shape=[self.dim+self.units, self.units], initializer='uniform', name='reset_gate', trainable=True)
        self.b_r = self.add_weight(shape=[self.units], initializer='zeros', name='reset gate bias', trainable=True)

        self.w_z = self.add_weight(shape=[self.dim+self.units, self.units], initializer='uniform', name='update_gate', trainable=True)
        self.b_z = self.add_weight(shape=[self.units], initializer='zeros', name='update gate bias', trainable=True) 

        self.w_n = self.add_weight(shape=[self.dim+self.units, self.units], initializer='uniform', name='intetim', trainable=True)
        self.b_n = self.add_weight(shape=[self.units], initializer='zeros', name='interim gate bias', trainable=True)

        self.build = True


    def call(self, inputs, states):
        
        states, = states
        
        r = tf.nn.sigmoid(inputs @ self.w_r[:self.dim] + states @ self.w_r[self.dim:] + self.b_r)
        z = tf.nn.sigmoid(inputs @ self.w_z[:self.dim] + states @ self.w_z[self.dim:] + self.b_z)
        n = tf.nn.tanh(inputs @ self.w_n[:self.dim] + (states * r) @ self.w_n[self.dim:] + self.b_n)
        
        output = (1 - z) * states + z * n

        return output, output

    def get_config(self):
        return {"units": self.units}

def create_model(units):
    model = keras.Sequential([
            keras.layers.RNN(GRUCell(units)),
            keras.layers.Dense(1)
            ])
    model.compile(optimizer=keras.optimizers.RMSprop(),
                loss='mae', metrics=['mse'])

    return model