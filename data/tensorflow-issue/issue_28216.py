import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_layers = 2
num_units = 128

rnn = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    num_units,
                    forget_bias=0.0,
                    state_is_tuple=False,
                )
                for _ in range(num_layers)
            ],
            state_is_tuple=False,
        )

# x.shape == (?, 128)  and state.shape == (?, 512)

x, state = rnn(x, state)

# x.shape == (?, 128)  and state.shape == (?, 512)

num_layers = 2
num_units = 128
batch_size = 20
embedding_size = 50

cells = [tf.keras.layers.LSTMCell(num_units) for _ in range(num_layers)]
rnn = tf.keras.layers.StackedRNNCells(cells)

# shape of x: [B, D]
x = tf.Variable(tf.random.normal([batch_size, embedding_size]), dtype=tf.float32)
# shape of state: (([B, H], [B, H]), ([B, H], [B, H]))
state = rnn.get_initial_state(x)

# shape of output: [B, H]
# shape of new_state: (([B, H], [B, H]), ([B, H], [B, H]))
output, new_state = rnn(x, state)

print(output)
print(new_state)

class MultiCellRNN(tf.keras.layers.Layer):

    def __init__(self, num_layers, num_units, **kwargs):
        super().__init__(**kwargs)

        self.cells = [
            tf.keras.layers.LSTMCell(
                num_units,
            )
            for _ in range(num_layers)
        ]

        self.cell = tf.keras.layers.StackedRNNCells(
            self.cells,
        )

    def call(self, x, state, **kwargs):

        state = tf.split(state, [128, 128, 128, 128], 1)
        x, state = self.cell(x, ([state[0], state[1]], [state[2], state[3]]))
        state = tf.concat([state[0][0], state[0][1], state[1][0], state[1][1]], 1)

        return x, state