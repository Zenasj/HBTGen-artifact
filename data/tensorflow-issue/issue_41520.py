import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

def turn_into_ragged(x):
    return tf.RaggedTensor.from_row_lengths(tf.concat(
        [tf.RaggedTensor.from_row_lengths(np.concatenate(sample, axis=0), list(map(len, sample))) for sample in x],
        axis=0), list(map(len, x)))

def get_batches_m(x,y,batch_size=10,random=False):
    """ Return a generator that yields batches from vars. """
    #batch_size = len(x) // n_batches
    if len(x[0]) % batch_size == 0:
        n_batches = (len(x[0]) // batch_size)
    else:
        n_batches = (len(x[0]) // batch_size) + 1

    sel = np.asarray(list(range(x[0].shape[0])))
    if random is True:
        np.random.shuffle(sel)

    for ii in range(0, n_batches * batch_size, batch_size):
        # If we're not on the last batch, grab data with size batch_size
        if ii != (n_batches - 1) * batch_size:
            sel_ind=sel[ii: ii + batch_size]
        else:
            sel_ind = sel[ii:]

        x_out = [turn_into_ragged(var[sel_ind]) for var in x]
        y_out = [var[sel_ind] for var in y]

        yield tuple(x_out),tuple(y_out)

def generate_syn_data(n_i=2000, n_s=30, n_t=200, shape=(10, 10, 3)):
    values = np.random.uniform(0, 1, (n_i, ) + shape).astype(np.float32)
    idx_t = np.random.choice(n_t, n_i)
    _, l = np.unique(idx_t, return_counts=True)
    rt0 = tf.RaggedTensor.from_row_lengths(values, l)
    idx_s = np.random.choice(n_s, n_t)
    _, l = np.unique(idx_s, return_counts=True)
    rt1 = tf.RaggedTensor.from_row_lengths(rt0, l)
    y = tf.constant(np.eye(2)[np.random.choice(2, n_s)].astype(np.float32))
    return  rt1, y

def basic_ragged_graph(input_shapes):
    ragged_inputs = [tf.keras.layers.Input(shape=(None, None) + shape, dtype=tf.float32, ragged=True) for shape in input_shapes]
    sample_aggregation = tf.concat([tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=(1, 2)))(ragged_input) for ragged_input in ragged_inputs], axis=1)
    logits = tf.keras.layers.Dense(units=2, activation=None)(tf.keras.layers.Flatten()(sample_aggregation))
    return tf.keras.Model(inputs=ragged_inputs, outputs=[logits])

#create simple model that takes 2 ragged inputs and returns 1 output
tile_shape = (10, 10, 3)
model = basic_ragged_graph([tile_shape,tile_shape])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

#generate synthetic data for inputs and outputs
x1,x2 = generate_syn_data()[0].numpy(),generate_syn_data()[0].numpy()
y = generate_syn_data()[1].numpy()

#create generator objeect to batch and convert data to tf raggged
train_gen = get_batches_m([x1,x2],[y],batch_size=5,random=True)
#this method uses generator and outputs x_train,y_train that work
for x_train, y_train in train_gen:
    model.fit(x_train, y_train)

#however, when one provides this generator to the model.fit, the model will not train
train_gen = get_batches_m([x1,x2],[y],batch_size=5,random=True)
model.fit(train_gen)