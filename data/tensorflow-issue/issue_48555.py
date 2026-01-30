import numpy as np
from tensorflow.keras import layers

import numpy
import tensorflow as tf
from tensorflow import keras

def test_class_weight_error():
    # the model simply return the input time_step*n_class=20x2 data as-is
    model = keras.Sequential([keras.layers.Reshape((20, 2), input_shape=(20, 2))])
    # run_eagerly improves the readability of the traceback a bit
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', run_eagerly=True)
    model.summary()
    # X (inputs, as well as y_pred): samples*time_step*n_classes=15x20x2
    xs = tf.reshape(tf.one_hot(tf.ones(300, dtype=tf.int32), 2), [-1, 20, 2]).numpy()
    # Y (labels i.e. y_true): samples*time_step=15x20, class labels of 0 or 1
    ys = np.ones([15, 20], dtype=np.int32)
    # without the line below (a.k.a. all labels being 1) there's no exception
    ys[:,:3] = 0
    # here's the crash
    model.fit(xs, ys, batch_size=3, class_weight={0:1.,1:1.})
    
test_class_weight_error()