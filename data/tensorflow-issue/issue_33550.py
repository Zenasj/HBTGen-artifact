from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

features = np.array([[-1.], [1.]], dtype=np.float32)
labels = np.array([[0], [1]], dtype=np.int32)

dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(1)

model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='sgd', loss='binary_crossentropy')

class_weight = {'bad_negative_label': 0.5, 'bad_positive_label': 0.5}

fit_ops = (('tf.data.Dataset', lambda: model.fit(dataset, class_weight=class_weight, verbose=0)),
           ('np.ndarray', lambda: model.fit(features, labels, class_weight=class_weight, verbose=0)))

for key, fit_op in fit_ops:
    try:
        print(f'fitting {key} with bad class_weight label')
        fit_op()
    except ValueError as e:
        print('failed as it should have')
        raise ValueError from e
    else:
        print('succeded but should have failed')