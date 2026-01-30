from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow import keras
import tensorflow as tf
import numpy as np


def get_model():
    inputs = keras.layers.Input(shape=(10, 10, 3))
    x = keras.layers.Flatten()(inputs)
    outputs = keras.layers.Dense(5)(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model


def map_fun(_):
    dummy_image = np.zeros((10, 10, 3))  
    dummy_label = np.array([0, 0, 1, 0, 0]) 
    return dummy_image, dummy_label


if __name__ == '__main__':
    # dummy dataset
    dataset = tf.data.Dataset.from_tensor_slices([1, 2])  # values are ignored, dummy data generated in map()
    dataset = dataset.map(map_func=lambda x: tf.py_function(map_fun, [x], [tf.uint8, tf.uint8])).batch(2)

    # dummy model
    model = get_model()

    # call fit() without class weights - ok
    model.fit(dataset, epochs=1)

    # define class weights
    class_weight = {idx: weight for (idx, weight) in enumerate([1., 1., 1., 1., 1.])}

    # transform dataset to iterator, call fit() with class weights - ok
    model.fit(dataset.as_numpy_iterator(), class_weight=class_weight, epochs=1)

    # call fit() with class weights on tf.data.Dataset - error
    model.fit(dataset, class_weight=class_weight, epochs=1)

if y.shape.rank > 2:   # <== this is where the error occurs, because y.shape.rank is None
      raise ValueError("`class_weight` not supported for "
                       "3+ dimensional targets.")

if class_weight:
      dataset = dataset.map(_make_class_weight_map_fn(class_weight))

def weighted_categorical_crossentropy( weights ):
    # weights = [ 0.9, 0.05, 0.04, 0.01 ]
    def wcce( y_true, y_pred ):
        tf_weights = tf.constant( weights )
        if not tf.is_tensor( y_pred ):
            y_pred = tf.constant( y_pred )

        y_true = tf.cast( y_true, y_pred.dtype )
        return tf.keras.losses.categorical_crossentropy( y_true, y_pred ) * tf.experimental.numpy.sum( y_true * tf_weights, axis = -1 )
    return wcce

...
config['loss'] = weighted_categorical_crossentropy( config['classWeight'] )
model.compile(
    loss = config['loss'],
    optimizer = config['optimizer'],
    metrics = ['accuracy'],
    run_eagerly = True
)