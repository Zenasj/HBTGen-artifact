import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import keras
model = keras.Sequential()
model.add(keras.layers.Dense(units=1,
                                activation='sigmoid',
                                input_shape=(10, )))
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.save('./model.h5')

# Convert keras model to TF estimator
tf_files_path = './tf'
estimator =\
    tf.keras.estimator.model_to_estimator(keras_model=model,
                                          model_dir=tf_files_path)
def serving_input_receiver_fn():
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        {model.input_names[0]: tf.placeholder(tf.float32, shape=[None, 10])})

# Export the estimator
export_path = './export'
estimator.export_savedmodel(
    export_path,
    serving_input_receiver_fn=serving_input_receiver_fn())

model = tf.keras.models.load_model('path_to_model.h5')

# In 1.x
tf.keras.experimental.export_saved_model(model, 'path/to/saved/model')

# After the RFC is accepted/implemented in 2.x
model.save('path/to/saved/model')