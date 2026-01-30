from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

if __name__ == '__main__':

    sess = tf.keras.backend.get_session()

    model = Sequential()

    model.add(Dense(units=64, activation='relu', input_dim=4))
    model.add(Dense(units=4, activation='linear'))

    opt = SGD(lr=0.01)
    model.compile(loss='mse', optimizer=opt)

    graph = tf.get_default_graph()

    model.predict(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))

    for _ in range(100):
        with graph.as_default():
            time.sleep(1)
            model.fit(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
            temp_export_path = '/tmp/models/' + str(time.time()).split(".")[0]

            # Saving
            builder = tf.saved_model.builder.SavedModelBuilder(temp_export_path)
            signature = predict_signature_def(inputs={'state': model.input},
                                              outputs={t.name: t for t in model.outputs})
            builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],
                                                 signature_def_map={
                                                     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
            builder.save()