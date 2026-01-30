from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import gc


def build_and_save_own_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.save('my_model')
    tf.keras.backend.clear_session()
    del model
    gc.collect()


def profile_load_model(path):
    model = tf.saved_model.load(path)
    tf.keras.backend.clear_session()
    del model
    gc.collect()


def run_model():
    model_path = 'my_model'
    build_and_save_own_model()
    print("load model in loops:")
    c = 1
    while True:
        print("----------- iter", c)
        profile_load_model(model_path)
        c += 1


if __name__ == '__main__':
    print("*****************************************************")
    print("START LOADING MODEL")
    print(tf.version.GIT_VERSION, tf.version.VERSION)
    print("*****************************************************")
    run_model()

model_copy =  tf.keras.models.clone_model(model)
model_copy.build()
model_copy.set_weights(model.get_weights())