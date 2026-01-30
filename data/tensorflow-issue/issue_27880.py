import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.contrib.quantize.create_training_graph(input_graph=tf.keras.backend.get_session().graph, quant_delay=0)
# create a new session after rewriting the graph
new_session = tf.Session()
tf.keras.backend.set_session(new_session)

def build_keras_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

def build_keras_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

tf.keras.backend.clear_session()
g = tf.keras.backend.get_session().graph
with tf.Session(graph=g) as session:

    model_clean = build_keras_model()
    tf.contrib.quantize.create_eval_graph(input_graph=g)
    optimizer_not_used, loss = ...

    # initialize automatically quantized variables
    session.run(tf.global_variables_initializer())
    # compile the model
    model_clean.compile(optimizer=optimizer_not_used,
                        loss=loss,
                        metrics=['accuracy'])
    # recover the model
    saver = tf.train.Saver()
    saver.restore(session, model_path)

def build_keras_model():
    return keras.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu", padding="same", use_bias=False, input_shape=(28, 28, 1)),
        keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False),
        tf.keras.layers.AveragePooling2D(pool_size=7),
        tf.keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])

train_images = np.reshape(train_images, [-1, 28, 28, 1])
test_images = np.reshape(test_images, [-1, 28, 28, 1])

def build_keras_model():
  
    return keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(28,28,1)),
            keras.layers.Activation("relu"),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.Conv2D(32, (3, 3), padding="same"),
            keras.layers.Activation("relu"),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.Activation("relu"),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.Activation("relu"),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(512),
            keras.layers.Activation("relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(classes),
            keras.layers.Activation("softmax")
    ])