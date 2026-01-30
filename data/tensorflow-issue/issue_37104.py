import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(units = 12, activation='relu', use_bias = True, kernel_initializer= 'glorot_normal', bias_initializer = 'zeros', name = 'd1'),
    tf.keras.layers.Dense(units = 6, activation='relu', use_bias = True, kernel_initializer= 'glorot_normal', bias_initializer = 'zeros', name = 'd2'),
    tf.keras.layers.Dense(units = 2, activation='softmax', name = 'out')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.Precision()])
    return model

def create_model():
    model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(units = 12, activation='relu', use_bias = True, kernel_initializer= 'glorot_normal', bias_initializer = 'zeros', name = 'd1'),
    tf.keras.layers.Dense(units = 6, activation='relu', use_bias = True, kernel_initializer= 'glorot_normal', bias_initializer = 'zeros', name = 'd2'),
    tf.keras.layers.Dense(units = 1, activation='sigmoid', name = 'out')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy',tf.keras.metrics.Precision()])
    return model