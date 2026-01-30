from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

def build_and_compile_model():
    
    input = tf.keras.Input((20,))
    x = tf.keras.layers.BatchNormalization()(input)
    y = tf.keras.layers.Dense(2)(x)
    
    model = tf.keras.Model(inputs=input, outputs=y)
    
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    
    return model

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_and_compile_model()
model.save('test', save_format='tf')