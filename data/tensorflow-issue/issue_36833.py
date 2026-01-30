from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
tf.debugging.enable_check_numerics()

def build_and_compile_model():
    
    input = tf.keras.Input((20,))
    y = tf.keras.layers.Dense(2)
    model = tf.keras.Model(inputs=input, outputs=y)
    
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    
    return model

model = build_and_compile_model()
model.save('test', save_format='tf')

y = tf.keras.layers.Dense(2)(input)