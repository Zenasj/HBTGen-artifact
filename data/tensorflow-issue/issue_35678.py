import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

def build_model_tf(lr, window_size, features, classes):
    # input layer
    inp = tf.keras.layers.Input(shape=[window_size, features], batch_size=42)
    # batch norm layer to normalize data
    x = tf.keras.layers.BatchNormalization()(inp)

    # two convolutional layers
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(filters=32, kernel_size=6)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # 3 lstm layers
    x = tf.keras.layers.LSTM(32, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.L1L2(0.5, 0.1))(x)
    x = tf.keras.layers.LSTM(24, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.L1L2(0.5, 0.1))(x)
    x = tf.keras.layers.LSTM(16, return_sequences=False, recurrent_regularizer=tf.keras.regularizers.L1L2(0.5, 0.1))(x)

    # dense layer
    x = tf.keras.layers.Dense(8)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    #
    # output layer
    out = tf.keras.layers.Dense(classes, activation='softmax')(x)

    # create functional model object
    model = tf.keras.models.Model(inputs=inp, outputs=out)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr, decay=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# convert keras model to concrete functions
tf.saved_model.save(tf_keras_model, join(ROOT, 'trained_models', model_name + '_conv'))
model = tf.saved_model.load(join(ROOT, 'trained_models', model_name + '_conv'))
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# initialize the tf_light converter
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# add experimental flag, because rnn structures are not covered by the default converter
converter.experimental_new_converter = True
# convert model to tf_light
tf_lite_model = converter.convert()

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]