import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

def create_gru_model():
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.GRU(128))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(49,activation=tf.nn.softmax))
    model.compile(optimizer='adam',loss=customLoss,metrics = ['accuracy'])
    return model
model = create_gru_model()
model.fit(x_train, y_train, epochs = 3, callbacks = [cp_callback], validation_data = (x_valid,y_valid), verbose=0)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter=True
tflite_model = converter.convert()

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

def customLoss(yTrue,yPred):

    return tf.nn.softmax_cross_entropy_with_logits(labels=yTrue,logits=yPred)


def create_gru_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(1, 49), name='input'),
        tf.keras.layers.GRU(2),
        tf.keras.layers.Dense(49, activation=tf.nn.softmax, name='output')
    ])
    model.compile(optimizer = 'adam',
                  loss = customLoss,
                  metrics = ['accuracy'])
    return model
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

checkpoint_path = "training_gru/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model = create_gru_model()
model.fit(x_train, y_train, epochs = 1)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
open("gru3.tflite", "wb").write(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = custom_generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                      tf.lite.OpsSet.SELECT_TF_OPS]