import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def compile_keras_model(dtype):
  policy = tf.keras.mixed_precision.experimental.Policy(dtype)
  tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)

  optimizer = tf.optimizers.SGD(learning_rate=0.1, momentum=0.9)

  model = tf.keras.applications.resnet50.ResNet50(weights=None)

  # Create new model that is `model` except the output is float32
  inp = tf.keras.layers.Input(batch_shape=model.input_shape)
  out = model(inp)
  # This layer simply casts to float32
  out = tf.keras.layers.Activation('linear', dtype='float32')(out)
  model = tf.keras.Model(inp, out)

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=['sparse_categorical_accuracy'])
  return model

gpu_model = compile_keras_model('mixed_float16')
tpu_model = compile_keras_model('mixed_bfloat16')

out = tf.keras.layers.Dense(32, activation='softmax')(out)

out = tf.keras.layers.Dense(32)(out)
out = tf.keras.layers.Activation('softmax', dtype='float32')(out)