import tensorflow as tf
from tensorflow import keras

strategy = tf.distribute.experimental.TPUStrategy(
    tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS))
with strategy.scope():
  Inp, output = build_model()
  opt = tf.train.AdamOptimizer(learning_rate)
  tpu_model = tf.keras.Model(inputs=[Inp], outputs=[output])
  tpu_model.compile(
      optimizer=opt,
      loss='categorical_crossentropy',
      metrics=['acc'])