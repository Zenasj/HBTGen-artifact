import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session()

training_model = lstm_model(seq_len=100, batch_size=128, stateful=False)

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    training_model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

tpu_model_reused = tf.contrib.tpu.keras_to_tpu_model(
    tf.keras.layers.Dense(1000)(training_model(training_model.inputs)),
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

tpu_model.fit_generator(
    training_generator(seq_len=100, batch_size=1024),
    steps_per_epoch=100,
    epochs=10,
)
tpu_model.save_weights('/tmp/bard.h5', overwrite=True)