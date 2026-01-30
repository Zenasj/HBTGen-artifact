import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

3
TPU_ADDRESS = "grpc://" + "10.0.0.2:8470"

with tf.compat.v1.Session(TPU_ADDRESS) as session:
    print('TPU devices:')
    pprint.pprint(session.list_devices())

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
try:
    tf.config.experimental_connect_to_cluster(resolver)
except tf.errors.UnimplementedError as uie:
    print(uie, "This appears to be caused by the TPU already being connected. Ignoring.", sep='\n')
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)

3
with tpu_strategy.scope():
    in_id = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), name="input_ids", dtype=np.int32)
    in_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), name="input_masks", dtype=np.int32)
    in_segment = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), name="segment_ids", dtype=np.int32)
    bert_inputs = {"input_ids": in_id, "input_mask": in_mask, "segment_ids": in_segment}

    bert_layer = hub.KerasLayer(BERT_MODEL_HUB, signature="tokens", output_key="pooled_output")(bert_inputs)
    bert_layer.trainable = True

    dense = tf.keras.layers.Dense(256, activation='relu')(bert_layer)
    pred = tf.keras.layers.Dense(len(unique_labels), activation='sigmoid')(dense)

    model = tf.keras.Model(inputs=bert_inputs, outputs=pred)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=["accuracy"]
    )

3
bert_layer = hub.KerasLayer(BERT_MODEL_HUB, signature="tokens", output_key="pooled_output")(bert_inputs)
bert_layer.trainable = True

3
bert_layer = hub.KerasLayer(BERT_MODEL_HUB, signature="tokens", output_key="pooled_output", trainable=True)(bert_inputs)