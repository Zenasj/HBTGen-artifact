import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import sys
import tensorflow as tf
import numpy as np

def build_model_():

	input_a_size = 20
	input_b_size = 4
	num_classes = 2
	len_embedding = 256

	input_a = tf.keras.layers.Input(shape=(input_a_size,), name='input_a', dtype=np.uint8)
	input_b = tf.keras.layers.Input(shape=(input_b_size,), name='input_b', dtype=np.float32)

	x = tf.keras.layers.Embedding(len_embedding, 100)(input_a)
	x = tf.keras.layers.Conv1D(128, 4, activation='relu')(x)
	x = tf.keras.layers.MaxPooling1D(4)(x)
	x = tf.keras.layers.Flatten()(x)
	branch_a = tf.keras.layers.Dense(64, activation='relu')(x)

	x = tf.keras.layers.Dense(32, activation='relu')(input_b)
	branch_b = tf.keras.layers.Dense(32, activation='relu')(x)

	concat = tf.keras.layers.Concatenate()([
				                            branch_a,
				                            branch_b,
				                           ])

	x = tf.keras.layers.Dense(512, activation = 'relu')(concat)
	output = tf.keras.layers.Dense(num_classes, name='output', activation='softmax')(x)

	model = tf.keras.models.Model(inputs=[
				                          input_a,
				                          input_b,
				                         ],
				                  outputs=[output])

	return model

strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
with strategy.scope():
    model = build_model_()
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

y_train = True
y_train = tf.keras.utils.to_categorical(y_train, 2)

dataset = tf.data.Dataset.from_tensors(
    (
        {"input_a": [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]], 
         "input_b": [[1.], [1.], [1.], [1.]],}, 
        {"output": y_train},
    )
).repeat(1000000).batch(256)

history = model.fit(
    x = dataset,
    epochs=10,
    verbose = 1,
)

import os
import sys

import numpy as np
import tensorflow as tf


def build_model():
    input_a = tf.keras.Input(shape=(256,), dtype=tf.int64, name="input_a")
    input_b = tf.keras.Input(shape=(256,), dtype=tf.int64, name="input_b")

    emb_a = tf.keras.layers.Embedding(
        input_dim=20000,
        output_dim=128,
        input_length=256,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        trainable=True)
    emb_b = tf.keras.layers.Embedding(
        input_dim=20000,
        output_dim=128,
        input_length=256,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        trainable=True)

    x = tf.keras.layers.add([emb_a(input_a), emb_b(input_b)])
    x = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=3,
                               activation="relu")(x)
    x = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=3,
                               activation="relu")(x)
    x = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=3,
                               activation="relu")(x)
    x = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=3,
                               activation="relu")(x)
    x = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=3,
                               activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(x)

    model = tf.keras.models.Model(
        inputs=[
            input_a,
            input_b,
        ],
        outputs=output
    )

    return model


def main():
    assert len(sys.argv) == 2
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    gpus = sys.argv[1].split(",")
    if len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = build_model()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

    batch_size = 1024 * strategy.num_replicas_in_sync
    dataset = tf.data.Dataset.from_tensors(
        (
            {"input_a": np.random.randint(0, 20001, (256,), int),
             "input_b": np.random.randint(0, 20001, (256,), int)},
            {"output": np.random.randint(0, 2, (1,), int)},
        )
    ).repeat(1000000).batch(batch_size)

    model.fit(
        x=dataset,
        epochs=10,
        verbose=1,
    )


if __name__ == "__main__":
    main()

py
dataset = tf.data.Dataset.from_tensors((
  {"input_a": np.random.randint(0, 20001, (256,), int),
   "input_b": np.random.randint(0, 20001, (256,), int)},
  {"output": np.random.randint(0, 2, (1,), int)},
)).repeat(1000000)
# determine the cardinality/size of the dataset, then distribute it
cardinality = tf.data.experimental.cardinality(dataset).numpy()
dataset = strategy.experimental_distribute_dataset(dataset.batch(batch_size))
# determine the number steps/epoch (necessary for distributed datasets)
epochs = 10
batches = cardinality // batch_size
steps_per_epoch = batches // epochs
# train the model...
history = model.fit(
  x=dataset,
  epochs=epochs,
  steps_per_epoch=steps_per_epoch,
  verbose=1)