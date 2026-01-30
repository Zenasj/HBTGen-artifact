import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
#tf.debugging.set_log_device_placement(True)

VOCAB_SIZE = 32000
BATCH_SIZE = 20
LEARNING_RATE = 0.001

class SimpleModel(tf.keras.Model):
    def __init__(self, target_vocab_size):
        super(SimpleModel, self).__init__()
        self.d_model = 1024
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
        self.body = tf.keras.layers.Dense(self.d_model)

    def call(self, input, training):
        # input shape: (batch, seq_len)
        x = self.embedding(input)  # (batch, seq_len, d_model)
        x = self.body(x)
        logits = tf.matmul(x, self.embedding.embeddings, transpose_b=True)
        return logits


def get_dataset():
    ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform((1000, 20), 0, VOCAB_SIZE, dtype=tf.int64))
    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=[None])
    ds = ds.repeat()
    ds = ds.prefetch(2)
    return ds


def train(ds, model, optimizer):

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def calculate_loss(real, pred):
        loss_ = loss_object(real, pred)
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(batch):
        tar_inp = batch[:, :-1]
        tar_real = batch[:, 1:]

        with tf.GradientTape() as tape:
            logits = model(tar_inp, True)
            loss = calculate_loss(tar_real, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return None

    #tf.summary.trace_on(graph=True, profiler=True)

    for i, batch in enumerate(ds):
        _ = train_step(batch)

        #if i == 10:
        #   tf.summary.trace_export(name="model_trace", step=0, profiler_outdir="/tmp/profiling/5")
        #   break


if __name__ == "__main__":
    ds = get_dataset()
    model = SimpleModel(VOCAB_SIZE)
    optimizer = tf.keras.optimizers.SGD(LEARNING_RATE)

    train(ds, model, optimizer)