import pathlib
from concurrent.futures import ProcessPoolExecutor

import gensim
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tensorflow import keras as tfk
from tqdm import tqdm

from src.preprocessing.sequence import create_dataset
from src.preprocessing.text import JiebaTokenizer


class TextCNN(tfk.Model):

    LABEL_IDX_DICT = {"constellation": 0, "education": 1, "entertainment": 2, "fashion": 3, "finance": 4, "game": 5,
                      "house": 6, "land": 7, "lottery": 8, "political": 9, "social": 10, "sports": 11, "stock": 12,
                      "technology": 13}
    IDX_LABEL_DICT = {idx: label for label, idx in LABEL_IDX_DICT.items()}

    def __init__(self, embeddings):
        super(TextCNN, self).__init__()
        self.embedding_layer = tfk.layers.Embedding(embeddings.shape[0], embeddings.shape[1], trainable=False,
                                                    embeddings_initializer=tf.initializers.Constant(embeddings))
        self.conv_layers = [tfk.layers.Conv1D(FLAGS.filters, FLAGS.kernel_size + idx * FLAGS.kernel_distance,
                                              strides=FLAGS.conv_strides, padding=FLAGS.conv_padding)
                            for idx in range(FLAGS.conv_pool_num)]
        self.pool_layers = [tfk.layers.MaxPool1D(pool_size=FLAGS.pool_size, strides=FLAGS.pool_strides,
                                                 padding=FLAGS.pool_padding)
                            for idx in range(FLAGS.conv_pool_num)]
        self.dense1_layer = tfk.layers.Dense(FLAGS.dense_units, activation=FLAGS.activation)
        self.batchnorm_layer = tfk.layers.BatchNormalization()
        self.dense2_layer = tfk.layers.Dense(len(self.LABEL_IDX_DICT), activation=tfk.activations.softmax)
        self.dropout_layer = tfk.layers.Dropout(FLAGS.dropout_rate)

    def call(self, inputs, training=None):
        x = inputs
        x_embed = self.embedding_layer(x)
        assert FLAGS.conv_pool_num > 0, ValueError("conv_pool_num must > 0")
        encodings = [self.conv_layers[idx](x_embed) for idx in range(FLAGS.conv_pool_num)]
        encodings_pool = [self.pool_layers[idx](encodings[idx]) for idx in range(FLAGS.conv_pool_num)]
        encodings_concat = tf.concat(encodings_pool, axis=-1)
        encodings_concat = self.batchnorm_layer(encodings_concat, training=training)
        encodings_flatten = tfk.layers.Flatten()(encodings_concat)
        encodings_flatten = self.dropout_layer(encodings_flatten, training=training)
        hidden = self.dense1_layer(encodings_flatten)
        y_pred = self.dense2_layer(hidden)
        return y_pred

w2v = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.embeddings_path)
words = list(w2v.vocab.keys())[:FLAGS.vocab_size]
tokenizer = JiebaTokenizer(words)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
embeddings = np.zeros(shape=(FLAGS.vocab_size + 2, w2v.vector_size), dtype=np.float32)
for word, idx in tokenizer.word2idx_dict.items():
    if word in w2v.vocab:
        embeddings[idx] = w2v.word_vec(word)
x, y = TextCNN.preprocess_dataset(tokenizer, FLAGS.trainset_path)
trainset = create_dataset([x, y])
trainset = trainset.batch(FLAGS.batch_size).shuffle(FLAGS.buffer_size)
val_x, val_y = TextCNN.preprocess_dataset(tokenizer, FLAGS.valset_path)
valset = create_dataset([val_x, val_y])
valset = valset.batch(FLAGS.batch_size).shuffle(FLAGS.buffer_size)
model = TextCNN(embeddings)
optimizer = tf.optimizers.Adam(FLAGS.lr)
model.compile(optimizer=optimizer, loss=tf.losses.sparse_categorical_crossentropy, metrics=["acc"])
model.fit(x=x, y=y, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, validation_data=(val_x, val_y))

w2v = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.embeddings_path)
words = list(w2v.vocab.keys())[:FLAGS.vocab_size]
tokenizer = JiebaTokenizer(words)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
embeddings = np.zeros(shape=(FLAGS.vocab_size + 2, w2v.vector_size), dtype=np.float32)
for word, idx in tokenizer.word2idx_dict.items():
    if word in w2v.vocab:
        embeddings[idx] = w2v.word_vec(word)
x, y = TextCNN.preprocess_dataset(tokenizer, FLAGS.trainset_path)
trainset = create_dataset([x, y])
trainset = trainset.batch(FLAGS.batch_size).shuffle(FLAGS.buffer_size)
val_x, val_y = TextCNN.preprocess_dataset(tokenizer, FLAGS.valset_path)
valset = create_dataset([val_x, val_y])
valset = valset.batch(FLAGS.batch_size).shuffle(FLAGS.buffer_size)
model = TextCNN(embeddings)
optimizer = tf.optimizers.Adam(FLAGS.lr)
loss_object = tf.losses.SparseCategoricalCrossentropy()
train_acc = tf.metrics.SparseCategoricalAccuracy()
val_acc = tf.metrics.SparseCategoricalAccuracy()
train_loss = tf.metrics.Mean()
val_loss = tf.metrics.Mean()

@tf.function
def train_op(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_object(y, y_pred)
        train_acc.update_state(y, y_pred)
        train_loss.update_state(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

@tf.function
def val_op(x, y):
    y_pred = model(x, training=False)
    loss = loss_object(y, y_pred)
    val_acc.update_state(y, y_pred)
    val_loss.update_state(loss)

for epoch in range(FLAGS.epochs):
    tf.print("Epoch {}/{}".format(epoch + 1, FLAGS.epochs))
    bar = tfk.utils.Progbar(target=len(x), unit_name="sample")
    train_acc.reset_states()
    val_acc.reset_states()
    train_loss.reset_states()
    val_loss.reset_states()
    for batch_x, batch_y in trainset:
        train_op(batch_x, batch_y)
        bar.add(len(batch_y))
    for batch_x, batch_y in valset:
        val_op(batch_x, batch_y)
    template = "loss: {:.4f}\nacc: {:.4f}\nval_loss: {:.4f}\nval_acc: {:.4f}"
    message = template.format(train_loss.result().numpy(), train_acc.result().numpy(),
                              val_loss.result().numpy(), val_acc.result().numpy())
    tf.print(message)

class TextCNN(tfk.Model):
    LABEL_IDX_DICT = {"constellation": 0, "education": 1, "entertainment": 2, "fashion": 3, "finance": 4, "game": 5,
                      "house": 6, "land": 7, "lottery": 8, "political": 9, "social": 10, "sports": 11, "stock": 12,
                      "technology": 13}
    IDX_LABEL_DICT = {idx: label for label, idx in LABEL_IDX_DICT.items()}

    def __init__(self, embeddings):
        super(TextCNN, self).__init__()
        self.embedding_layer = tfk.layers.Embedding(embeddings.shape[0], embeddings.shape[1], trainable=False,
                                                    embeddings_initializer=tf.initializers.Constant(embeddings))
        self.conv_layers = [tfk.layers.Conv1D(FLAGS.filters, FLAGS.kernel_size + idx * FLAGS.kernel_distance,
                                              strides=FLAGS.conv_strides, padding=FLAGS.conv_padding)
                            for idx in range(FLAGS.conv_pool_num)]
        self.pool_layers = [tfk.layers.MaxPool1D(pool_size=FLAGS.pool_size, strides=FLAGS.pool_strides,
                                                 padding=FLAGS.pool_padding)
                            for idx in range(FLAGS.conv_pool_num)]
        self.dense1_layer = tfk.layers.Dense(FLAGS.dense_units, activation=FLAGS.activation)
        self.dense2_layer = tfk.layers.Dense(len(self.LABEL_IDX_DICT), activation=tfk.activations.softmax)


    def call(self, inputs, training=None):
        x = inputs
        x_embed = self.embedding_layer(x)
        assert FLAGS.conv_pool_num > 0, ValueError("conv_pool_num must > 0")
        encodings = [self.conv_layers[idx](x_embed) for idx in range(FLAGS.conv_pool_num)]
        encodings_pool = [self.pool_layers[idx](encodings[idx]) for idx in range(FLAGS.conv_pool_num)]
        encodings_concat = tf.concat(encodings_pool, axis=-1)
        encodings_flatten = tfk.layers.Flatten()(encodings_concat)
        hidden = self.dense1_layer(encodings_flatten)
        y_pred = self.dense2_layer(hidden)
        return y_pred

        @tf.function
        def train_op(x, y):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                tf.print(y_pred)
                loss = loss_object(y, y_pred)
                train_loss.update_state(loss)
                train_acc.update_state(y, y_pred)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

        @tf.function
        def val_op(x, y):
            y_pred = model(x, training=False)
            tf.print(y_pred)
            loss = loss_object(y, y_pred)
            val_loss.update_state(loss)
            val_acc.update_state(y, y_pred)

@tf.function
def train_op(x, y):
    y_pred = model(x, training=True)
    tf.print(y_pred) # In my case the shapes of the output required an extra tf.squeeze
    loss = loss_object(y, y_pred)
    train_loss.update_state(loss)
    train_acc.update_state(y, y_pred)
    grads = tf.gradients(loss, model.trainable_variables)
    optimizer.apply_gradients(grads, model.trainable_variables)