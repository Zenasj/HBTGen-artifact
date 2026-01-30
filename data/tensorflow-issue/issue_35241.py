import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import numpy as np
import tensorflow as tf

def positional_encoding(shape, dtype=tf.float32):
    """
    positional encoding initializer. Note that we can't freeze the embedding layer because tf is shit
    :param shape:
    :param dtype:
    :return:
    """
    n_pos, dim = shape
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])

    # apply sin to even indices in the array; 2i
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

    # pos_encoding = position_enc[np.newaxis, ...]

    return tf.cast(position_enc, dtype=dtype)


class CustomEmbedding(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, vocab_size, hidden_size,  max_position_embeddings, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_idx = 0
        self.max_position_embeddings += 1


    def build(self, input_shape):
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                shape=(self.max_position_embeddings, self.hidden_size),
                initializer=positional_encoding,
                trainable=False,
                dtype=self.dtype)

        with tf.name_scope("word_embeddings"):
            self.word_embeddings = self.add_weight(
                "token_weight",
                shape=(self.vocab_size, self.hidden_size),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
                dtype=self.dtype
            )
        super(CustomEmbedding, self).build(input_shape)

    def call(self, inputs):
        input_ids, position_ids = inputs

        inputs_embeds = tf.nn.embedding_lookup(self.word_embeddings, input_ids)
        position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids)

        embeddings = inputs_embeds + position_embeddings
        return embeddings


class TestModel(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size,  max_position_embeddings, num_class, **kwargs):
        super(TestModel, self).__init__(**kwargs)
        self.emb = CustomEmbedding(vocab_size, hidden_size, max_position_embeddings)
        self.dense = tf.keras.layers.Dense(num_class, name="class_prj")

    def call(self, inputs):
        word_emb = self.emb(inputs)
        sent_emb = tf.reduce_mean(word_emb, axis=1)
        logit = self.dense(sent_emb)
        return logit

VOCAB_SIZE = 100
HIDDEN_SIZE = 5
MAX_POSITION_EMBEDDING = 30
NUM_CLASS = 3
EPOCH = 1
LR = 0.001


def compute_loss(label, logit):
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)
    return tf.reduce_mean(cross_ent)


if __name__ == '__main__':
    tf.random.set_seed(0)
    # construct the model
    model = TestModel(VOCAB_SIZE, HIDDEN_SIZE, MAX_POSITION_EMBEDDING, NUM_CLASS)

    # generate dataset
    seqs = np.array([[0, 0, 1], [1, 1, 1], [2, 3, 2], [3, 4, 3], [4, 4, 4]], dtype=np.int64)
    poss = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.int64)
    label = np.array([0, 1, 2, 1, 1], dtype=np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((seqs, poss, label)).batch(3)

    # def optim
    optim = tf.keras.optimizers.RMSprop(LR, clipnorm=1.)

    # def checkpoint
    ckpt = tf.train.Checkpoint(model=model, optimizer=optim)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              directory="./tmp/test_ckp",
                                              max_to_keep=2)


    def train_step(inputs):
        seq, pos, label = inputs
        with tf.GradientTape() as tape:
            logit = model((seq, pos))
            loss = compute_loss(label, logit)

        gradients = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


    for epoch in range(EPOCH):
        loss = 0.
        for batch in dataset:
            b_loss = train_step(batch)
            loss += b_loss.numpy()

        print(loss)
        ckpt_save_path = ckpt_manager.save()