import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

vocab_size=2
label_size = 2
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.word_embedding = tf.keras.layers.Embedding(vocab_size, 100, trainable=True)
        self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True))
        self.classifier = tf.keras.layers.Dense(label_size)
        
    def call(self, word_ids):
        x = self.word_embedding(word_ids)
        x = self.encoder(x)
        x = self.classifier(x)
        return x


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    real = tf.reshape(real, (-1,))
    pred = tf.reshape(pred, (-1, pred.shape[-1]))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def accuracy(y_true, y_pred):
    y_pred = tf.argmax(tf.reshape(y_pred, (-1, y_pred.shape[-1])), axis=-1)
    y_true = tf.reshape(y_true, (-1,))
    y_pred = y_pred[y_true > 0]
    y_true = y_true[y_true > 0]
    return tf.metrics.categorical_accuracy(y_true[y_true > 0], y_pred[y_true > 0])


model = Model()
model.compile(optimizer=tf.optimizers.Adam(), loss=loss_function, metrics=[accuracy])
model.fit(dataset)

def generator():
    for data in zip([[1, 1, 1], [1, 1, 1, 1]], [[2, 2, 2], [2, 2, 2, 2]]):
        yield data
dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.int64, tf.int64))
dataset = dataset.padded_batch(1, ((None,), (None,)))