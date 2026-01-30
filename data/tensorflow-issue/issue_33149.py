import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class SegmentedMean(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(SegmentedMean, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features, segments = inputs
        return tf.math.segment_mean(features, segments)


df = pd.DataFrame({'list': [
    [[1, 2, 3, 4], [2, 3, 6]],
    [[1, 2], [1, 5, 8]],
    [[6, 7, 8], [2, 4, 10], [1, 6]],
    [[3, 4, 6, 8], [1, 8], [2]],
    [[3, 6, 8]],
],
    'label': [0, 0, 1, 1, 1]})
df['list_len'] = df['list'].apply(len)
batch_size = 8
# list of all unique numbers used
nums_used = list(set([i for j in df['list'].apply(lambda k: [i for j in k for i in j]) for i in j]))
list_enc = MultiLabelBinarizer().fit([[i] for i in nums_used])


def create_batch():
    batch_ub = df.sample(batch_size // 2)  # upper bound, will trim so the batch size is correct after expansion
    batch_end = (batch_ub['list_len'].cumsum() <= batch_size)[::-1].idxmax()
    batch = batch_ub.loc[:batch_end].copy()
    batch['id'] = np.arange(0, len(batch))
    feat_bags = batch['list'].apply(list_enc.transform)
    feats = np.concatenate(feat_bags.values)
    # so I know which data to group together during segmentation
    segments = batch['id'].repeat(batch['list_len']).values
    labels = batch['label'].values.astype(np.int32)
    return (feats, segments), labels


settings = {'k': 40, 'iterations': 10}
feats_len = len(nums_used)
inputs = tf.keras.Input(shape=(feats_len,), name='features')
segments = tf.keras.Input(shape=(), name='segments', dtype=tf.int32)
x = tf.keras.layers.Dense(settings['k'], activation=tf.nn.relu)(inputs)
x = tf.keras.layers.Dense(settings['k'])(x)

x = SegmentedMean()((x, segments))
x = tf.keras.layers.Dense(settings['k'], activation=tf.nn.relu)(x)
logits = tf.keras.layers.Dense(2, name='output_logits')(x)
probs = tf.keras.layers.Softmax()(logits)
model = tf.keras.Model(inputs=(inputs, segments), outputs=(logits, probs), name='mil_model')
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
for step in range(settings['iterations']):
    x_train, y_train = create_batch()
    with tf.GradientTape() as tape:
        logits, probs = model(x_train)
        loss_value = loss(y_train, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))