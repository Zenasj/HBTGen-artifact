from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf

from tensorflow.keras.layers import Input, Activation, Embedding, LSTM, Dense, Dropout, Flatten, Concatenate, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

#tf.config.set_visible_devices([], 'GPU')

strategy = tf.distribute.MirroredStrategy()

words = tf.constant(((1,1,1,1,1),(1,1,1,1,1)))
products = tf.ones((10000,101), dtype=tf.int32)

test_dataset = tf.data.Dataset.from_tensor_slices(products)
test_dataset = test_dataset.batch(128, drop_remainder=True)
test_dataset = strategy.experimental_distribute_dataset(test_dataset)

def create_model(words_count):
    input_words = Input(shape=tf.TensorShape(5), dtype='int32', name='input_words')

    x = Embedding(output_dim=32, input_dim=words_count, input_length=5, mask_zero=True)(input_words)
    #x = Embedding(output_dim=32, input_dim=words_count, input_length=5, mask_zero=False)(input_words)

    x = LSTM(32)(x)
    #x = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(32))(x)

    model = Model(inputs=input_words, outputs=x)
    return model

with strategy.scope():
    model = create_model(11111)
    optimizer = tf.keras.optimizers.SGD()

print(model.summary())

@tf.function
def test_step(b_cmp):
    cmp_words = tf.gather(words, b_cmp)
    tmp = tf.reshape(cmp_words, (-1,5))
    tmp = model(tmp, training=False)
   
    # ...

    r = tf.reduce_sum(tmp)
    return r

for b_cmp in test_dataset:
    strategy.run(test_step, args=(b_cmp,))

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)