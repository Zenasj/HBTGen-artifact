from tensorflow.keras import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import glorot_normal, he_normal, lecun_normal

dataset, info = tfds.load('binary_alpha_digits', with_info=True, split='train')
data = dataset.map(lambda x: (tf.cast(x['image'], tf.float32), x['label'])).batch(8)


class Model(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = Dense(16, kernel_initializer=he_normal)
        self.layer2 = Dense(units=info.features['label'].num_classes)

    def call(self, inputs, training=None, **kwargs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x


model = Model()
_ = model(next(iter(data))[0])