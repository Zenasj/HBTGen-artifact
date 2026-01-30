import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DenseModel(Model):
    def __init__(self):
        super(DenseModel, self).__init__(name='class_dense_model')
        self.dense_1 = layers.Dense(1)
        self.dense_64 = layers.Dense(64, activation=tf.nn.relu)
        self.dense_100 = layers.Dense(100,activation=tf.nn.relu)
        self.dense_200 = layers.Dense(200,activation=tf.nn.relu)
        self.dense_200_2 = layers.Dense(200, activation=tf.nn.relu)

    def call(self, input_tensor, training=False, **kwargs):
        out_1 = self.dense_64(input_tensor)
        out_2 = self.dense_100(out_1)
        out_3 = self.dense_200(out_2)
        out_4 = self.dense_200(out_3)
        return self.dense_1(out_4)

    def build_graph(self, shape):
        x = tf.keras.layers.Input(shape=shape)
        return Model(inputs=[x], outputs=self.call(x))

model = DenseModel()
model.build_graph(INPUT_SHAPE)

out_4 = self.dense_200(out_3)

out_4 = self.dense_200_2(out_3)