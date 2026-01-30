import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DenseNormLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation, name=None):
        super(DenseNormLayer, self).__init__(name=name)
        self.dense_layer = tf.keras.layers.Dense(units=units, activation=activation)
        self.batch_norm = tf.keras.layers.BatchNormalization()
    
    def call(self, x):
        x = self.batch_norm(x)
        x = self.dense_layer(x)
        return x

class BaselineModel(tf.keras.Model):
    def __init__(self, targets, name="BaselineModel"):
        super(BaselineModel, self).__init__(name=name)
        self.block1 = DenseNormLayer(units=1024, activation="relu", name="block1")
        self.block2 = DenseNormLayer(units=1024, activation="relu", name="block2")
        self.d1 = tf.keras.layers.Dense(units=512, activation="relu", name="d1")
        self.d2 = tf.keras.layers.Dense(units=1024, activation="relu", name="d2")
        self.dp = tf.keras.layers.Dropout(rate=0.2, name="dropout_layer")
        self.sigmoid_layer = tf.keras.layers.Dense(units=targets, activation="sigmoid", name="sigmoid_layer")
    
    def call(self, X):
        x = self.block1(X)
        x = self.block2(x)
        x = self.d1(x)
        x = self.dp(x)
        x = self.d2(x)
        x = self.sigmoid_layer(x)
        return x
    
    def build_graph(self, dim):
        x = tf.keras.Input(shape=dim)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

test_model = BaselineModel(targets=100)
test_model.build(input_shape=(None, 100))
test_model.summary()

import os
tf.keras.utils.plot_model(
    test_model.build_graph(dim=100), to_file=os.path.join(".", "model.png"),
    dpi=96, show_shapes=True, show_layer_names=True, expand_nested=False
)

test_model = BaselineModel(targets=100)
test_model.build(input_shape=(None, 100))
test_model.summary()