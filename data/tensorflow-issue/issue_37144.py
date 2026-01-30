import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GroupSoftmax(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(GroupSoftmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return tf.divide(inputs, tf.reduce_sum(inputs, axis=self.axis))

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(GroupSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

'''
-----------------network of g-----------------
'''
gModel = tf.keras.Sequential([
# 添加一个有Nodes个神经元的全连接层，“input_shape”为该层接受的输入数据的维度，“activation”指定该层所用的激活函数
layers.Dense(Nodes, activation='sigmoid', input_shape=(60,), use_bias = False),#封装数据应该为（3000，10，6）
# 添加第二个网络层
layers.Dense(Nodes, activation='sigmoid', use_bias = False),
# 添加第3个网络层
layers.Dense(Nodes, activation='sigmoid', use_bias = False),
# 添加第4个网络层
layers.Dense(Nodes, activation='sigmoid', use_bias = False),
# 添加第5个网络层
layers.Dense(Nodes, activation='sigmoid', use_bias = False),
# 添加第6个网络层,改变节点数目
layers.Dense(66, activation='sigmoid', use_bias = False),
# 添加第7个网络层,改变shape
layers.Reshape((11, 6)),
# 添加output网络层,分组softmax
#layers.Dense(6, activation=layers.Softmax(axis=0),input_shape=(11,6), use_bias = False), # [11,6]
#layers.Softmax(axis=0)
GroupSoftmax(axis=0)
])

gModel.summary()

ds_train     = ds_train.map(lambda img, label: (img, tuple([label])))

unpack_label = lambda img, label: (img, tuple([label]))
unpack_label = tf.autograph.do_not_convert(unpack_label)  # Runtime not compatible
ds_train = ds_train.map(unpack_label)

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

model = Sequential([
    Flatten(input_shape=X_train.shape[1:]),
    Dense(30, activation='relu'),
    Dense(30, activation='relu'),
    Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30)

y_pred = model.predict(X_test)  # <--- Error occurs here
print(np.argmax(predictions[0]))