# tf.random.uniform((B,)) where B = num_blocks * word_size * 2 (input is vector of shape (num_blocks*word_size*2,))
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Reshape, Permute, Conv1D, BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.regularizers import l2

class MyModel(tf.keras.Model):
    def __init__(self, num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=5,
                 reg_param=0.0001, final_activation='sigmoid'):
        super().__init__()
        self.num_blocks = num_blocks
        self.word_size = word_size

        # Layers corresponding to the original make_resnet_preprocess model
        self.reshape = Reshape((2 * num_blocks, word_size))
        self.permute = Permute((2, 1))

        self.conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))
        self.bn0 = BatchNormalization()
        self.act0 = Activation('relu')

        # Residual blocks
        self.res_blocks = []
        for _ in range(depth):
            conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))
            bn1 = BatchNormalization()
            act1 = Activation('relu')
            conv2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))
            bn2 = BatchNormalization()
            act2 = Activation('relu')
            self.res_blocks.append((conv1, bn1, act1, conv2, bn2, act2))

        self.add = Add()
        self.flatten = Flatten()
        self.dense1 = Dense(d1, kernel_regularizer=l2(reg_param))
        self.bn_dense1 = BatchNormalization()
        self.act_dense1 = Activation('relu')
        self.dense2 = Dense(d2, kernel_regularizer=l2(reg_param))
        self.bn_dense2 = BatchNormalization()
        self.out_activation = Activation('relu')
        # Note: original final Dense layer with "num_outputs" and final_activation is commented out / omitted

    def call(self, inputs, training=False):
        # inputs assumed shape (batch_size, num_blocks * word_size * 2)
        x = self.reshape(inputs)
        x = self.permute(x)
        x = self.conv0(x)
        x = self.bn0(x, training=training)
        x = self.act0(x)

        shortcut = x
        for (conv1, bn1, act1, conv2, bn2, act2) in self.res_blocks:
            y = conv1(shortcut)
            y = bn1(y, training=training)
            y = act1(y)
            y = conv2(y)
            y = bn2(y, training=training)
            y = act2(y)
            shortcut = self.add([shortcut, y])

        x = self.flatten(shortcut)
        x = self.dense1(x)
        x = self.bn_dense1(x, training=training)
        x = self.act_dense1(x)
        x = self.dense2(x)
        x = self.bn_dense2(x, training=training)
        out = self.out_activation(x)  # relu activated final vector

        return out

def my_model_function():
    # Using default parameters similar to original make_resnet_preprocess call depth=1 in example
    # Assuming num_blocks=2, word_size=16 by default here.
    return MyModel()

def GetInput():
    # Inputs are shape (batch_size, num_blocks * word_size * 2)
    # Using batch_size=1 for example, num_blocks=2, word_size=16 => input shape = (1, 64)
    num_blocks = 2
    word_size = 16
    input_shape = (num_blocks * word_size * 2,)
    # Random uniform input tensor matching expected input shape
    return tf.random.uniform((1,) + input_shape, dtype=tf.float32)

