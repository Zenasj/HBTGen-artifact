from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
DISABLE_EAGER = 1
resnet_depth = 96

if DISABLE_EAGER:
    disable_eager_execution()
if True:
    from tensorflow.keras.optimizers import *
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import *


def init():
    # game params
    board_x, board_y = 3, 3
    action_size = 10
    depth_dim = 2

    input_boards = Input(
        shape=(board_x, board_y, depth_dim))
    num_chan = 4
    h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
        Conv2D(num_chan, 1, padding='same', use_bias=False)(input_boards)))
    for i in range(resnet_depth):
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(num_chan, 1, padding='same', use_bias=False)(h_conv1)))
    hf = Flatten()(h_conv1)
    s_fc1 = Dropout(0.3)(Activation('relu')(BatchNormalization(axis=1)(
        Dense(16, use_bias=False)(hf))))
    pi = Dense(action_size, activation='softmax', name='pi')(
        s_fc1)

    model = Model(inputs=input_boards, outputs=pi)
    model.compile(
        loss=['categorical_crossentropy'], optimizer=Adam(0.001))
    return model


m = init()
print('inited model')
m.fit(x=np.zeros((1, 3, 3, 2)), y=np.zeros((1, 10)))