from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow.keras.backend as K
import os
import datetime
from tensorflow.python.tools import freeze_graph
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Activation, BatchNormalization
from tensorflow.keras.models import Model

inputs = Input(shape=(128, 128, 1))
x = Conv2D(4, (3, 3))(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)
x = Dense(5, activation='softmax')(x)
model = Model(inputs, x, name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

K.set_learning_phase(0)
save_dir = "./tmp_{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())
tf.saved_model.simple_save(K.get_session(),
                           save_dir,
                           inputs={"input": model.inputs[0]},
                           outputs={"output": model.outputs[0]})

freeze_graph.freeze_graph(None,
                          None,
                          None,
                          None,
                          model.outputs[0].op.name,
                          None,
                          None,
                          os.path.join(save_dir, "frozen_model.pb"),
                          False,
                          "",
                          input_saved_model_dir=save_dir)

def create_model():
    inputs = Input(...)
    ...
    return model

model = create_model()

K.clear_session()
K.set_learning_phase(0)

model = create_model()
model.load_weights("weights.h5")

import tensorflow as tf
import tensorflow.keras.backend as K
import os
import datetime
from tensorflow.python.tools import freeze_graph
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Activation, BatchNormalization, Lambda
from tensorflow.keras.models import Model


def create_model():
    inputs = Input(shape=(128, 128, 1))
    x = Conv2D(4, (3, 3))(inputs)
    x = BatchNormalization()(x)
    # x = Lambda((lambda x: tf.layers.batch_normalization(x)))(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(5, activation='softmax')(x)
    model = Model(inputs, x, name='test')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = create_model()

# Training goes here...

model.save_weights("weights.h5")

K.clear_session()
K.set_learning_phase(0)

model = create_model()
model.load_weights("weights.h5")

save_dir = "./tmp_{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())
tf.saved_model.simple_save(K.get_session(),
                           save_dir,
                           inputs={"input": model.inputs[0]},
                           outputs={"output": model.outputs[0]})

freeze_graph.freeze_graph(None,
                          None,
                          None,
                          None,
                          model.outputs[0].op.name,
                          None,
                          None,
                          os.path.join(save_dir, "frozen_model.pb"),
                          False,
                          "",
                          input_saved_model_dir=save_dir)

import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.keras.backend as K

source = "./freeze_test/frozen_model.pb"

session = K.get_session()
with gfile.Open(source, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    session.graph.as_default()
    tf.import_graph_def(graph_def, name='')

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, output_idx, **kwargs):
        self.output_idx = output_idx
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, x):
        
        y1 = p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x + p[4]*x*x*x*x + p[5]*x*x*x*x*x + p[6]*x*x*x*x*x*x + p[7]*x*x*x*x*x*x*x
        y2 = p[8] + p[9]*x + p[10]*x*x + p[11]*x*x*x
        y = tf.where(x>-5, y2, y1)
        
        return y


    def get_config(self):    
        config = super().get_config().copy()
        config.update({
            'output_idx' : self.output_idx
        })
        return config