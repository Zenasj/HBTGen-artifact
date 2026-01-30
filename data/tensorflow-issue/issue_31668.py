from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow.keras.backend as K
import os
import datetime
import shutil
from tensorflow.python.tools import freeze_graph
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Activation, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.platform import gfile

# Clear the session
K.clear_session()

# TF version
print("TF version is " + tf.__version__)

# Create model
print('# Creating model')
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

# Remove old frozen graph directory
if os.path.exists('/content/frozen/'):
  shutil.rmtree('/content/frozen/')

# Create saved model
print('# Saving model')
save_dir = "/content/frozen/"
tf.saved_model.simple_save(K.get_session(),
                           save_dir,
                           inputs={"input": model.inputs[0]},
                           outputs={"output": model.outputs[0]})

# Freeze graph from saved model checkpoint
print('# Freezing graph')
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

# Try to load frozen graph
print('# Loading graph')
source = "/content/frozen/frozen_model.pb"
session = K.get_session()
with gfile.Open(source, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    session.graph.as_default()
    tf.import_graph_def(graph_def, name='')