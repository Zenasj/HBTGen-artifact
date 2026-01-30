from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.compat.v2.keras.layers import Bidirectional, GRU, Input, Activation
from tensorflow.compat.v2.keras.layers import TimeDistributed, Dense, Dropout
from tensorflow.compat.v2.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf

def create_model():

    layer_input   = Input(shape=(20, 256)) 
    layer_bi_rnn  = Bidirectional(GRU(units=512, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, recurrent_initializer='glorot_uniform'))(layer_input)
    layer_dropout = Dropout(0.2)(layer_bi_rnn)
    layer_dense   = TimeDistributed(Dense(50))(layer_dropout)
    layer_act     = Activation('softmax')(layer_dense)
    model         = Model([layer_input], layer_act)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001))

    return model

model = create_model()
tf.saved_model.save(model, 'testdata/')