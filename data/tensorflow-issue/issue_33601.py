from tensorflow.keras import layers
from tensorflow.keras import models

import argparse
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Embedding, Dense, Input
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM
from tensorflow.keras.utils import to_categorical
import numpy as np

parser = argparse.ArgumentParser(description='train recurrent net.')
parser.add_argument('--epochs', dest='epochs',  type=int, default=2)
parser.add_argument('--hidden_size', dest='hidden_size',  type=int, default=50)
parser.add_argument('--RNN_type', dest='RNN_type',  type=str, default='GRU')
parser.add_argument('--epoch_size', dest='epoch_size',  type=int, default=1000)
parser.add_argument('--pretrained_name', dest='pretrained_name',  type=str, default=None)

args = parser.parse_args()

RNN_type = {}

RNN_type['LSTM'] = LSTM
RNN_type['GRU'] = GRU
RNN_type['SimpleRNN'] = SimpleRNN

LSTM_use = RNN_type[args.RNN_type]

max_output = 10
full_python_file_string = [1,3,2,4,5,3,2,3,4,5]
class KerasBatchGenerator(object):
    def __init__(self, data_set):
        self.data_set = data_set
            
    def generate(self):
        while True:
            tmp_x = np.array([full_python_file_string], dtype=int)
            tmp_y = np.array([full_python_file_string], dtype=int)
            yield tmp_x, to_categorical(tmp_y, num_classes=max_output)

train_data_generator = KerasBatchGenerator(0)
test_data_generator = KerasBatchGenerator(0)

hidden_size = args.hidden_size

if args.pretrained_name is not None:
  from tensorflow.keras.models import load_model
  model = load_model(args.pretrained_name)
else:
  inputs = Input(batch_shape=(1,None,))
  embeds = Embedding(max_output, max_output, embeddings_initializer='identity', trainable=True)(inputs)
  lstm1 = LSTM_use(hidden_size, return_sequences=True, stateful = True)(embeds)
  x = Dense(max_output)(lstm1)
  predictions = Activation('softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)

checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)

model.compile(loss='categorical_crossentropy', optimizer = 'SGD', metrics=['categorical_accuracy'])
model.fit_generator(train_data_generator.generate(), args.epoch_size, args.epochs, 
                    validation_data=test_data_generator.generate(), 
                    validation_steps=args.epoch_size / 10, callbacks=[checkpointer])