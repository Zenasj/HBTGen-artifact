import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
TF2 = True
if TF2:
	### currently, there is a bug in tf.keras: model.reset_states() does not work
	from tensorflow.keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, Bidirectional
	from tensorflow.keras.models import Model
else:
	### in the old keras, bidi-RNNs with stateful=True behave smae as stateful=False
	from keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, Bidirectional
	from keras.models import Model

sequence_length = 3
feature_dim = 1
features_in = Input(batch_shape=(1, sequence_length, feature_dim)) 

rnn_out = Bidirectional( SimpleRNN(1, activation=None, use_bias=False, return_sequences=True, return_state=False, stateful=False))(features_in)
stateless_model = Model(inputs=[features_in], outputs=[rnn_out])

stateful_rnn_out = Bidirectional( SimpleRNN(1, activation=None, use_bias=False, return_sequences=True, return_state=False, stateful=True))(features_in)
stateful_model = Model(inputs=features_in, outputs=stateful_rnn_out)

toy_weights = [ np.asarray([[1.0]], dtype=np.float32), np.asarray([[-0.5]], dtype=np.float32), np.asarray([[1.0]], dtype=np.float32), np.asarray([[-0.5]], dtype=np.float32)]

stateless_model.set_weights(toy_weights)
stateful_model.set_weights(toy_weights)

x_in = np.zeros(sequence_length)
x_in[0] = 1
x_in = x_in.reshape( (1, sequence_length, feature_dim) )

def print_bidi_out(non_stateful_out, stateful_out):
	fb = ['FWD::', 'BWD::']

	for i in range(2):
		print(fb[i])
		print(f'non_stateful: {non_stateful_out.T[i]}')
		print(f'stateful: {stateful_out.T[i]}')
		print(f'delta: {stateful_out.T[i]-non_stateful_out.T[i]}')


non_stateful_out = stateless_model.predict(x_in).reshape((sequence_length,2))
stateful_out = stateful_model.predict(x_in).reshape((sequence_length,2))
print_bidi_out(non_stateful_out, stateful_out)

non_stateful_out = stateless_model.predict(x_in).reshape((sequence_length,2))
stateful_out = stateful_model.predict(x_in).reshape((sequence_length,2))
print_bidi_out(non_stateful_out, stateful_out)

print('\n** RESETING STATES in STATEFUL MODEL **\n')
stateful_model.reset_states()
non_stateful_out = stateless_model.predict(x_in).reshape((sequence_length,2))
stateful_out = stateful_model.predict(x_in).reshape((sequence_length,2))
print_bidi_out(non_stateful_out, stateful_out)