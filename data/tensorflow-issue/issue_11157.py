from tensorflow.keras import layers
from tensorflow.keras import models

model.fit_generator(self.train_inputs, steps_per_epoch=self.train_inputs.steps_per_epoch(),
                    validation_data=test_input_sequence, validation_steps=steps_test,
                    max_queue_size=self.train_inputs.workers, epochs=i+1, initial_epoch=i,
                    workers=self.train_inputs.workers, use_multiprocessing=True,
                    callbacks = callbacks)

history = CumulativeHistory()
callbacks = [history]
from keras import backend as K
if K.backend() == 'tensorflow':
  board = keras.callbacks.TensorBoard(log_dir=f"{self.prefix_folder_logs}{time()}",
                                    histogram_freq=1, write_graph=True, write_images=True)
  callbacks.append(board)
metric_to_compare = 'val_euclidean_distance'
print("Begin of training model...")
for i in range(MAX_NUM_EPOCHS):
  model.fit_generator(self.train_inputs, steps_per_epoch=self.train_inputs.steps_per_epoch(),
                      validation_data=test_input_sequence, validation_steps=steps_test,
                      max_queue_size=self.train_inputs.workers, epochs=i+1, initial_epoch=i,
                      workers=self.train_inputs.workers, use_multiprocessing=True,
                      callbacks = callbacks)
  try:
    metrics_diff = history.history[metric_to_compare][i] - min(history.history[metric_to_compare][:i])
  except:
    metrics_diff = -1
  if metrics_diff < 0:
    self._save_models(i)
    self.data_processor = None  # Empty memory
    best_epoch = i
    num_worse_epochs = 0
  elif metrics_diff > 0:
    num_worse_epochs += 1
    if num_worse_epochs >= PATIENCE:
      print("Ran out of patience. Stopping training.")
      break
print("End of training model.")

import pickle
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.metrics import categorical_accuracy

model = Sequential()
model.add(LSTM(20, return_sequences=True, stateful=False, batch_input_shape=(10, 20, 4)))
model.add(Dense(3, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=[categorical_accuracy],
              sample_weight_mode='temporal')

data_path = '/home/ubuntu/invoice/data/'     #any path to store pickle dump
output_file_path = data_path + 'model.dat'
with open(output_file_path, 'wb') as f:
    pickle.dump(model, f)