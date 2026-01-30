import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = pd.DataFrame(np.random.uniform(size=(1000,3)), columns=['Sales', 'SalesDiff7', 'SalesAggMean7'])

multi_inputs = []
multi_outputs = []
window_size = 1

for i in range(data.shape[1]):
    ti = keras.Input(shape=(window_size, 1), name=f't{i}')
    tlstm = layers.LSTM(32)(ti)
    tp = keras.layers.Dense(units=1)(tlstm)
    multi_inputs.append(ti)
    multi_outputs.append(tp)
    
r = tf.concat(multi_outputs, -1)
c = keras.layers.Flatten()(r)
result = keras.layers.Dense(units=1)(c)

n = len(data)
train_df = data[0:int(n*0.7)]
val_df = data[int(n*0.7):int(n*0.9)]
test_df = data[int(n*0.9):]

default_batch_size = 32

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test

def split_multi_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])

  # split to multi-timeseries
  inputs = tf.split(inputs, num_or_size_splits=features.shape[-1], axis=len(features.shape)-1)
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_multi_window = split_multi_window

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=default_batch_size,)

  ds = ds.map(self.split_multi_window)  # here is the problem

  return ds

WindowGenerator.make_dataset = make_dataset

single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['Sales'])

model = keras.Model(
    inputs=multi_inputs,
    outputs=result,
)
model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

history = model.fit( single_step_window.train ,  epochs=MAX_EPOCHS,
                  validation_data=single_step_window.val,
                  callbacks=[early_stopping])

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=default_batch_size,)

  def gen_ds():
    for features in ds:
      yield self.split_3d_window(ds)

  ds = tf.data.Dataset.from_generator(gen_ds, (list, tf.float32))
    
  return ds