import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

# separate label from df
target = df.pop('label')

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# loops over entire dataset and does not produce tensors of size n features, target 
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

# runs ok
train_dataset = dataset.shuffle(len(df)).batch(1)

# define model
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae', 'acc'])
  return model

model = get_compiled_mo

model = get_compiled_model()

# returns error
model.fit(train_dataset, epochs=15)