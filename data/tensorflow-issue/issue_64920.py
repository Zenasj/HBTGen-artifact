import numpy as np
import tensorflow as tf
from tensorflow.keras import models

df = pd.read_csv("SisFall_All.csv")
df.drop(columns=['acc2_x', 'acc2_y', 'acc2_z'], inplace=True)
df.set_index(df.index, inplace=True)

# Resample the DataFrame index to downsample from 200Hz to 50Hz
df = df.iloc[::4]

# Reset the index to retain the original row indices
df.reset_index(drop=True, inplace=True)

acc1_resolution = 13
acc1_range = 16

gyro_resolution = 16
gyro_range = 2000

# Convert accelerometer data to gravity units
df['acc1_x'] = (2 * acc1_range / 2**acc1_resolution) * df['acc1_x']
df['acc1_y'] = (2 * acc1_range / 2**acc1_resolution) * df['acc1_y']
df['acc1_z'] = (2 * acc1_range / 2**acc1_resolution) * df['acc1_z']

# Convert gyroscope data to degrees per second
df['gyro_x'] = (2 * gyro_range / 2**gyro_resolution) * df['gyro_x']
df['gyro_y'] = (2 * gyro_range / 2**gyro_resolution) * df['gyro_y']
df['gyro_z'] = (2 * gyro_range / 2**gyro_resolution) * df['gyro_z']

# Convert activity labels to "ADL" or "Fall" based on their first letter
df['label'] = df['label'].apply(lambda x: 'ADL' if x.startswith('D') else 'Fall' if x.startswith('F') else x)
df.drop(columns=['Subject ID_Trial'], inplace=True)

time_steps = 200
features = 6
step = 20
segments = []
labels = []
for i in range(0, len(df) - time_steps, step):
  axs = df['acc1_x'].values[i: i + time_steps]
  ays = df['acc1_y'].values[i: i + time_steps]
  azs = df['acc1_z'].values[i: i + time_steps]
  gxs = df['gyro_x'].values[i: i + time_steps]
  gys = df['gyro_y'].values[i: i + time_steps]
  gzs = df['gyro_z'].values[i: i + time_steps]
#  amags = df['acc_magnitude'].values[i: i + time_steps]

  # label_counts = np.unique(df['label'][i: i + time_steps], return_counts=True)
  # label = label_counts[0][np.argmax(label_counts[1])]
  # segments.append([axs, ays, azs, gxs, gys, gzs])
  # labels.append(label)

  label = stats.mode(df['label'][i: i + time_steps])
  label = label[0][0]
  segments.append([axs, ays, azs, gxs, gys, gzs])
  labels.append(label)

np.array(segments).shape

reshaped_segment = np.asarray(segments, dtype = np.float32).reshape(-1, time_steps, features)
reshaped_segment.shape

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

print(labels)
print(labels.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reshaped_segment, labels, test_size = 0.2, random_state = 42)

model = Sequential()
# RNN layer
model.add(LSTM(128, input_shape = (200, 6), return_sequences = True, name = 'lstm_1'))
# Apply Dense operations individually to each time sequence
model.add(TimeDistributed(Dense(64, activation='relu'), name='time_distributed'))
# Flatten layer
model.add(Flatten(name='flatten'))
# Dense layer with ReLu
model.add(Dense(64, activation='relu', name='dense_1'))
# Softmax layer
model.add(Dense(2, activation = 'softmax', name='output'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

from keras.callbacks import ModelCheckpoint

callbacks = [ModelCheckpoint('model.h5', save_weights_only=False, save_best_only=True, verbose=1)]
history = model.fit(X_train, y_train, epochs = 10, validation_split = 0.20, batch_size = 256, 
                    verbose = 1, callbacks = callbacks)

from keras.models import load_model
model = load_model('model.h5')
model.summary()

loss, accuracy = model.evaluate(X_test, y_test, batch_size = 128, verbose = 1)
print("Test Accuracy :", accuracy)
print("Test Loss :", loss)

from keras import backend as k
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

input_node_name = ['lstm_1_input']
output_node_name = 'output/Softmax'
model_name = 'fall_model'

tf.train.write_graph(k.get_session().graph_def, 'models', model_name + '_graph.pbtxt')
saver = tf.train.Saver()
saver.save(k.get_session(), 'models/' + model_name + '.chkp')

freeze_graph.freeze_graph('models/' +model_name + '_graph.pbtxt', None, False, 'models/' +model_name+'.chkp',output_node_name, 'save/restore_all', 'save/Const:0', 'models/frozen_'+model_name+'.pb', True, "")