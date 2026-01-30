import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

# TF-2.0
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# #tf.debugging.set_log_device_placement(True)

# TF-1.0
tf.compat.v1.enable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

assert tf.executing_eagerly()

batch_size = 256
num_tsteps = 144
num_features = 130
num_units = 88

n_files = 3320
#n_files = 10
num_epochs = 1000

seq_len_max_trunc = batch_size * num_tsteps
flen = 3728

X = np.random.rand(flen + 1, num_features + 2)
n_label0 = int((flen + 1) * 0.2)
Y = np.concatenate((
    np.zeros((n_label0, 1)), # label 0
    np.ones((flen - n_label0 + 1, 1)), # label 1
), axis=0)
ds_out = tf.data.Dataset.from_tensor_slices((X, Y))
ds_ser = ds_out.map(lambda *x: 
   tf.reshape(tf.py_function(lambda *v: 
       tf.train.Example(features=tf.train.Features(feature={
           "features": tf.train.Feature(float_list=tf.train.FloatList(value=v[0].numpy())),
           "label": tf.train.Feature(float_list=tf.train.FloatList(value=v[1].numpy())),
       })).SerializeToString(), x, tf.string
   ), ()), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
writer = tf.data.experimental.TFRecordWriter("temp.tfrecord")
writer.write(ds_ser)

files = ["temp.tfrecord"] * n_files

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(num_tsteps, num_features), batch_size=batch_size),
    #tf.keras.layers.Masking(mask_value=0.0, input_shape=(num_tsteps, num_features)),
    tf.keras.layers.LSTM(num_units,  batch_input_shape=(batch_size, num_tsteps, num_features), return_sequences=True, stateful=False),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
    tf.keras.layers.Activation('sigmoid'),
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


def _prep_ds_file(file):
    _ds = tf.data.TFRecordDataset(file)
    _ds = _ds.map(lambda x: tf.io.parse_single_example(x, {
        "features": tf.io.FixedLenFeature([132], tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.float32),
    }), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    _ds = _ds.flat_map(lambda v: tf.data.Dataset.from_tensors((v["features"][2:], v["label"])))

    _trunc = min(seq_len_max_trunc, ((flen + 1) // num_tsteps) * num_tsteps)
    _ds = _ds.take(_trunc)

    _c_pad = (batch_size - ((flen + 1) // num_tsteps)) * num_tsteps
    if _c_pad >= 0:
        assert _c_pad + ((flen + 1) // num_tsteps * num_tsteps) == seq_len_max_trunc
        _ds_pad = tf.data.Dataset.from_tensors((
            tf.constant(0.0, shape=[num_features,]),
            tf.constant(0.0, shape=[1,])))
        _ds_pad = _ds_pad.repeat(_c_pad)
        _ds = _ds.concatenate(_ds_pad) # pad to correct size

    _ds = _ds.window(size=num_tsteps, shift=None, stride=1, drop_remainder=True)
    _ds = _ds.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(num_tsteps), y.batch(num_tsteps))))

    _ds = _ds.batch(batch_size, drop_remainder=True)
    
    return _ds


ds_fs = tf.data.Dataset.list_files(files, shuffle=True, seed=1)
fs_train = ds_fs.take(int(n_files * 0.7))
fs_val = ds_fs.skip(int(n_files * 0.7)).take(int(n_files * 0.1))

ds_train = [_prep_ds_file(f) for f in fs_train.take(1)][0]
for f in fs_train.skip(1):
    ds_train = ds_train.concatenate(_prep_ds_file(f))
ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

ds_val = [_prep_ds_file(f) for f in fs_val.take(1)][0]
for f in fs_val.skip(1):
    ds_val = ds_val.concatenate(_prep_ds_file(f))
ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
]
model.fit(ds_train, epochs=num_epochs, verbose=1, shuffle=False,
          validation_data=ds_val, validation_steps=None, callbacks=cbs)

model.fit(ds_train, epochs=num_epochs, verbose=1, shuffle=False,
          validation_data=ds_val, validation_steps=None, callbacks=cbs)

model.fit(ds_train, epochs=num_epochs, verbose=1, shuffle=False,
          validation_data=None, validation_steps=None, callbacks=None)

import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

assert tf.executing_eagerly()

batch_size = 256
num_tsteps = 144
num_features = 130

n_files = 3320
flen = 3728

def generate_data():
    X = np.random.rand(flen + 1, num_features + 2)
    n_label0 = int((flen + 1) * 0.2)
    Y = np.concatenate((
        np.zeros((n_label0, 1)), # label 0
        np.ones((flen - n_label0 + 1, 1)), # label 1
    ), axis=0)
    ds_out = tf.data.Dataset.from_tensor_slices((X, Y))
    ds_ser = ds_out.map(lambda *x: 
       tf.reshape(tf.py_function(lambda *v: 
           tf.train.Example(features=tf.train.Features(feature={
               "features": tf.train.Feature(float_list=tf.train.FloatList(value=v[0].numpy())),
               "label": tf.train.Feature(float_list=tf.train.FloatList(value=v[1].numpy())),
           })).SerializeToString(), x, tf.string
       ), ()), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    writer = tf.data.experimental.TFRecordWriter("temp.tfrecord")
    writer.write(ds_ser)


generate_data()
files = ["temp.tfrecord"] * n_files


def _prep_ds_file(file):
    _ds = tf.data.TFRecordDataset(file)
    _ds = _ds.map(lambda x: tf.io.parse_single_example(x, {
        "features": tf.io.FixedLenFeature([132], tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.float32),
    }), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    _ds = _ds.flat_map(lambda v: tf.data.Dataset.from_tensors((v["features"][2:], v["label"])))
    
    return _ds


ds_fs = tf.data.Dataset.list_files(files, shuffle=True, seed=1)
fs_train = ds_fs.take(int(n_files * 0.7))

ds_train = [_prep_ds_file(f) for f in fs_train.take(1)][0]
for f in fs_train.skip(1):
    ds_train = ds_train.concatenate(_prep_ds_file(f))
ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


for e in ds_train.take(batch_size):
    print("The training dataset contains at least {} elements.".format(batch_size))

for i, e in enumerate(ds_train):
    print(i)
    if e > batch_size:
        break

ds_train = [_prep_ds_file(f) for f in fs_train.take(1)][0]
for f in fs_train.skip(1):
    ds_train = ds_train.concatenate(_prep_ds_file(f))
ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

ds_train = fs_train.flat_map(_prep_ds_file)
ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)