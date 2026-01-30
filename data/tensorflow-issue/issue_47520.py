from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import shutil
import glob
import tensorflow as tf
from tensorflow.core.util import event_pb2

workdir = 'tmp_workdir/'
shutil.rmtree(workdir, ignore_errors=True)

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

inputs = tf.keras.Input(shape=(2,))
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

model.fit(x=x, y=y, batch_size=2, epochs=2, callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir=workdir, write_steps_per_second=True, update_freq=1,
                            profile_batch=0)])

max_step = 0
for record in tf.data.TFRecordDataset(glob.glob(workdir + 'train/events.out*')[0]):
    event = event_pb2.Event.FromString(record.numpy())
    if event.HasField('summary') and event.summary.value[0].tag == 'batch_steps_per_second':
        max_step = max(max_step, event.step)

if max_step == 0:
    print("FAILURE")
else:
    print("OK")