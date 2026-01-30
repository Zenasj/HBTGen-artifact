import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

filenames = [data_path]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset

feature = {'train/image': tf.io.FixedLenFeature([], tf.string),
            'train/label': tf.io.FixedLenFeature([], tf.int64),
           'train/audio': tf.io.FixedLenFeature([], tf.string),}

def _parse_function(example_proto):
  features = tf.io.parse_single_example(example_proto, feature)
  audio_raw = tf.expand_dims(tf.io.decode_raw(features['train/audio'],tf.float32),axis=-1)
  image_raw = tf.io.decode_raw(features['train/image'],tf.float32)
  image = tf.reshape(image_raw,[5,500,500,3])
  labels =features['train/label']
  labels = tf.one_hot(labels,depth=2)
  return audio_raw,image,labels

dataset = raw_dataset.map(_parse_function)
dataset

BATCH_SIZE = 4
BUFFER_SIZE = 10000

dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
dataset

am = action_model((5,500,500,3))
afm = audio_and_final_model((105000,1))

combined = tf.keras.layers.Concatenate(axis=-1)([am.output,afm.output])
z = tf.keras.layers.Dense(32,activation='relu')(combined)
dropOut = tf.keras.layers.Dropout(0.3)(z)
out = tf.keras.layers.Dense(2,activation='tanh')(dropOut)
model = tf.keras.models.Model(inputs=[am.input,afm.input],outputs=out)
model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])

# print(model.summary())
EPOCHS=30
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(verbose=1,monitor='loss',patience=2),tf.keras.callbacks.ModelCheckpoint('/content/drive/My Drive/Model.h5',monitor='loss',save_best_only=True,save_freq='epoch',save_weights_only=True),tf.keras.callbacks.TensorBoard()]
print("Starting training")
model.fit(dataset,epochs=EPOCHS,callbacks=callbacks)
print("Saving Model")
model.save("Model.h5")