import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

dataset, info = tfds.load(name="stanford_dogs", with_info=True)
training_data = dataset['train']
val_data = dataset['test']

IMG_LEN = 224
IMG_SHAPE = (IMG_LEN,IMG_LEN,3)
N_BREEDS = 120

def preprocess(ds_row):
    image = tf.image.convert_image_dtype(ds_row['image'], dtype=tf.float32)
    image = tf.image.resize(image, (IMG_LEN, IMG_LEN), method='nearest')
    
    label = tf.one_hot(ds_row['label'], N_BREEDS)  # TODO: Can remove one_hot and change loss function
    return image, label

def prepare(dataset, batch_size=None):
    ds = dataset.map(preprocess, num_parallel_calls=4)
    ds = ds.shuffle(buffer_size=1000)
    if batch_size:
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

train_batches = prepare(training_data, batch_size=32)
val_batches = prepare(val_data, batch_size=32)

base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(N_BREEDS, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adamax(0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

history = model.fit(train_batches,
                    epochs=30,
                    validation_data=val_batches)

# Split out test images and test labels
test_images = dataset['test'].map(
    lambda x:  (tf.image.resize(x['image'], (IMG_LEN, IMG_LEN), method='nearest'))
).batch(1)

_test_labels = dataset['test'].map(
    lambda y: (y['label'])
)
test_labels = [l.numpy() for l in _test_labels]

# Use for loop to get good predictions
_preds = []
_argmax = []
_actuals = []

for dog, label in zip(dataset['test'].take(n), test_labels[:n]):

    pic, _ = preprocess(dog)  # Convert to float32 and resize

    img_tensor = tf.expand_dims(pic,0)
    pred = model(img_tensor)
    
    am = tf.argmax(pred, axis=1)
    _argmax.append(am)

    top_components = tf.reshape(tf.math.top_k(pred, k=5).indices,shape=[-1])
    top_matches = [get_name(i) for i in top_components]
    actual = get_name(label)

    _preds.append(top_components[0])
    _actuals.append(label)

conf_matrix = tf.math.confusion_matrix(
    _actuals,
    _argmax
)

plt.figure(figsize=(20,15))
sns.heatmap(conf_matrix)
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Now using model.predict()
tf_preds = model.predict(test_images)
preds = np.array(tf.math.argmax(tf_preds, axis=1))

conf_matrix = tf.math.confusion_matrix(
    _actuals,
    preds
)

plt.figure(figsize=(20,15))
sns.heatmap(conf_matrix)
plt.ylabel('True label')
plt.xlabel('Predicted label')