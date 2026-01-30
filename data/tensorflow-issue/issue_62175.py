import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

out_dir = Path(r'')

image_input = tf.keras.Input(shape=[224, 224, 3])

conv1 = tf.keras.layers.SeparableConv2D(32, (3, 3))(image_input)
conv1 = tf.keras.layers.BatchNormalization()(conv1)
conv1 = tf.keras.layers.ReLU()(conv1)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = tf.keras.layers.SeparableConv2D(64, (3, 3))(pool1)
conv2 = tf.keras.layers.BatchNormalization()(conv2)
conv2 = tf.keras.layers.ReLU()(conv2)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

out = tf.keras.layers.Dense(4, activation='softmax')(pool2)

model = tf.keras.Model(image_input, out)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['categorical_accuracy'])

model.save(out_dir / 'model_tf_12_1')

model_1 = tf.saved_model.load(out_dir / 'model_tf_12_1')

np.random.seed(123) # predetermined seed
random_tensor = np.random.random([1, 224, 224, 3]).astype('float32')

pred_1 = model_1(random_tensor)

with open(out_dir / 'pred_1', mode='wb') as f:
    pickle.dump(pred_1, f)

model_1 = tf.saved_model.load(out_dir / 'model_tf_12_1')

np.random.seed(123) # predetermined seed
random_tensor = np.random.random([1, 224, 224, 3]).astype('float32')

pred_2 = model_1(random_tensor)

with open(out_dir / 'pred_1', mode='rb') as f:
    pred_1 = pickle.load(f)

np.testing.assert_allclose(pred_1, pred_2)