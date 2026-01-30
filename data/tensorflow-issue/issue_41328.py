import random
from tensorflow import keras
from tensorflow.keras import layers

download_and_prepare()

batch_size = 12
n_train = 1281167
ds_train,  ds_train_info = tfds.load(name='imagenet2012', download=True, with_info=True,
                                     data_dir='/Data/tfds/', split='train', as_supervised=True,
                                     download_and_prepare_kwargs={'download_dir':'/hdd/Data/tfds/imagenet2012/',})
ds_train = ds_train.map(lambda x, y: (tf.image.resize(x, [224, 224], method='bilinear'), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds_train = iter(ds_train)

model = tf.keras.applications.VGG19(include_top=True, weights=None, input_tensor=None, input_shape=None,
                                    pooling=None, classes=1000, classifier_activation='softmax')
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
loss_instance = SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer=optimizer, loss=loss_instance)

epochs = 500
steps_per_epoch = n_train//batch_size
history = model.fit(x=ds_train, epochs=epochs,
                    steps_per_epoch=steps_per_epoch)

import tensorflow as tf
features =  tf.random.normal(shape=(100, 1, 10))
labels = tf.random.normal((100,1,1))
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
ds_iter = iter(dataset)
x = tf.keras.layers.Input(shape=[10])
y_pred = tf.keras.layers.Dense(1, activation='sigmoid', name="L0")(x)
model = tf.keras.Model(x, y_pred)
model.compile(optimizer='sgd', loss='mse',)
model.fit(ds_iter, epochs=1)