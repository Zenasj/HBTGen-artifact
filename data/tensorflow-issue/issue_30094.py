from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# model architecture
inputs = tf.keras.Input(shape=(784,), name='flattened_image')
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
# creating the model
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='test1')

# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# create the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# shuffle, batch and prefetch for optimizing io reads
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64).prefetch(1024)
# create the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# batch and prefetch for optimizing io reads
val_dataset = val_dataset.batch(64).prefetch(1024)

# compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'],
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./tensorboard_logs/',
    # enabling histogram will crash the learninig in non-eager mode
    histogram_freq=1, # every epoch
    write_images=True, # visualize model weights in image form
    update_freq='batch', # this can be 'epoch' to make training faster (less logs)
)

# fit the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback],
    epochs=2,
    steps_per_epoch=50
)

# the returned "history" object holds a record of the loss values and metric values during training
print('\nhistory dict:', history.history)