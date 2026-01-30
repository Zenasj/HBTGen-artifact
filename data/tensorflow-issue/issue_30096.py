from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

## model architecture
input_layer = tf.keras.Input(shape=(28, 28, 1), name='image_input')
layer = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='initial_padding')(input_layer)
# add convolutional layer
layer = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=8,
    padding='same',
    name='conv_layer'
)(layer)
# batch normalization
layer = tf.keras.layers.BatchNormalization(axis=3, name='bn_layer')(layer)
# activation
layer = tf.keras.layers.Activation('relu', name='activation_layer')(layer)
# down sample
net = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)
# flatten
net = tf.keras.layers.Flatten(name='flatten_layer')(net)
# dense layer with ReLU-activation.
net = tf.keras.layers.Dense(64, activation='relu', name='dense_layer')(net)
# dropout layer
net = tf.keras.layers.Dropout(0.2, name='dropout_layer')(net)
# last fully-connected / dense layer with softmax-activation so it can be used for classification.
output_layer = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(net)
# creating the model
model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='test1')

# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# add channel to x; also divide by 255 to normalize the data
x_train = x_train.reshape(60000, 28, 28, 1)#.astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1)#.astype('float32') / 255

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
    metrics=['accuracy']
)

# fit the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=2,
    steps_per_epoch=50
)

# the returned "history" object holds a record of the loss values and metric values during training
print('\nhistory dict:', history.history)

model.save('mnist_model')

del model
# recreate the exact same model purely from the file:
model = tf.keras.models.load_model('mnist_model')