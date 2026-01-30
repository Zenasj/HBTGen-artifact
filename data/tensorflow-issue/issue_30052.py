from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

# model architecture
inputs = tf.keras.Input(shape=(784,), name='flattened_image')
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='error_showcase')

# loading mnist data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# create the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# shuffle, batch and prefetch
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64).prefetch(1024)
# create the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# shuffle, batch and prefetch
val_dataset = val_dataset.batch(64).prefetch(1024)

# compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# defining checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    './checkpoints/',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# fit the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[checkpoint_callback],
)

print('\nhistory dict:', history.history)

checkpointer = keras.callbacks.ModelCheckpoint("./models/")
model.fit(train_ds, steps_per_epoch=train_steps_per_epoch, epochs=EPOCH, validation_steps=val_steps_per_epoch,
              validation_data=val_ds, callbacks=[checkpointer])
result = model.evaluate(val_ds)