from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

initial_run = True

batch_size = 1000

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.batch(batch_size).repeat()

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
            optimizer=tf.keras.optimizers.SGD(momentum=0.9),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

if not initial_run:
    model.load_weights("latest_weights")

model.fit(
        train_dataset,
        steps_per_epoch=len(train_images) / batch_size,
        epochs=1000,
        initial_epoch=int(model.optimizer.iterations.numpy() // (len(train_images) / batch_size)),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath="latest_weights",
                save_weights_only=True)])