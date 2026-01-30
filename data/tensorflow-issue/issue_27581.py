from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

logging.set_verbosity(logging.INFO)
# Define the estimator's input_fn
STEPS_PER_EPOCH = 5
#BUFFER_SIZE = 10 # Use a much larger value for real code. 
BATCH_SIZE = 64
NUM_EPOCHS = 5


def input_fn():
    datasets, ds_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
    
        return image, label[..., tf.newaxis]

    train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_data.repeat()


def make_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.02),
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

model = make_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

training_dataset=input_fn()

print("train")
model.fit(training_dataset,
          steps_per_epoch=5,
          epochs=10,
          verbose = 1)

print("evaluate")
model.evaluate(training_dataset,
              steps=1)

print("predict on batch")
model.predict_on_batch(training_dataset)

model.predict_on_batch(training_dataset)